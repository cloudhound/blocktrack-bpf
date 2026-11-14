// SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
//
// the CloudHound block change tracker's ebpf program and the small assembler
// that emits it. this file is offered under either the BSD 2-Clause licence
// or the GNU General Public License version 2, at the recipient's option, so
// that the program the kernel loads is GPL-compatible: from linux 6.2 the
// verifier refuses direct kernel-structure reads to programs that are not.
// the same file is published at https://github.com/cloudhound/blocktrack-bpf
// with both licence texts; keep the two copies identical. everything else in
// the agent, including the runtime that loads this program, is proprietary
// and is not covered by this notice
//
//go:build linux && (amd64 || arm64)

package replication

import (
	"encoding/binary"
	"fmt"
	"math"
)

// the program marks at most this many consecutive counters per range. a
// range of at most this many regions marks them directly; a wider one (a
// discard, write zeroes or write same reaching past the region size) is
// marked at the first level of the counter ladder where it covers no more
// than this many cells (trackLevels)
const trackUnroll = 4

// a discard's ranges are read from its bio chain: the first trackDiscardBios
// are each marked on their own, the rest up to trackDiscardChain are folded
// into one range from the lowest start to the highest end and marked once,
// and a longer chain is counted as overflow. nvme merges up to 256 ranges
// into one dataset management command (max_discard_segments 256), so a
// fragmented fstrim on nvme reaches the full count. a ladder per bio for all
// 256 made the cap program 66097 insns and its load 0.6-1.4 s on the lab
// kernels (4.18 holds the global verifier mutex for all of it); the fold
// costs a few insns per bio
const (
	trackDiscardBios  = 64
	trackDiscardChain = 256
)

const (
	// src_reg value marking an ld_imm64 as a map file descriptor reference
	bpfPseudoMapFD = 1
	// helper number of bpf_map_lookup_elem
	bpfFuncMapLookupElem = 1
)


// stats map slots
const statOverflow = 0 // requests the program cannot mark: a passthrough, a range past the tracked disk, a discard with no bio or more bios than the walk covers

// btfField is a struct member located for a load: byte offset and size
type btfField struct {
	off  uint32
	size uint32
}

// one 64-bit ebpf instruction as the kernel lays it out
type bpfInsn struct {
	op   uint8
	regs uint8 // dst in the low nibble, src in the high nibble
	off  int16
	imm  int32
}

func (i bpfInsn) encode(b []byte) {
	b[0] = i.op
	b[1] = i.regs
	binary.LittleEndian.PutUint16(b[2:], uint16(i.off))
	binary.LittleEndian.PutUint32(b[4:], uint32(i.imm))
}

func encodeInsns(ins []bpfInsn) []byte {
	out := make([]byte, 8*len(ins))
	for n, i := range ins {
		i.encode(out[n*8:])
	}
	return out
}

// opcode building blocks (class | size/op | source)
const (
	bpfClassLD  = 0x00
	bpfClassLDX = 0x01
	bpfClassST  = 0x02
	bpfClassSTX = 0x03
	bpfClassJMP = 0x05
	bpfClassALU = 0x07 // 64-bit alu

	bpfSrcK = 0x00
	bpfSrcX = 0x08

	bpfSizeW    = 0x00
	bpfSizeH    = 0x08
	bpfSizeB    = 0x10
	bpfSizeDW   = 0x18
	bpfModeIMM  = 0x00
	bpfModeMEM  = 0x60
	bpfModeXADD = 0xc0

	bpfOpADD = 0x00
	bpfOpSUB = 0x10
	bpfOpMUL = 0x20
	bpfOpOR  = 0x40
	bpfOpAND = 0x50
	bpfOpLSH = 0x60
	bpfOpRSH = 0x70
	bpfOpMOV = 0xb0

	bpfJA   = 0x00
	bpfJEQ  = 0x10
	bpfJGT  = 0x20
	bpfJNE  = 0x50
	bpfCALL = 0x80
	bpfEXIT = 0x90
)

const (
	r0 = iota
	r1
	r2
	r3
	r4
	r5
	r6
	r7
	r8
	r9
	r10
)

func mkInsn(op, dst, src uint8, off int16, imm int32) bpfInsn {
	return bpfInsn{op: op, regs: dst | src<<4, off: off, imm: imm}
}

func movReg(dst, src uint8) bpfInsn { return mkInsn(bpfClassALU|bpfOpMOV|bpfSrcX, dst, src, 0, 0) }
func movImm(dst uint8, imm int32) bpfInsn {
	return mkInsn(bpfClassALU|bpfOpMOV|bpfSrcK, dst, 0, 0, imm)
}
func addReg(dst, src uint8) bpfInsn { return mkInsn(bpfClassALU|bpfOpADD|bpfSrcX, dst, src, 0, 0) }
func addImm(dst uint8, imm int32) bpfInsn {
	return mkInsn(bpfClassALU|bpfOpADD|bpfSrcK, dst, 0, 0, imm)
}
func subReg(dst, src uint8) bpfInsn { return mkInsn(bpfClassALU|bpfOpSUB|bpfSrcX, dst, src, 0, 0) }
func subImm(dst uint8, imm int32) bpfInsn {
	return mkInsn(bpfClassALU|bpfOpSUB|bpfSrcK, dst, 0, 0, imm)
}
func orReg(dst, src uint8) bpfInsn { return mkInsn(bpfClassALU|bpfOpOR|bpfSrcX, dst, src, 0, 0) }
func mulReg(dst, src uint8) bpfInsn { return mkInsn(bpfClassALU|bpfOpMUL|bpfSrcX, dst, src, 0, 0) }
func mulImm(dst uint8, imm int32) bpfInsn {
	return mkInsn(bpfClassALU|bpfOpMUL|bpfSrcK, dst, 0, 0, imm)
}
func andReg(dst, src uint8) bpfInsn { return mkInsn(bpfClassALU|bpfOpAND|bpfSrcX, dst, src, 0, 0) }
func andImm(dst uint8, imm int32) bpfInsn {
	return mkInsn(bpfClassALU|bpfOpAND|bpfSrcK, dst, 0, 0, imm)
}
func lshImm(dst uint8, imm int32) bpfInsn {
	return mkInsn(bpfClassALU|bpfOpLSH|bpfSrcK, dst, 0, 0, imm)
}
func rshImm(dst uint8, imm int32) bpfInsn {
	return mkInsn(bpfClassALU|bpfOpRSH|bpfSrcK, dst, 0, 0, imm)
}
func rshReg(dst, src uint8) bpfInsn { return mkInsn(bpfClassALU|bpfOpRSH|bpfSrcX, dst, src, 0, 0) }

// dst = *(size *)(src + off)
func ldx(size, dst, src uint8, off int16) bpfInsn {
	return mkInsn(bpfClassLDX|bpfModeMEM|size, dst, src, off, 0)
}

// *(size *)(dst + off) = src
func stx(size, dst, src uint8, off int16) bpfInsn {
	return mkInsn(bpfClassSTX|bpfModeMEM|size, dst, src, off, 0)
}

// *(u64 *)(dst + off) += src, atomically (lock xadd on x86), so concurrent
// completions on different cpus never lose an increment
func xaddDW(dst, src uint8, off int16) bpfInsn {
	return mkInsn(bpfClassSTX|bpfModeXADD|bpfSizeDW, dst, src, off, 0)
}

// the fetch flag of an atomic add (linux 5.12): src receives the old value.
// arm64 compiles it to a fully ordered add, where the legacy add above is
// unordered against the loads and stores around it
const bpfFetch = 0x01

// *(u64 *)(dst + off) += src atomically, a fetch add when fetch
func atomicAddDW(fetch bool, dst, src uint8, off int16) bpfInsn {
	if fetch {
		return mkInsn(bpfClassSTX|bpfModeXADD|bpfSizeDW, dst, src, off, bpfFetch)
	}
	return xaddDW(dst, src, off)
}

// *(size *)(dst + off) = imm
func stImm(size, dst uint8, off int16, imm int32) bpfInsn {
	return mkInsn(bpfClassST|bpfModeMEM|size, dst, 0, off, imm)
}

func callHelper(fn int32) bpfInsn { return mkInsn(bpfClassJMP|bpfCALL, 0, 0, 0, fn) }
func exitInsn() bpfInsn          { return mkInsn(bpfClassJMP|bpfEXIT, 0, 0, 0, 0) }

// ld_imm64 of a map fd is two instruction slots; the second is all zero
func ldMapFD(dst uint8, fd int) [2]bpfInsn {
	return [2]bpfInsn{
		mkInsn(bpfClassLD|bpfModeIMM|bpfSizeDW, dst, bpfPseudoMapFD, 0, int32(fd)),
		{},
	}
}

// bpfAsm is a tiny assembler with symbolic jump targets, resolved to relative
// offsets once the whole program is emitted
type bpfAsm struct {
	ins    []bpfInsn
	jumps  map[int]string
	labels map[string]int
	dup    string // the first label named twice, refused at resolve
}

func newBpfAsm() *bpfAsm {
	return &bpfAsm{jumps: map[int]string{}, labels: map[string]int{}}
}

func (a *bpfAsm) emit(ins ...bpfInsn) { a.ins = append(a.ins, ins...) }

// label names the next emitted instruction. a name used twice would send
// every jump meant for the first to the second, so it fails the build
func (a *bpfAsm) label(name string) {
	if _, ok := a.labels[name]; ok && a.dup == "" {
		a.dup = name
	}
	a.labels[name] = len(a.ins)
}

// jump emits a conditional or unconditional jump to a label. imm is the
// compared immediate for the K forms; src the compared register for X forms
func (a *bpfAsm) jump(op, dst, src uint8, imm int32, target string) {
	a.jumps[len(a.ins)] = target
	a.emit(mkInsn(bpfClassJMP|op, dst, src, 0, imm))
}

func (a *bpfAsm) resolve() ([]bpfInsn, error) {
	if a.dup != "" {
		return nil, fmt.Errorf("bpf asm: label %q is defined twice", a.dup)
	}
	for at, name := range a.jumps {
		to, ok := a.labels[name]
		if !ok {
			return nil, fmt.Errorf("bpf asm: unresolved label %q", name)
		}
		off := to - (at + 1)
		if off < math.MinInt16 || off > math.MaxInt16 {
			return nil, fmt.Errorf("bpf asm: jump at %d to %q is %d insns away, past a jump's reach", at, name, off)
		}
		a.ins[at].off = int16(off)
	}
	return a.ins, nil
}

// trackLayout is what the program needs from the kernel's type information:
// where to find the request's position, operation and disk, which operation
// codes change data, and the tracepoint typedef to attach to
type trackLayout struct {
	attachBTFID uint32
	sector      btfField // struct request .__sector (u64)
	cmdFlags    btfField // struct request .cmd_flags (u32); the low byte is the operation
	bio         btfField // struct request .bio (pointer): the first bio still to complete
	bioSector   btfField // struct bio .bi_iter.bi_sector (u64): where that bio's remaining range starts
	bioSize     btfField // struct bio .bi_iter.bi_size (u32): the bytes of it still to complete
	bioNext     btfField // struct bio .bi_next (pointer): the next bio of the request
	q           btfField // struct request .q (pointer)
	qDisk       btfField // struct request_queue .disk (pointer) from 5.15, else unset
	qBDI        btfField // struct request_queue .backing_dev_info (pointer), used only without qDisk
	bdiOwner    btfField // struct backing_dev_info .owner (pointer): the disk's struct device
	devT        btfField // struct device .devt (u32)
	major       btfField // struct gendisk .major (int)
	firstMinor  btfField // struct gendisk .first_minor (int)
	writeOps    []int32  // REQ_OP_* values that change data
	discardOps  []int32  // the subset the block layer may merge from separate ranges, marked from their bios
	readOps     []int32  // REQ_OP_* values that never change data
	dataInOps   []int32  // driver passthrough that moves data from the device (DRV_IN, SCSI_IN)
	drvInOps    []int32  // the DRV_IN subset, which passPolicy virtio lets through
	pass        passPolicy
	scsi        scsiCmdLayout // set only with pass == passSD
	nvme        nvmeAdminLayout
	// the kernel 5.4 mode: the program is attached by name with an untyped
	// context, every read through a kernel pointer is a bpf_probe_read whose
	// result is checked, and it marks one-byte dirty flags
	// (buildTrackProgramClassic) instead of counters. attachBTFID is unused
	classic bool
}

// where an nvme admin request's controller and command are, for letting the
// queries an nvme controller's admin queue carries through (nvmeAdminFor).
// the zero value turns the check off
type nvmeAdminLayout struct {
	probeRead  int32    // helper id: bpf_probe_read_kernel, or bpf_probe_read in the kernel 5.4 program
	pdu        uint32   // sizeof(struct request): every nvme driver puts struct nvme_request first in the per-request data after it
	reqCmd     btfField // struct nvme_request .cmd (pointer to struct nvme_command, whose first byte is the opcode)
	reqCtrl    btfField // struct nvme_request .ctrl (pointer)
	ctrlAdminQ btfField // struct nvme_ctrl .admin_q (pointer)
	ctrlTagSet btfField // struct nvme_ctrl .admin_tagset (pointer)
	qTagSet    btfField // struct request_queue .tag_set (pointer)
}

func (n nvmeAdminLayout) usable() bool { return n.probeRead != 0 }

// the admin opcodes let through on an nvme controller's admin queue: Get Log
// Page, Identify, Abort, Get Features, Asynchronous Event Request and Keep
// Alive. the nvme hwmon driver's temp*_min and temp*_max reads are Get
// Features with no data (5.5+), the driver keeps an event request
// outstanding on every controller, and fabrics hosts send keep alive every
// few seconds. none of them can change the media. every other admin
// command (Format NVM, Sanitize, namespace management, vendor codes) counts,
// whatever it moved
var nvmeAdminQueryOps = []int32{0x02, 0x06, 0x08, 0x0a, 0x0c, 0x18}

// stack slots for the nvme admin check, which runs on its own path
const (
	nvmeQSlot   = -8  // the request's queue
	nvmePtrSlot = -16 // the controller, then the command pointer
	nvmeOpSlot  = -24 // the opcode, 1 byte
)

// the operation codes that change what is on the disk. write-same is gone
// from newer kernels and simply absent there; zone append is refused at
// attach with the whole zoned device. discards are the codes the block layer
// merges from non-adjacent ranges into one request, whose position and length
// say nothing about where its ranges are; each bio still carries its own
var trackWriteOpNames = []string{"REQ_OP_WRITE", "REQ_OP_DISCARD", "REQ_OP_SECURE_ERASE", "REQ_OP_WRITE_SAME", "REQ_OP_WRITE_ZEROES"}

var trackDiscardOpNames = map[string]bool{"REQ_OP_DISCARD": true, "REQ_OP_SECURE_ERASE": true}

// every other operation on the disk (driver passthrough sent through sg, bsg,
// SG_IO or the nvme ioctls, zone management, any code a newer kernel adds)
// can change data at places the program cannot see, so it is counted as
// overflow and the next pass reads in full. reads and flushes never change
// data. a read-direction passthrough (the code a command with no data phase
// takes too) is judged by the disk's passPolicy: on an sd disk only the
// commands on scsiPassRules leave, whatever their length, since a FORMAT UNIT,
// SANITIZE or WRITE SAME can be sent with a buffer from the device. the older
// kernels name the scsi forms of the same operations
var trackReadOpNames = []string{"REQ_OP_READ", "REQ_OP_FLUSH"}

var trackDataInOpNames = []string{"REQ_OP_DRV_IN", "REQ_OP_SCSI_IN"}

// stack slots for the command check, below every other slot the program uses
const (
	scsiCmdLenSlot = -8  // cmd_len, 2 bytes
	scsiPtrSlot    = -16 // the cmnd pointer on kernels where it is one
	scsiCDBSlot    = -32 // the first 16 command bytes
)

// the operation lives in the low byte of cmd_flags (REQ_OP_BITS is 8)
const reqOpMask = 0xff

// loadOff narrows a btf byte offset to the instruction's signed 16-bit
// displacement, refusing anything the encoding cannot express
func loadOff(f btfField) (int16, error) {
	if f.off > 32767 {
		return 0, fmt.Errorf("field offset %d exceeds the load displacement range", f.off)
	}
	return int16(f.off), nil
}

// trackLevels lays out the counter map for a disk of the given number of
// regions. level 0 has one counter per region; each level above has one per
// pair of cells of the level below, up to the first level of at most
// trackUnroll cells, so any range of regions fits in trackUnroll cells of
// some level. the levels sit back to back in one map: bases[l] is where level
// l starts and cells is the map's size, under 2*regions + 2*21
func trackLevels(regions uint32) (bases []uint32, cells uint32) {
	if regions == 0 {
		return nil, 0
	}
	last := regions - 1
	for l := uint(0); ; l++ {
		bases = append(bases, cells)
		cells += (last >> l) + 1
		if last>>l <= trackUnroll-1 {
			return bases, cells
		}
	}
}

// buildTrackProgram assembles the raw tracepoint program for one disk. the
// context is the tracepoint's argument array: args[0] the request, args[2]
// the completed byte count. loads through the request pointer are typed by
// the verifier from btf and fault-safe (a null pointer reads as zero, which
// never matches the disk). register use: r6 = ctx, r2 = request, r3 =
// operation, r9 = the disk then a discard's bio, r7 = start sector then
// first region, r8 = end sector then last region, and in the scsi command
// check r7 = the scsi_cmnd, r8 = the cdb pointer then cmd_len, r9 = a cdb
// byte, and in the nvme admin check r7 = the request then its
// nvme_request, r8 = the controller then the command and opcode, r9 = the
// queue, with r3 kept in its stack slot; r7, r8 and r9 survive the helper
// calls. the immediate compared
// against dev is sign-extended by the 64-bit jump, so the caller guarantees
// dev has bit 31 clear. nothing is
// counted, for any disk, before the device match, and a read leaves before it
func buildTrackProgram(l trackLayout, dev uint32, shift uint, regions uint32, regionMapFD, statsMapFD int) ([]bpfInsn, error) {
	return buildTrackProgramBanks(l, dev, shift, regions, regionMapFD, statsMapFD, nil)
}

// like buildTrackProgram, and also sets the disk's fine
// banks when bank is not nil (blocktrack_bank_linux.go). every write then
// moves the active counter up before it counts and down at every end after
// it, so the plain write and the discard blocks have their own ends: the
// shared ones are reached before the counter moves. a plain write of at most
// trackFineUnroll fine cells counts its regions and then sets its fine
// cells; a wider one, and every discard range, sets the ladder cell of each
// counter cell it counts
func buildTrackProgramBanks(l trackLayout, dev uint32, shift uint, regions uint32, regionMapFD, statsMapFD int, bank *trackBankProg) ([]bpfInsn, error) {
	if bank != nil && (bank.fshift < fineShift || bank.fshift > shift || shift-bank.fshift > trackBankMaxPerRegion) {
		return nil, fmt.Errorf("fine cell shift %d does not fit regions of shift %d", bank.fshift, shift)
	}
	offs, err := trackProgramOffsets(l, dev, shift, regions)
	if err != nil {
		return nil, err
	}
	bases, _ := trackLevels(regions)
	a := newBpfAsm()
	a.emit(movReg(r6, r1))
	a.emit(ldx(bpfSizeDW, r2, r6, 0))                   // r2 = rq
	a.emit(ldx(bpfSizeDW, r7, r2, offs["sector"]))      // r7 = rq->__sector
	a.emit(ldx(bpfSizeW, r3, r2, offs["cmd_flags"]))    // r3 = rq->cmd_flags
	a.emit(andImm(r3, reqOpMask))                      // r3 = op
	for _, op := range l.readOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "exit")
	}

	// the disk is the one the request's queue belongs to, never the disk the
	// request names: before 5.16 sg and bsg send a passthrough to this disk's
	// queue naming the sg node's disk or none. from 5.15 that is the queue's
	// gendisk, before it the device that owns the queue's backing_dev_info,
	// whose devt is the disk's. anything on another disk leaves here. a queue
	// with no disk is a controller's own: an nvme format or sanitize of this
	// disk runs on the admin queue, so passthrough there counts unless it is
	// a proven nvme admin query
	a.emit(ldx(bpfSizeDW, r9, r2, offs["q"]))
	if l.qDisk.size != 0 {
		a.emit(ldx(bpfSizeDW, r9, r9, offs["q_disk"]))
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "nodisk")
		a.emit(ldx(bpfSizeW, r0, r9, offs["major"]))
		a.emit(lshImm(r0, 20))
		a.emit(ldx(bpfSizeW, r1, r9, offs["first_minor"]))
		a.emit(orReg(r0, r1))
	} else {
		a.emit(ldx(bpfSizeDW, r9, r9, offs["q_bdi"]))
		a.emit(ldx(bpfSizeDW, r9, r9, offs["bdi_owner"]))
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "nodisk")
		a.emit(ldx(bpfSizeW, r0, r9, offs["devt"]))
	}
	a.jump(bpfJNE|bpfSrcK, r0, 0, int32(dev), "exit")
	for _, op := range l.writeOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "write")
	}
	emitTrackPassthrough(a, l, offs["q"])

	a.label("write")
	a.emit(ldx(bpfSizeDW, r4, r6, 16)) // r4 = nr_bytes completed
	a.jump(bpfJEQ|bpfSrcK, r4, 0, 0, "exit")
	for _, op := range l.discardOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "discard")
	}

	// a plain write is the one range [__sector, __sector + nr_bytes), a
	// partial byte count rounded up to a whole sector
	if bank == nil {
		a.emit(addImm(r4, 511))
		a.emit(rshImm(r4, 9))
		a.emit(movReg(r8, r7))
		a.emit(addReg(r8, r4))
		emitTrackRange(a, "p", shift, regions, bases, counterMark(regionMapFD), true, "overflow", "exit")
	} else {
		emitBankedWrite(a, shift, regions, bases, regionMapFD, statsMapFD, bank)
	}

	// the shared ends sit before the discard walk, which is longer than a
	// jump reaches at the largest geometry: every walk block below has its
	// own copy of them
	emitTrackOverflow(a, "", statsMapFD, nil)

	// a discard merged from several ranges says nothing about them in its
	// position and length, and part way through a partial completion its
	// position is only the first remaining bio's. every bio still on the
	// request carries its own remaining range, so each is marked: on a
	// partial completion that marks the rest early, which a later completion
	// of it marks again. bios past the first trackDiscardBios are marked as
	// the one range covering them all, which may take in the space between
	// them but never leaves a bio out. a discard with no bio, or a chain
	// longer than trackDiscardChain, cannot be placed and is counted
	if len(l.discardOps) > 0 {
		a.label("discard")
		mark := counterMark(regionMapFD)
		if bank != nil {
			mark = ladderMark(regionMapFD, bank)
			// a failed lookup has moved nothing, so it ends at an overflow
			// that leaves the counter alone
			emitActiveUp(a, bank, "dpreoverflow")
			a.jump(bpfJA, 0, 0, 0, "dgo")
			emitTrackOverflow(a, "dpre", statsMapFD, nil)
			a.label("dgo")
		}
		a.emit(ldx(bpfSizeDW, r9, r6, 0))
		a.emit(ldx(bpfSizeDW, r9, r9, offs["bio"]))
		// jumps only go forward (no back-edge before linux 5.3), so the walk
		// counts a missing chain at its own copy of the ends
		a.jump(bpfJNE|bpfSrcK, r9, 0, 0, "walk")
		emitTrackOverflow(a, "nobio", statsMapFD, bank)
		a.label("walk")
		for i := 0; i < trackDiscardBios; i++ {
			p := fmt.Sprintf("b%d", i)
			a.emit(ldx(bpfSizeDW, r7, r9, offs["bio_sector"]))
			a.emit(ldx(bpfSizeW, r8, r9, offs["bio_size"]))
			a.jump(bpfJEQ|bpfSrcK, r8, 0, 0, p+"next")
			a.emit(addImm(r8, 511))
			a.emit(rshImm(r8, 9))
			a.emit(addReg(r8, r7))
			emitTrackRange(a, p, shift, regions, bases, mark, false, p+"overflow", p+"next")
			a.label(p + "next")
			a.emit(ldx(bpfSizeDW, r9, r9, offs["bio_next"]))
			a.jump(bpfJNE|bpfSrcK, r9, 0, 0, p+"more")
			a.jump(bpfJA, 0, 0, 0, p+"exit")
			emitTrackOverflow(a, p, statsMapFD, bank)
			a.label(p + "more")
		}
		emitTrackFold(a, offs, shift, regions, bases, mark, statsMapFD, bank)
	}
	return a.resolve()
}

// trackProgramOffsets checks what both builds of the program need and
// returns the load displacement of every request, queue, disk and bio field
func trackProgramOffsets(l trackLayout, dev uint32, shift uint, regions uint32) (map[string]int16, error) {
	if dev >= 1<<31 {
		return nil, fmt.Errorf("device number %#x cannot be compared by the program", dev)
	}
	if regions == 0 || regions > maxRegions {
		return nil, fmt.Errorf("region count %d is outside 1..%d", regions, maxRegions)
	}
	if shift < minRegionShift || shift > maxRegionShift {
		return nil, fmt.Errorf("region shift %d is outside %d..%d", shift, minRegionShift, maxRegionShift)
	}
	if len(l.writeOps) == 0 || len(l.readOps) == 0 {
		return nil, fmt.Errorf("operation codes missing")
	}
	if l.qDisk.size == 0 && (l.qBDI.size == 0 || l.bdiOwner.size == 0 || l.devT.size == 0) {
		return nil, fmt.Errorf("no way to find a request queue's disk")
	}
	if len(l.discardOps) > 0 && (l.bio.size == 0 || l.bioSector.size == 0 || l.bioSize.size == 0 || l.bioNext.size == 0) {
		return nil, fmt.Errorf("no way to walk a discard's bios")
	}
	offs := map[string]int16{}
	for name, f := range map[string]btfField{"sector": l.sector, "cmd_flags": l.cmdFlags, "bio": l.bio, "bio_sector": l.bioSector, "bio_size": l.bioSize, "bio_next": l.bioNext, "q": l.q, "q_disk": l.qDisk, "q_bdi": l.qBDI, "bdi_owner": l.bdiOwner, "devt": l.devT, "major": l.major, "first_minor": l.firstMinor} {
		o, err := loadOff(f)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		offs[name] = o
	}
	return offs, nil
}

// emitTrackFold folds the rest of the chain from r9 into r1 = lowest start,
// r2 = highest end and r3 = highest start, without a branch per bio, and
// marks [r1, r2) once. a start at or past the highest end (a range that
// wrapped) is counted. the sums use the sign bit of a difference, which
// holds because a bio's sector is inside its disk (the block layer refuses
// one past the end at submit), far below 2^62
func emitTrackFold(a *bpfAsm, offs map[string]int16, shift uint, regions uint32, bases []uint32, mark func(a *bpfAsm, overflow, next string), statsMapFD int, bank *trackBankProg) {
	a.emit(movImm(r1, 1))
	a.emit(lshImm(r1, 62))
	a.emit(movImm(r2, 0))
	a.emit(movImm(r3, 0))
	for i := trackDiscardBios; i < trackDiscardChain; i++ {
		q := fmt.Sprintf("f%d", i)
		a.emit(ldx(bpfSizeDW, r4, r9, offs["bio_sector"]))
		a.emit(ldx(bpfSizeW, r5, r9, offs["bio_size"]))
		a.jump(bpfJEQ|bpfSrcK, r5, 0, 0, q+"next")
		a.emit(addImm(r5, 511))
		a.emit(rshImm(r5, 9))
		a.emit(addReg(r5, r4))
		// r1 += (r4 - r1) when r4 < r1
		a.emit(movReg(r0, r4), subReg(r0, r1), movReg(r7, r0), rshImm(r7, 63), mulReg(r0, r7), addReg(r1, r0))
		// r2 -= (r2 - r5) when r2 < r5
		a.emit(movReg(r0, r2), subReg(r0, r5), movReg(r7, r0), rshImm(r7, 63), mulReg(r0, r7), subReg(r2, r0))
		// r3 -= (r3 - r4) when r3 < r4
		a.emit(movReg(r0, r3), subReg(r0, r4), movReg(r7, r0), rshImm(r7, 63), mulReg(r0, r7), subReg(r3, r0))
		a.label(q + "next")
		a.emit(ldx(bpfSizeDW, r9, r9, offs["bio_next"]))
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "foldmark")
	}
	emitTrackOverflow(a, "chain", statsMapFD, bank)
	a.label("foldmark")
	// every bio past the walk was empty
	a.jump(bpfJEQ|bpfSrcK, r2, 0, 0, "foldexit")
	a.jump(bpfJGT|bpfSrcX, r2, r3, 0, "foldrange")
	a.jump(bpfJA, 0, 0, 0, "foldoverflow")
	a.label("foldrange")
	a.emit(movReg(r7, r1))
	a.emit(movReg(r8, r2))
	emitTrackRange(a, "fold", shift, regions, bases, mark, false, "foldoverflow", "foldexit")
	emitTrackOverflow(a, "fold", statsMapFD, bank)
}

// the passthrough blocks of both completion programs, for a request that is
// no read and no write: r3 is its operation on the disk's own queue, and a
// queue with no disk jumps to nodisk. on the disk's queue the disk's
// passPolicy decides (passFor): an sd disk's DRV_IN and SCSI_IN go to the
// command check, virtio_blk's DRV_IN leaves, and everything else counts. on
// a queue with no disk only a proven nvme admin query leaves. nothing leaves
// by the bytes it moved. the request is reloaded from the context, since
// r1-r5 are the helper calls' arguments
func emitTrackPassthrough(a *bpfAsm, l trackLayout, qOff int16) {
	pass := l.passFor()
	switch pass {
	case passSD:
		for _, op := range l.dataInOps {
			a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "scsiquery")
		}
	case passVirtio:
		for _, op := range l.drvInOps {
			a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "exit")
		}
	}
	a.jump(bpfJA, 0, 0, 0, "overflow")
	a.label("nodisk")
	if l.nvme.usable() {
		emitNVMeAdminPass(a, l.nvme, qOff, ldx(bpfSizeDW, r7, r6, 0))
	}
	a.jump(bpfJA, 0, 0, 0, "overflow")
	if pass == passSD {
		emitSCSIRules(a, l.scsi, ldx(bpfSizeDW, r7, r6, 0), ldx(bpfSizeDW, r7, r6, 16))
		a.jump(bpfJA, 0, 0, 0, "overflow")
	}
}

// the policy the program is built with: an sd disk whose command cannot be
// read, or a kernel with no data-in operation for the check to take, counts
func (l trackLayout) passFor() passPolicy {
	if l.pass == passSD && !(l.scsi.usable() && len(l.dataInOps) > 0) {
		return passCount
	}
	return l.pass
}

// the layout with the scsi and nvme command checks off, for a kernel that
// refuses the program carrying them: an sd disk then counts every passthrough
// as passCount does, and virtio_blk, whose rule needs no read, keeps it
func trackLayoutWithoutChecks(l trackLayout) trackLayout {
	l.scsi, l.nvme = scsiCmdLayout{}, nvmeAdminLayout{}
	if l.pass == passSD {
		l.pass = passCount
	}
	return l
}

// a passthrough on a queue with no disk that is an nvme controller's admin
// queue and carries one of nvmeAdminQueryOps leaves for exit. anything else,
// including any probe read that fails, falls through to nvmeother at the
// end of the block, where the caller counts it. the request is an
// nvme admin request only when the controller its struct nvme_request names
// has this queue as its admin queue and its admin tag set is the queue's tag
// set. either test alone could pass by chance on another driver's
// per-request data (a scsi_cmnd points at a scsi_device that names its own
// queue), so both must hold. loadRq puts the request in r7 (the completion
// programs read it from their context, the mark issue program copies it
// from r6). the layout's helper is bpf_probe_read in the kernel 5.4
// program. clobbers r0-r5 and r7-r9
func emitNVMeAdminPass(a *bpfAsm, n nvmeAdminLayout, qOff int16, loadRq bpfInsn) {
	read := func(slot int16, size int32, src uint8, off int32) {
		a.emit(movReg(r1, r10))
		a.emit(addImm(r1, int32(slot)))
		a.emit(movImm(r2, size))
		a.emit(movReg(r3, src))
		a.emit(addImm(r3, off))
		a.emit(callHelper(n.probeRead))
		a.jump(bpfJNE|bpfSrcK, r0, 0, 0, "nvmeother")
	}
	a.emit(loadRq)
	read(nvmeQSlot, 8, r7, int32(qOff))
	a.emit(ldx(bpfSizeDW, r9, r10, nvmeQSlot)) // r9 = rq->q
	a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "nvmeother")
	a.emit(addImm(r7, int32(n.pdu))) // r7 = the struct nvme_request after it
	read(nvmePtrSlot, 8, r7, int32(n.reqCtrl.off))
	a.emit(ldx(bpfSizeDW, r8, r10, nvmePtrSlot)) // r8 = its controller
	a.jump(bpfJEQ|bpfSrcK, r8, 0, 0, "nvmeother")
	read(nvmeQSlot, 8, r8, int32(n.ctrlAdminQ.off))
	a.emit(ldx(bpfSizeDW, r1, r10, nvmeQSlot)) // r1 = the controller's admin queue
	a.jump(bpfJNE|bpfSrcX, r1, r9, 0, "nvmeother")
	read(nvmePtrSlot, 8, r8, int32(n.ctrlTagSet.off))
	read(nvmeQSlot, 8, r9, int32(n.qTagSet.off))
	a.emit(ldx(bpfSizeDW, r1, r10, nvmePtrSlot)) // r1 = the controller's admin tag set
	a.emit(ldx(bpfSizeDW, r2, r10, nvmeQSlot))   // r2 = the queue's tag set
	a.jump(bpfJEQ|bpfSrcK, r1, 0, 0, "nvmeother")
	a.jump(bpfJNE|bpfSrcX, r1, r2, 0, "nvmeother")
	read(nvmePtrSlot, 8, r7, int32(n.reqCmd.off))
	a.emit(ldx(bpfSizeDW, r8, r10, nvmePtrSlot)) // r8 = the command
	a.jump(bpfJEQ|bpfSrcK, r8, 0, 0, "nvmeother")
	read(nvmeOpSlot, 1, r8, 0)
	a.emit(ldx(bpfSizeB, r8, r10, nvmeOpSlot)) // r8 = the opcode
	for _, op := range nvmeAdminQueryOps {
		a.jump(bpfJEQ|bpfSrcK, r8, 0, op, "exit")
	}
	a.label("nvmeother")
}

// counts a completion that could not be placed and ends the program, at
// labels p+"overflow" and p+"exit". with bank the end moves the active
// counter back down, after everything the write set
func emitTrackOverflow(a *bpfAsm, p string, statsMapFD int, bank *trackBankProg) {
	a.label(p + "overflow")
	a.emit(stImm(bpfSizeW, r10, -4, statOverflow))
	s := ldMapFD(r1, statsMapFD)
	a.emit(s[0], s[1])
	a.emit(movReg(r2, r10))
	a.emit(addImm(r2, -4))
	a.emit(callHelper(bpfFuncMapLookupElem))
	a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, p+"exit")
	a.emit(movImm(r1, 1))
	a.emit(xaddDW(r0, r1, 0))
	a.label(p + "exit")
	if bank != nil {
		emitActiveDown(a, bank)
	}
	a.emit(movImm(r0, 0))
	a.emit(exitInsn())
}

// emitTrackRange marks sectors [r7, r8) and continues at next, or jumps to
// overflow when the range cannot be placed; the caller
// has made the range nonempty, and an end that wrapped below the start is
// counted. the first and last regions are walked up the levels together,
// without a branch, until they are at most trackUnroll-1 cells apart: at
// each level a carry of 1 when they are further apart shifts both up a level
// and adds that level's size to the base. the cells from first to last at the
// level reached are then counted, covering the range and less than three
// times as much again. a range reaching past the regions the tracker was
// sized for is counted, since its keys would land in the next level. r5 is
// the base, r2-r4 scratch; r0-r5, r7 and r8 are clobbered. mark marks cell
// r7, clobbering r0-r5 only. with direct, a range of at most trackUnroll
// regions skips the walk and marks its level 0 cells, which is where the walk
// leaves it: every carry is 0. a plain write is at most max_sectors_kb, so
// nearly every one takes that path, and on a 100 GiB disk of 1 MiB regions
// the walk it skips is 135 instructions; a wider one still walks. the discard
// blocks keep the one path, so the verifier's states do not double for each
// of their ranges
func emitTrackRange(a *bpfAsm, p string, shift uint, regions uint32, bases []uint32, mark func(a *bpfAsm, overflow, next string), direct bool, overflow, next string) {
	a.jump(bpfJGT|bpfSrcX, r8, r7, 0, p+"nonempty")
	a.jump(bpfJA, 0, 0, 0, overflow)
	a.label(p + "nonempty")
	a.emit(subImm(r8, 1))
	a.emit(rshImm(r8, int32(shift-9)))
	a.emit(rshImm(r7, int32(shift-9)))
	a.jump(bpfJGT|bpfSrcK, r8, 0, int32(regions-1), overflow)
	a.emit(movImm(r5, 0))
	if direct && len(bases) > 1 {
		a.emit(movReg(r2, r8))
		a.emit(subReg(r2, r7))
		a.jump(bpfJGT|bpfSrcK, r2, 0, trackUnroll-1, p+"ladder")
		a.jump(bpfJA, 0, 0, 0, p+"cells")
		a.label(p + "ladder")
	}
	for lv := 0; lv+1 < len(bases); lv++ {
		a.emit(movReg(r2, r8))
		a.emit(subReg(r2, r7))
		a.emit(movImm(r3, trackUnroll-1))
		a.emit(subReg(r3, r2))
		a.emit(rshImm(r3, 63)) // 1 when last - first > trackUnroll-1
		a.emit(rshReg(r7, r3))
		a.emit(rshReg(r8, r3))
		// the level's size times the carry, never a mask of 0 - carry: the
		// verifier of newer kernels (lab 6.8.0-146 and 7.0.0-38) forks a
		// state on an and of a [-1, 0] scalar, so a mask doubled the paths at
		// every level and the cap program ran out of the 1M insn budget
		a.emit(movReg(r4, r3))
		a.emit(mulImm(r4, int32(bases[lv+1]-bases[lv])))
		a.emit(addReg(r5, r4))
	}
	// unreachable at run time (the top level has at most trackUnroll cells),
	// but nothing past this point relies on it
	a.emit(movReg(r2, r8))
	a.emit(subReg(r2, r7))
	a.jump(bpfJGT|bpfSrcK, r2, 0, trackUnroll-1, overflow)
	if direct && len(bases) > 1 {
		a.label(p + "cells")
	}
	a.emit(addReg(r7, r5))
	a.emit(addReg(r8, r5))
	for i := 0; i < trackUnroll; i++ {
		a.jump(bpfJGT|bpfSrcX, r7, r8, 0, next)
		mark(a, overflow, next)
		a.emit(addImm(r7, 1))
	}
	a.jump(bpfJA, 0, 0, 0, next)
}

// adds one to counter r7 of the counter map. a missing value (unreachable
// for an array map with the key in range) marks nothing
func counterMark(regionMapFD int) func(a *bpfAsm, overflow, next string) {
	return counterMarkWith(regionMapFD, false)
}

// counterMark with a fetch add when fetch, which orders the count before
// the bank cells the write sets after it
func counterMarkWith(regionMapFD int, fetch bool) func(a *bpfAsm, overflow, next string) {
	return func(a *bpfAsm, overflow, next string) {
		a.emit(stx(bpfSizeW, r10, r7, -4))
		m := ldMapFD(r1, regionMapFD)
		a.emit(m[0], m[1])
		a.emit(movReg(r2, r10))
		a.emit(addImm(r2, -4))
		a.emit(callHelper(bpfFuncMapLookupElem))
		a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, next)
		a.emit(movImm(r1, 1))
		a.emit(atomicAddDW(fetch, r0, r1, 0))
	}
}

// counts counter cell r7 and sets its ladder cell in both banks, so every
// region under the cell reads whole. a missing ladder element (unreachable,
// the map holds a ladder cell per counter cell) is an overflow
func ladderMark(regionMapFD int, bank *trackBankProg) func(a *bpfAsm, overflow, next string) {
	count := counterMarkWith(regionMapFD, bank.fetch)
	return func(a *bpfAsm, overflow, next string) {
		count(a, overflow, next)
		emitBankCell(a, bank, r7, bank.ladder, false, overflow)
	}
}

// stack slots of a banked write, below the command checks'
const (
	trackActiveSlot = -40 // the active counter's value pointer
	trackStartSlot  = -48 // a plain write's start sector
	trackEndSlot    = -56 // a plain write's end sector
)

// moves this cpu's active counter up and keeps its pointer for the ends; a
// failed lookup jumps to fail with nothing moved. clobbers r0-r5
func emitActiveUp(a *bpfAsm, bank *trackBankProg, fail string) {
	a.emit(stImm(bpfSizeW, r10, -4, 0))
	m := ldMapFD(r1, bank.activeFD)
	a.emit(m[0], m[1])
	a.emit(movReg(r2, r10))
	a.emit(addImm(r2, -4))
	a.emit(callHelper(bpfFuncMapLookupElem))
	a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, fail)
	a.emit(stx(bpfSizeDW, r10, r0, trackActiveSlot))
	a.emit(movImm(r1, 1))
	a.emit(atomicAddDW(bank.fetch, r0, r1, 0))
}

// moves the active counter emitActiveUp moved back down. the pointer is the
// one the up kept, so a program moved to another cpu part way still brings
// down the counter it raised. clobbers r1 and r2
func emitActiveDown(a *bpfAsm, bank *trackBankProg) {
	a.emit(ldx(bpfSizeDW, r1, r10, trackActiveSlot))
	a.emit(movImm(r2, -1))
	a.emit(atomicAddDW(bank.fetch, r1, r2, 0))
}

// sets cell idx+base of both banks to 1. with skip a cell both banks already
// hold is left as it is, so a write to a block written since the clear dirties
// no cache line. a missing element jumps to fail. clobbers r0-r5
func emitBankCell(a *bpfAsm, bank *trackBankProg, idx uint8, base uint32, skip bool, fail string) {
	a.emit(movReg(r1, idx))
	if base != 0 {
		a.emit(addImm(r1, int32(base)))
	}
	a.emit(rshImm(r1, 2))
	a.emit(stx(bpfSizeW, r10, r1, -4))
	m := ldMapFD(r1, bank.bankFD)
	a.emit(m[0], m[1])
	a.emit(movReg(r2, r10))
	a.emit(addImm(r2, -4))
	a.emit(callHelper(bpfFuncMapLookupElem))
	a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, fail)
	a.emit(movReg(r1, idx))
	if base != 0 {
		a.emit(addImm(r1, int32(base)))
	}
	a.emit(andImm(r1, 3))
	a.emit(addReg(r0, r1))
	done := fmt.Sprintf("bankset%d", len(a.ins))
	if skip {
		a.emit(ldx(bpfSizeB, r1, r0, 0))
		a.emit(ldx(bpfSizeB, r2, r0, 4))
		a.emit(andReg(r1, r2))
		a.jump(bpfJNE|bpfSrcK, r1, 0, 0, done)
	}
	a.emit(stImm(bpfSizeB, r0, 0, 1))
	a.emit(stImm(bpfSizeB, r0, 4, 1))
	if skip {
		a.label(done)
	}
}

// the plain write of a disk with banks: [r7, r7 + nr_bytes) in sectors, the
// active counter up first, then the count, then the fine or ladder cells,
// ending at its own p ends, which bring the counter down
func emitBankedWrite(a *bpfAsm, shift uint, regions uint32, bases []uint32, regionMapFD, statsMapFD int, bank *trackBankProg) {
	emitActiveUp(a, bank, "overflow")
	a.emit(ldx(bpfSizeDW, r4, r6, 16)) // r4 = nr_bytes completed, clobbered by the lookup
	a.emit(addImm(r4, 511))
	a.emit(rshImm(r4, 9))
	a.emit(movReg(r8, r7))
	a.emit(addReg(r8, r4))
	a.emit(stx(bpfSizeDW, r10, r7, trackStartSlot))
	a.emit(stx(bpfSizeDW, r10, r8, trackEndSlot))
	fs := int32(bank.fshift - 9)
	// fine cells last - first; a range that wrapped reads as a huge count and
	// takes the wide path, whose range check counts it
	a.emit(movReg(r2, r8))
	a.emit(subImm(r2, 1))
	a.emit(rshImm(r2, fs))
	a.emit(movReg(r3, r7))
	a.emit(rshImm(r3, fs))
	a.emit(subReg(r2, r3))
	a.jump(bpfJGT|bpfSrcK, r2, 0, trackFineUnroll-1, "pwide")
	emitTrackRange(a, "pn", shift, regions, bases, counterMarkWith(regionMapFD, bank.fetch), true, "poverflow", "pnfine")
	a.label("pnfine")
	a.emit(ldx(bpfSizeDW, r7, r10, trackStartSlot))
	a.emit(rshImm(r7, fs))
	a.emit(ldx(bpfSizeDW, r8, r10, trackEndSlot))
	a.emit(subImm(r8, 1))
	a.emit(rshImm(r8, fs))
	for i := 0; i < trackFineUnroll; i++ {
		a.jump(bpfJGT|bpfSrcX, r7, r8, 0, "pexit")
		emitBankCell(a, bank, r7, 0, true, "poverflow")
		a.emit(addImm(r7, 1))
	}
	a.jump(bpfJA, 0, 0, 0, "pexit")
	a.label("pwide")
	a.emit(ldx(bpfSizeDW, r7, r10, trackStartSlot))
	a.emit(ldx(bpfSizeDW, r8, r10, trackEndSlot))
	emitTrackRange(a, "pw", shift, regions, bases, ladderMark(regionMapFD, bank), true, "poverflow", "pexit")
	emitTrackOverflow(a, "p", statsMapFD, bank)
}

// the kernel 5.4 mode (trackLayout.classic). before 5.5 a raw tracepoint
// program is attached by the tracepoint's name, its context is the untyped
// u64 argument array, and a read through any kernel pointer is a
// bpf_probe_read call. that helper zero-fills its buffer and returns an error
// on a fault and when the kernel is locked down for confidentiality; a zeroed
// request would match no disk and drop its write without a trace, so every
// call's result is checked and a failure counts as overflow. maps cannot be
// mmap'd before 5.5 and a syscall lookup copies a value with a plain memcpy,
// which can tear a counter the program is adding to, so this mode keeps no
// counters in the kernel: it sets one-byte dirty flags in one of two buffers,
// chosen by a control map only userspace writes, and userspace flips the
// buffers and drains the one no program can still be writing
// (classicTracker in blocktrack_linux.go)
const (
	bpfFuncProbeRead = 4    // bpf_probe_read, the only kernel read helper before 5.5
	trackFlagChunk   = 4096 // one-byte flags per element of the flag map

	// stack slots of the kernel 5.4 program, below the command check's
	classicProbeSlot = -40 // a probe read's 8-byte buffer
	classicBaseSlot  = -48 // the active buffer's first key (index * chunks)
	classicOpSlot    = -56 // the request's operation
	classicFoldLo    = -64 // the fold's lowest start
	classicFoldEnd   = -72 // the fold's highest end
	classicFoldTop   = -80 // the fold's highest start
)

// the flag map's elements per buffer: one flag per cell of every level, and
// one more at index cells for the buffer's overflow flag. the flags of the
// last element past that are padding nothing sets
func trackFlagChunks(cells uint32) uint32 {
	return (cells + trackFlagChunk) / trackFlagChunk
}

// where a cell's flag is: the element relative to the buffer's first key,
// and the byte in it
func trackFlagAt(cell uint32) (uint32, uint32) {
	return cell / trackFlagChunk, cell % trackFlagChunk
}

// classicProbe emits dst = *(size *)(base + off) through bpf_probe_read into
// the probe slot, and jumps to fail when the read does not succeed. base is
// moved into r3 before r1 and r2 are set, so it may be any register but r3's
// own scratch use; r0-r5 are clobbered
func classicProbe(a *bpfAsm, dst, size, base uint8, off int16, fail string) {
	n := int32(8)
	if size == bpfSizeW {
		n = 4
	}
	a.emit(movReg(r3, base))
	a.emit(addImm(r3, int32(off)))
	a.emit(movReg(r1, r10))
	a.emit(addImm(r1, classicProbeSlot))
	a.emit(movImm(r2, n))
	a.emit(callHelper(bpfFuncProbeRead))
	a.jump(bpfJNE|bpfSrcK, r0, 0, 0, fail)
	a.emit(ldx(size, dst, r10, classicProbeSlot))
}

// buildTrackProgramClassic assembles the kernel 5.4 program for one disk. it
// makes the decisions buildTrackProgram makes, in the same order, and sets a
// dirty flag in the active buffer of flagMapFD (2 * trackFlagChunks(cells)
// elements of trackFlagChunk bytes) where buildTrackProgram adds to a
// counter. the active buffer is read once on entry from ctlMapFD (one u32,
// 0 or 1); a control map that cannot be read sets the overflow flag of both
// buffers. any failed probe read sets the overflow flag, before or after the
// device match. registers: r6 = ctx, r9 = the request, then its queue and
// disk, then a discard's bio; r7 = start sector then first cell; r8 = the
// operation, then the end sector then last cell. r6-r9 survive the helper
// calls; the operation and the active buffer's first key are kept in stack
// slots, since r1-r5 do not
func buildTrackProgramClassic(l trackLayout, dev uint32, shift uint, regions uint32, flagMapFD, ctlMapFD int) ([]bpfInsn, error) {
	offs, err := trackProgramOffsets(l, dev, shift, regions)
	if err != nil {
		return nil, err
	}
	if l.scsi.usable() && l.scsi.probeRead != bpfFuncProbeRead {
		return nil, fmt.Errorf("the kernel 5.4 command check calls helper %d, not bpf_probe_read", l.scsi.probeRead)
	}
	if l.nvme.usable() && l.nvme.probeRead != bpfFuncProbeRead {
		return nil, fmt.Errorf("the kernel 5.4 nvme admin check calls helper %d, not bpf_probe_read", l.nvme.probeRead)
	}
	bases, cells := trackLevels(regions)
	chunks := trackFlagChunks(cells)
	ovKey, ovByte := trackFlagAt(cells)
	mark := flagMark(flagMapFD)
	ov := func(a *bpfAsm, p string) { emitFlagOverflow(a, p, flagMapFD, ovKey, ovByte) }
	probe := classicProbe

	a := newBpfAsm()
	a.emit(movReg(r6, r1))
	a.emit(stImm(bpfSizeW, r10, -4, 0))
	c := ldMapFD(r1, ctlMapFD)
	a.emit(c[0], c[1])
	a.emit(movReg(r2, r10))
	a.emit(addImm(r2, -4))
	a.emit(callHelper(bpfFuncMapLookupElem))
	a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, "noctl")
	a.emit(ldx(bpfSizeW, r1, r0, 0))
	a.emit(andImm(r1, 1))
	a.emit(mulImm(r1, int32(chunks)))
	a.emit(stx(bpfSizeDW, r10, r1, classicBaseSlot))

	// the operation first, so a read leaves after one probe read; the sector
	// is read only for what can change data
	a.emit(ldx(bpfSizeDW, r9, r6, 0)) // r9 = rq
	probe(a, r8, bpfSizeW, r9, offs["cmd_flags"], "overflow")
	a.emit(andImm(r8, reqOpMask)) // r8 = op
	a.emit(stx(bpfSizeDW, r10, r8, classicOpSlot))
	for _, op := range l.readOps {
		a.jump(bpfJEQ|bpfSrcK, r8, 0, op, "exit")
	}
	probe(a, r7, bpfSizeDW, r9, offs["sector"], "overflow")
	// the queue's disk, as buildTrackProgram finds it
	probe(a, r9, bpfSizeDW, r9, offs["q"], "overflow")
	if l.qDisk.size != 0 {
		probe(a, r9, bpfSizeDW, r9, offs["q_disk"], "overflow")
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "nodisk")
		probe(a, r8, bpfSizeW, r9, offs["major"], "overflow")
		a.emit(lshImm(r8, 20))
		probe(a, r0, bpfSizeW, r9, offs["first_minor"], "overflow")
		a.emit(orReg(r0, r8))
	} else {
		probe(a, r9, bpfSizeDW, r9, offs["q_bdi"], "overflow")
		// a queue with no bdi has no owner, as the btf program reads it
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "nodisk")
		probe(a, r9, bpfSizeDW, r9, offs["bdi_owner"], "overflow")
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "nodisk")
		probe(a, r0, bpfSizeW, r9, offs["devt"], "overflow")
	}
	a.jump(bpfJNE|bpfSrcK, r0, 0, int32(dev), "exit")
	a.emit(ldx(bpfSizeDW, r3, r10, classicOpSlot)) // r3 = op
	for _, op := range l.writeOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "write")
	}
	emitTrackPassthrough(a, l, offs["q"])

	a.label("write")
	a.emit(ldx(bpfSizeDW, r4, r6, 16)) // r4 = nr_bytes completed
	a.jump(bpfJEQ|bpfSrcK, r4, 0, 0, "exit")
	a.emit(ldx(bpfSizeDW, r3, r10, classicOpSlot))
	for _, op := range l.discardOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "discard")
	}
	a.emit(addImm(r4, 511))
	a.emit(rshImm(r4, 9))
	a.emit(movReg(r8, r7))
	a.emit(addReg(r8, r4))
	emitTrackRange(a, "p", shift, regions, bases, mark, true, "overflow", "exit")
	ov(a, "")

	// the control map is one array slot with key 0, so this is unreachable,
	// but the verifier needs the branch and it fails closed: both buffers'
	// overflow flags are set, so whichever one the next drain reads
	// answers a full read
	a.label("noctl")
	for b := uint32(0); b < 2; b++ {
		q := fmt.Sprintf("noctl%d", b)
		a.emit(stImm(bpfSizeW, r10, -4, int32(b*chunks+ovKey)))
		m := ldMapFD(r1, flagMapFD)
		a.emit(m[0], m[1])
		a.emit(movReg(r2, r10))
		a.emit(addImm(r2, -4))
		a.emit(callHelper(bpfFuncMapLookupElem))
		a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, q)
		a.emit(stImm(bpfSizeB, r0, int16(ovByte), 1))
		a.label(q)
	}
	a.emit(movImm(r0, 0))
	a.emit(exitInsn())

	// the discard walk of buildTrackProgram, every bio read checked. jumps
	// stay forward, so each block has its own copy of the overflow end
	if len(l.discardOps) > 0 {
		a.label("discard")
		a.emit(ldx(bpfSizeDW, r9, r6, 0))
		probe(a, r9, bpfSizeDW, r9, offs["bio"], "nobiooverflow")
		a.jump(bpfJNE|bpfSrcK, r9, 0, 0, "walk")
		ov(a, "nobio")
		a.label("walk")
		for i := 0; i < trackDiscardBios; i++ {
			p := fmt.Sprintf("b%d", i)
			probe(a, r7, bpfSizeDW, r9, offs["bio_sector"], p+"overflow")
			probe(a, r8, bpfSizeW, r9, offs["bio_size"], p+"overflow")
			a.jump(bpfJEQ|bpfSrcK, r8, 0, 0, p+"next")
			a.emit(addImm(r8, 511))
			a.emit(rshImm(r8, 9))
			a.emit(addReg(r8, r7))
			emitTrackRange(a, p, shift, regions, bases, mark, false, p+"overflow", p+"next")
			a.label(p + "next")
			probe(a, r9, bpfSizeDW, r9, offs["bio_next"], p+"overflow")
			a.jump(bpfJNE|bpfSrcK, r9, 0, 0, p+"more")
			a.jump(bpfJA, 0, 0, 0, p+"exit")
			ov(a, p)
			a.label(p + "more")
		}
		emitClassicFold(a, offs, shift, regions, bases, mark, ov)
	}
	return a.resolve()
}

// emitClassicFold is emitTrackFold with every bio read checked. the probe
// reads clobber r1-r5, so the lowest start, highest end and highest start
// are kept in stack slots between bios, and each bio's start and end are in
// r7 and r8
func emitClassicFold(a *bpfAsm, offs map[string]int16, shift uint, regions uint32, bases []uint32, mark func(a *bpfAsm, overflow, next string), ov func(a *bpfAsm, p string)) {
	a.emit(movImm(r1, 1))
	a.emit(lshImm(r1, 62))
	a.emit(stx(bpfSizeDW, r10, r1, classicFoldLo))
	a.emit(movImm(r1, 0))
	a.emit(stx(bpfSizeDW, r10, r1, classicFoldEnd))
	a.emit(stx(bpfSizeDW, r10, r1, classicFoldTop))
	for i := trackDiscardBios; i < trackDiscardChain; i++ {
		q := fmt.Sprintf("f%d", i)
		classicProbe(a, r7, bpfSizeDW, r9, offs["bio_sector"], "chainoverflow")
		classicProbe(a, r8, bpfSizeW, r9, offs["bio_size"], "chainoverflow")
		a.jump(bpfJEQ|bpfSrcK, r8, 0, 0, q+"next")
		a.emit(addImm(r8, 511))
		a.emit(rshImm(r8, 9))
		a.emit(addReg(r8, r7))
		a.emit(ldx(bpfSizeDW, r1, r10, classicFoldLo))
		a.emit(ldx(bpfSizeDW, r2, r10, classicFoldEnd))
		a.emit(ldx(bpfSizeDW, r3, r10, classicFoldTop))
		// r1 += (r7 - r1) when r7 < r1
		a.emit(movReg(r0, r7), subReg(r0, r1), movReg(r4, r0), rshImm(r4, 63), mulReg(r0, r4), addReg(r1, r0))
		// r2 -= (r2 - r8) when r2 < r8
		a.emit(movReg(r0, r2), subReg(r0, r8), movReg(r4, r0), rshImm(r4, 63), mulReg(r0, r4), subReg(r2, r0))
		// r3 -= (r3 - r7) when r3 < r7
		a.emit(movReg(r0, r3), subReg(r0, r7), movReg(r4, r0), rshImm(r4, 63), mulReg(r0, r4), subReg(r3, r0))
		a.emit(stx(bpfSizeDW, r10, r1, classicFoldLo))
		a.emit(stx(bpfSizeDW, r10, r2, classicFoldEnd))
		a.emit(stx(bpfSizeDW, r10, r3, classicFoldTop))
		a.label(q + "next")
		classicProbe(a, r9, bpfSizeDW, r9, offs["bio_next"], "chainoverflow")
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "foldmark")
	}
	ov(a, "chain")
	a.label("foldmark")
	a.emit(ldx(bpfSizeDW, r1, r10, classicFoldLo))
	a.emit(ldx(bpfSizeDW, r2, r10, classicFoldEnd))
	a.emit(ldx(bpfSizeDW, r3, r10, classicFoldTop))
	// every bio past the walk was empty
	a.jump(bpfJEQ|bpfSrcK, r2, 0, 0, "foldexit")
	a.jump(bpfJGT|bpfSrcX, r2, r3, 0, "foldrange")
	a.jump(bpfJA, 0, 0, 0, "foldoverflow")
	a.label("foldrange")
	a.emit(movReg(r7, r1))
	a.emit(movReg(r8, r2))
	emitTrackRange(a, "fold", shift, regions, bases, mark, false, "foldoverflow", "foldexit")
	ov(a, "fold")
}

// sets the dirty flag of cell r7 in the active buffer: element base + (r7 >>
// 12), byte r7 & 4095, which the verifier bounds inside the element. a store
// of 1 is the same from every cpu, so racing completions need no atomics. a
// missing element (unreachable for an array map with the key in range)
// counts as overflow
func flagMark(flagMapFD int) func(a *bpfAsm, overflow, next string) {
	return func(a *bpfAsm, overflow, next string) {
		a.emit(movReg(r2, r7))
		a.emit(rshImm(r2, 12))
		a.emit(ldx(bpfSizeDW, r1, r10, classicBaseSlot))
		a.emit(addReg(r2, r1))
		a.emit(stx(bpfSizeW, r10, r2, -4))
		m := ldMapFD(r1, flagMapFD)
		a.emit(m[0], m[1])
		a.emit(movReg(r2, r10))
		a.emit(addImm(r2, -4))
		a.emit(callHelper(bpfFuncMapLookupElem))
		a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, overflow)
		a.emit(movReg(r1, r7))
		a.emit(andImm(r1, trackFlagChunk-1))
		a.emit(addReg(r0, r1))
		a.emit(stImm(bpfSizeB, r0, 0, 1))
	}
}

// sets the active buffer's overflow flag and ends the program, at labels
// p+"overflow" and p+"exit"
func emitFlagOverflow(a *bpfAsm, p string, flagMapFD int, ovKey, ovByte uint32) {
	a.label(p + "overflow")
	a.emit(ldx(bpfSizeDW, r2, r10, classicBaseSlot))
	a.emit(addImm(r2, int32(ovKey)))
	a.emit(stx(bpfSizeW, r10, r2, -4))
	m := ldMapFD(r1, flagMapFD)
	a.emit(m[0], m[1])
	a.emit(movReg(r2, r10))
	a.emit(addImm(r2, -4))
	a.emit(callHelper(bpfFuncMapLookupElem))
	a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, p+"exit")
	a.emit(stImm(bpfSizeB, r0, int16(ovByte), 1))
	a.label(p + "exit")
	a.emit(movImm(r0, 0))
	a.emit(exitInsn())
}

// where an sd request's command is, for letting the queries on scsiPassRules
// through. one layout serves every program: the mark issue program at
// block_rq_issue, and the completion programs at block_rq_complete, where the
// issuer's bytes are still in place (scsi_init_command clears only what
// follows the issuer's struct scsi_request, and before 5.18 the scsi layer
// points scsi_cmnd.cmnd at the request's command without writing the request
// itself). the zero value turns the check off, and a disk that is not an sd
// disk on blk-mq always has it off. struct scsi_cmnd's own cmnd and cmd_len
// cannot be used before 5.18: on 4.18 the scsi layer clears them, starts the
// request (the issue tracepoint) and only then points them at the command, so
// the issuer's struct scsi_request (scsi_cmnd's first member there, which the
// clear leaves) is read instead. from 5.18 the issuer writes the bytes into
// scsi_cmnd.cmnd itself
type scsiCmdLayout struct {
	probeRead int32  // helper id: bpf_probe_read_kernel, or bpf_probe_read in the kernel 5.4 program
	pdu       uint32 // sizeof(struct request): the struct scsi_cmnd after it
	cmdLen    uint32 // offset in struct scsi_cmnd of the command length (u16)
	cmd       uint32 // offset in struct scsi_cmnd of the command bytes, or of the pointer to them
	inline    bool   // cmd holds the bytes (5.18+), else it points to them
	own       uint32 // pointer form: offset in struct scsi_cmnd of the issuer's own 16 command bytes
}

func (s scsiCmdLayout) usable() bool { return s.probeRead != 0 }

// one command a program lets through on an sd disk: its opcode, its exact
// length, whether it moves data from the device or none, and the bytes that
// must hold the given value under the mask
type scsiPassRule struct {
	name   string
	op     uint8
	len    uint8
	dataIn bool
	tests  []scsiPassByte
}

type scsiPassByte struct{ at, mask, val uint8 }

// the ata pass-through fields the rules test. the protocol is byte 1 bits
// 4:1 of both forms
const (
	ataProtoMask    = 0x1e
	ataProtoNonData = 0x06 // protocol 3
	ataProtoPIOIn   = 0x08 // protocol 4, pio data-in
	ataSMART        = 0xb0
	ataSMARTStatus  = 0xda // SMART RETURN STATUS
	ataSMARTData    = 0xd0 // SMART READ DATA
	ataSMARTThresh  = 0xd1 // SMART READ ATTRIBUTE THRESHOLDS
	ataSMARTLog     = 0xd5 // SMART READ LOG
	ataSMARTLBAMid  = 0x4f // the signature every SMART command carries in lba mid and high
	ataSMARTLBAHigh = 0xc2
	ataCheckPower   = 0xe5
	ataIdentify     = 0xec
	ataIdentifyPkt  = 0xa1
	ataReadLogExt   = 0x2f
)

// the 16 byte form keeps the command in byte 14, the low feature in byte 4
// with the high one in byte 3, and lba mid and high in bytes 10 and 12. the
// 12 byte form keeps them in bytes 9, 3, 6 and 7
func ata16(proto, cmd uint8, more ...scsiPassByte) []scsiPassByte {
	return append([]scsiPassByte{{1, ataProtoMask, proto}, {14, 0xff, cmd}}, more...)
}

func ata12(proto, cmd uint8, more ...scsiPassByte) []scsiPassByte {
	return append([]scsiPassByte{{1, ataProtoMask, proto}, {9, 0xff, cmd}}, more...)
}

func smart16(feature uint8) []scsiPassByte {
	return ata16(ataProtoPIOIn, ataSMART, scsiPassByte{4, 0xff, feature}, scsiPassByte{3, 0xff, 0}, scsiPassByte{10, 0xff, ataSMARTLBAMid}, scsiPassByte{12, 0xff, ataSMARTLBAHigh})
}

func smart12(feature uint8) []scsiPassByte {
	return ata12(ataProtoPIOIn, ataSMART, scsiPassByte{3, 0xff, feature}, scsiPassByte{6, 0xff, ataSMARTLBAMid}, scsiPassByte{7, 0xff, ataSMARTLBAHigh})
}

// the commands that cannot change the media, matched exactly. the ones with
// no data phase are TEST UNIT READY (udev's re-probe, multipathd's tur
// checker, smartd) and ATA PASS-THROUGH with the non-data protocol carrying
// CHECK POWER MODE or SMART RETURN STATUS (smartd on sata). the ones that
// move data from the device are the queries udev's scsi_id, smartctl, smartd,
// sg_inq, sg_readcap and the kernel's own revalidation send (captured on
// compute engine kernels 4.18 to 7.0), READ for multipathd's readsector0
// checker, and the ata reads smartd and smartctl send through ATA
// PASS-THROUGH on a sata disk (IDENTIFY, SMART READ DATA, READ THRESHOLDS and
// READ LOG, READ LOG EXT) under the pio data-in protocol only. the data phase
// must match: a FORMAT UNIT, SANITIZE or WRITE SAME sent with a buffer from
// the device still counts, as does any command off the list (SMART WRITE
// LOG and the SCT commands written through it, EXECUTE OFFLINE, SECURITY
// ERASE, DOWNLOAD MICROCODE, DATA SET MANAGEMENT), any other protocol, any
// length but the one given and any read of the command that fails
var scsiPassRules = []scsiPassRule{
	{name: "test unit ready", op: 0x00, len: 6},
	{name: "ata pass-through(16) check power mode", op: 0x85, len: 16, tests: ata16(ataProtoNonData, ataCheckPower)},
	{name: "ata pass-through(16) smart return status", op: 0x85, len: 16, tests: ata16(ataProtoNonData, ataSMART, scsiPassByte{4, 0xff, ataSMARTStatus}, scsiPassByte{3, 0xff, 0})},
	{name: "ata pass-through(12) check power mode", op: 0xa1, len: 12, tests: ata12(ataProtoNonData, ataCheckPower)},
	{name: "ata pass-through(12) smart return status", op: 0xa1, len: 12, tests: ata12(ataProtoNonData, ataSMART, scsiPassByte{3, 0xff, ataSMARTStatus})},
	{name: "inquiry", op: 0x12, len: 6, dataIn: true},
	{name: "request sense", op: 0x03, len: 6, dataIn: true},
	{name: "mode sense(6)", op: 0x1a, len: 6, dataIn: true},
	{name: "mode sense(10)", op: 0x5a, len: 10, dataIn: true},
	{name: "read capacity(10)", op: 0x25, len: 10, dataIn: true},
	{name: "read capacity(16)", op: 0x9e, len: 16, dataIn: true, tests: []scsiPassByte{{1, 0x1f, 0x10}}},
	{name: "get lba status", op: 0x9e, len: 16, dataIn: true, tests: []scsiPassByte{{1, 0x1f, 0x12}}},
	{name: "persistent reserve in", op: 0x5e, len: 10, dataIn: true},
	{name: "report luns", op: 0xa0, len: 12, dataIn: true},
	{name: "log sense", op: 0x4d, len: 10, dataIn: true},
	{name: "receive diagnostic results", op: 0x1c, len: 6, dataIn: true},
	{name: "report target port groups", op: 0xa3, len: 12, dataIn: true, tests: []scsiPassByte{{1, 0x1f, 0x0a}}},
	{name: "report supported operation codes", op: 0xa3, len: 12, dataIn: true, tests: []scsiPassByte{{1, 0x1f, 0x0c}}},
	{name: "report supported task management functions", op: 0xa3, len: 12, dataIn: true, tests: []scsiPassByte{{1, 0x1f, 0x0d}}},
	{name: "read defect data(10)", op: 0x37, len: 10, dataIn: true},
	{name: "read defect data(12)", op: 0xb7, len: 12, dataIn: true},
	{name: "read(10)", op: 0x28, len: 10, dataIn: true},
	{name: "read(16)", op: 0x88, len: 16, dataIn: true},
	{name: "ata pass-through(16) identify device", op: 0x85, len: 16, dataIn: true, tests: ata16(ataProtoPIOIn, ataIdentify)},
	{name: "ata pass-through(16) identify packet device", op: 0x85, len: 16, dataIn: true, tests: ata16(ataProtoPIOIn, ataIdentifyPkt)},
	{name: "ata pass-through(16) smart read data", op: 0x85, len: 16, dataIn: true, tests: smart16(ataSMARTData)},
	{name: "ata pass-through(16) smart read thresholds", op: 0x85, len: 16, dataIn: true, tests: smart16(ataSMARTThresh)},
	{name: "ata pass-through(16) smart read log", op: 0x85, len: 16, dataIn: true, tests: smart16(ataSMARTLog)},
	{name: "ata pass-through(16) read log ext", op: 0x85, len: 16, dataIn: true, tests: ata16(ataProtoPIOIn, ataReadLogExt)},
	{name: "ata pass-through(12) identify device", op: 0xa1, len: 12, dataIn: true, tests: ata12(ataProtoPIOIn, ataIdentify)},
	{name: "ata pass-through(12) identify packet device", op: 0xa1, len: 12, dataIn: true, tests: ata12(ataProtoPIOIn, ataIdentifyPkt)},
	{name: "ata pass-through(12) smart read data", op: 0xa1, len: 12, dataIn: true, tests: smart12(ataSMARTData)},
	{name: "ata pass-through(12) smart read thresholds", op: 0xa1, len: 12, dataIn: true, tests: smart12(ataSMARTThresh)},
	{name: "ata pass-through(12) smart read log", op: 0xa1, len: 12, dataIn: true, tests: smart12(ataSMARTLog)},
}

// whether a command is let through: the table every program is built from,
// applied in go by the blktrace mode to the command bytes el7 records with
// each passthrough completion
func scsiPassAllowed(cdb []byte, cmdLen int, dataIn bool) bool {
	if len(cdb) == 0 {
		return false
	}
	for _, r := range scsiPassRules {
		if cdb[0] != r.op || cmdLen != int(r.len) || dataIn != r.dataIn {
			continue
		}
		ok := true
		for _, t := range r.tests {
			if int(t.at) >= len(cdb) || cdb[t.at]&t.mask != t.val {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

// the scsiquery block: a data-in passthrough on an sd disk's own queue whose
// command is on scsiPassRules leaves for exit; anything else, and any read of
// the command that fails, falls through to scsiother at the end of the block,
// where the caller counts it. loadRq puts the request in r7 and loadLen the
// bytes the command moves in r7, from r6, which the helper calls leave: the
// mark issue program reads __data_len, the completion programs the completed
// byte count of their context, which for a passthrough is the length the
// issuer declared. the command is read with the layout's probe read helper,
// since the verifier bounds btf loads to struct request. a failed read
// zero-fills its buffer and zero is TEST UNIT READY's opcode, so every read's
// result is checked and the length must match a rule exactly. uses the
// command check's stack slots, which nothing on the callers' paths to it has
// written. clobbers r0-r5 and r7-r9
func emitSCSIRules(a *bpfAsm, s scsiCmdLayout, loadRq, loadLen bpfInsn) {
	a.label("scsiquery")
	a.emit(loadRq)
	a.emit(addImm(r7, int32(s.pdu))) // r7 = the struct scsi_cmnd after it
	read := func(slot int16, size int32, src uint8, off int32) {
		a.emit(movReg(r1, r10))
		a.emit(addImm(r1, int32(slot)))
		a.emit(movImm(r2, size))
		a.emit(movReg(r3, src))
		a.emit(addImm(r3, off))
		a.emit(callHelper(s.probeRead))
		a.jump(bpfJNE|bpfSrcK, r0, 0, 0, "scsiother")
	}
	read(scsiCmdLenSlot, 2, r7, int32(s.cmdLen))
	if s.inline {
		read(scsiCDBSlot, 16, r7, int32(s.cmd))
	} else {
		// a command of 16 bytes or fewer sits in the issuer's own bytes
		// (scsi_req_init points cmd there and only a longer one is moved),
		// so a pointer anywhere else is a longer command or a layout that is
		// not what the type information says, and counts
		read(scsiPtrSlot, 8, r7, int32(s.cmd))
		a.emit(ldx(bpfSizeDW, r8, r10, scsiPtrSlot))
		a.emit(movReg(r9, r7))
		a.emit(addImm(r9, int32(s.own)))
		a.jump(bpfJNE|bpfSrcX, r8, r9, 0, "scsiother")
		read(scsiCDBSlot, 16, r8, 0)
	}
	a.emit(ldx(bpfSizeH, r8, r10, scsiCmdLenSlot)) // r8 = the command's length
	a.emit(ldx(bpfSizeB, r9, r10, scsiCDBSlot))    // r9 = its opcode
	a.emit(loadLen)                                // r7 = the bytes it moves
	for i, r := range scsiPassRules {
		next := fmt.Sprintf("scsirule%d", i+1)
		if i == len(scsiPassRules)-1 {
			next = "scsiother"
		}
		a.label(fmt.Sprintf("scsirule%d", i))
		a.jump(bpfJNE|bpfSrcK, r9, 0, int32(r.op), next)
		a.jump(bpfJNE|bpfSrcK, r8, 0, int32(r.len), next)
		if r.dataIn {
			a.jump(bpfJEQ|bpfSrcK, r7, 0, 0, next)
		} else {
			a.jump(bpfJNE|bpfSrcK, r7, 0, 0, next)
		}
		for _, t := range r.tests {
			a.emit(ldx(bpfSizeB, r1, r10, scsiCDBSlot+int16(t.at)))
			if t.mask != 0xff {
				a.emit(andImm(r1, int32(t.mask)))
			}
			a.jump(bpfJNE|bpfSrcK, r1, 0, int32(t.val), next)
		}
		a.jump(bpfJA, 0, 0, 0, "exit")
	}
	a.label("scsiother")
}
