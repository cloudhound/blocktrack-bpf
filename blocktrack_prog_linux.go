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
//go:build linux && amd64

package replication

import (
	"encoding/binary"
	"fmt"
)

// the program marks at most this many consecutive regions per request. the
// region size is chosen so a request never spans more than two, so the
// headroom only ever matters for oversized discards, which are counted as
// overflow instead
const trackUnroll = 4

const (
	// src_reg value marking an ld_imm64 as a map file descriptor reference
	bpfPseudoMapFD = 1
	// helper number of bpf_map_lookup_elem
	bpfFuncMapLookupElem = 1
)


// stats map slots
const statOverflow = 0 // requests the program cannot mark: wider than the unrolled loop, a merged discard, a passthrough

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
func andImm(dst uint8, imm int32) bpfInsn {
	return mkInsn(bpfClassALU|bpfOpAND|bpfSrcK, dst, 0, 0, imm)
}
func lshImm(dst uint8, imm int32) bpfInsn {
	return mkInsn(bpfClassALU|bpfOpLSH|bpfSrcK, dst, 0, 0, imm)
}
func rshImm(dst uint8, imm int32) bpfInsn {
	return mkInsn(bpfClassALU|bpfOpRSH|bpfSrcK, dst, 0, 0, imm)
}

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
}

func newBpfAsm() *bpfAsm {
	return &bpfAsm{jumps: map[int]string{}, labels: map[string]int{}}
}

func (a *bpfAsm) emit(ins ...bpfInsn) { a.ins = append(a.ins, ins...) }

// label names the next emitted instruction
func (a *bpfAsm) label(name string) { a.labels[name] = len(a.ins) }

// jump emits a conditional or unconditional jump to a label. imm is the
// compared immediate for the K forms; src the compared register for X forms
func (a *bpfAsm) jump(op, dst, src uint8, imm int32, target string) {
	a.jumps[len(a.ins)] = target
	a.emit(mkInsn(bpfClassJMP|op, dst, src, 0, imm))
}

func (a *bpfAsm) resolve() ([]bpfInsn, error) {
	for at, name := range a.jumps {
		to, ok := a.labels[name]
		if !ok {
			return nil, fmt.Errorf("bpf asm: unresolved label %q", name)
		}
		a.ins[at].off = int16(to - (at + 1))
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
	segments    btfField // struct request .nr_phys_segments (u16); the range count of a merged discard
	bio         btfField // struct request .bio (pointer): the first bio still to complete
	bioSector   btfField // struct bio .bi_iter.bi_sector (u64): where that bio really starts
	q           btfField // struct request .q (pointer)
	qDisk       btfField // struct request_queue .disk (pointer) from 5.15, else unset
	qBDI        btfField // struct request_queue .backing_dev_info (pointer), used only without qDisk
	bdiOwner    btfField // struct backing_dev_info .owner (pointer): the disk's struct device
	devT        btfField // struct device .devt (u32)
	major       btfField // struct gendisk .major (int)
	firstMinor  btfField // struct gendisk .first_minor (int)
	writeOps    []int32  // REQ_OP_* values that change data
	discardOps  []int32  // the subset the block layer may merge from separate ranges
	readOps     []int32  // REQ_OP_* values that never change data
	dataInOps   []int32  // driver passthrough that moves data from the device (DRV_IN, SCSI_IN)
}

// the operation codes that change what is on the disk. write-same is gone
// from newer kernels and simply absent there; zone append is refused at
// attach with the whole zoned device. discards are the codes the block layer
// merges from non-adjacent ranges into one request, whose completion reports
// only the first range and the total length
var trackWriteOpNames = []string{"REQ_OP_WRITE", "REQ_OP_DISCARD", "REQ_OP_SECURE_ERASE", "REQ_OP_WRITE_SAME", "REQ_OP_WRITE_ZEROES"}

var trackDiscardOpNames = map[string]bool{"REQ_OP_DISCARD": true, "REQ_OP_SECURE_ERASE": true}

// every other operation on the disk (driver passthrough sent through sg, bsg,
// SG_IO or the nvme ioctls, zone management, any code a newer kernel adds)
// can change data at places the program cannot see, so it is counted as
// overflow and the next pass reads in full. reads and flushes never change
// data. a passthrough that moved data from the device is a query (INQUIRY,
// the agent's own PERSISTENT RESERVE IN on every pass, smart and log pages)
// and is let through; one that moved nothing may be FORMAT UNIT, SANITIZE,
// WRITE SAME with no data-out buffer or an nvme write zeroes, so it counts.
// the older kernels name the scsi forms of the same operations
var trackReadOpNames = []string{"REQ_OP_READ", "REQ_OP_FLUSH"}

var trackDataInOpNames = []string{"REQ_OP_DRV_IN", "REQ_OP_SCSI_IN"}

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

// buildTrackProgram assembles the raw tracepoint program for one disk. the
// context is the tracepoint's argument array: args[0] the request, args[2]
// the completed byte count. loads through the request pointer are typed by
// the verifier from btf and fault-safe (a null pointer reads as zero, which
// never matches the disk). register use: r6 = ctx, r2 = request, r3 =
// operation, r9 = the disk, r7 = position then first region, r8 = last
// region; r7 and r8 survive the helper calls. the immediate compared against
// dev is sign-extended by the 64-bit jump, so the caller guarantees dev has
// bit 31 clear. nothing is counted, for any disk, before the device match, and a
// read leaves before it
func buildTrackProgram(l trackLayout, dev uint32, shift uint, regionMapFD, statsMapFD int) ([]bpfInsn, error) {
	if dev >= 1<<31 {
		return nil, fmt.Errorf("device number %#x cannot be compared by the program", dev)
	}
	if len(l.writeOps) == 0 || len(l.readOps) == 0 {
		return nil, fmt.Errorf("operation codes missing")
	}
	if l.qDisk.size == 0 && (l.qBDI.size == 0 || l.bdiOwner.size == 0 || l.devT.size == 0) {
		return nil, fmt.Errorf("no way to find a request queue's disk")
	}
	offs := map[string]int16{}
	for name, f := range map[string]btfField{"sector": l.sector, "cmd_flags": l.cmdFlags, "segments": l.segments, "bio": l.bio, "bio_sector": l.bioSector, "q": l.q, "q_disk": l.qDisk, "q_bdi": l.qBDI, "bdi_owner": l.bdiOwner, "devt": l.devT, "major": l.major, "first_minor": l.firstMinor} {
		o, err := loadOff(f)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		offs[name] = o
	}
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
	// disk runs on the admin queue, so passthrough there is judged as if it
	// were on this disk
	a.emit(ldx(bpfSizeDW, r9, r2, offs["q"]))
	if l.qDisk.size != 0 {
		a.emit(ldx(bpfSizeDW, r9, r9, offs["q_disk"]))
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "passthrough")
		a.emit(ldx(bpfSizeW, r0, r9, offs["major"]))
		a.emit(lshImm(r0, 20))
		a.emit(ldx(bpfSizeW, r1, r9, offs["first_minor"]))
		a.emit(orReg(r0, r1))
	} else {
		a.emit(ldx(bpfSizeDW, r9, r9, offs["q_bdi"]))
		a.emit(ldx(bpfSizeDW, r9, r9, offs["bdi_owner"]))
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "passthrough")
		a.emit(ldx(bpfSizeW, r0, r9, offs["devt"]))
	}
	a.jump(bpfJNE|bpfSrcK, r0, 0, int32(dev), "exit")
	for _, op := range l.writeOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "write")
	}
	a.label("passthrough")
	for _, op := range l.dataInOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "datain")
	}
	a.jump(bpfJA, 0, 0, 0, "overflow")
	// the verifier refuses unreachable code, so the block exists only when a
	// data-in code jumps to it. nr_bytes is the length of the buffer the issuer
	// declared, zero for a command with no data phase
	if len(l.dataInOps) > 0 {
		a.label("datain")
		a.emit(ldx(bpfSizeDW, r4, r6, 16)) // r4 = nr_bytes completed
		a.jump(bpfJEQ|bpfSrcK, r4, 0, 0, "overflow")
		a.jump(bpfJA, 0, 0, 0, "exit")
	}

	a.label("write")
	a.emit(ldx(bpfSizeDW, r4, r6, 16)) // r4 = nr_bytes completed
	a.jump(bpfJEQ|bpfSrcK, r4, 0, 0, "exit")

	// a discard merged from several ranges cannot be marked from its first
	// range and total length, and once part of it has completed the request's
	// position no longer says where the remaining range is. count both cases
	// as overflow: more than one range left, or a position that disagrees with
	// the first remaining bio's own sector
	for _, op := range l.discardOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "discard")
	}
	a.jump(bpfJA, 0, 0, 0, "single")
	a.label("discard")
	a.emit(ldx(bpfSizeH, r5, r2, offs["segments"]))
	a.jump(bpfJGT|bpfSrcK, r5, 0, 1, "overflow")
	a.emit(ldx(bpfSizeDW, r5, r2, offs["bio"]))
	a.emit(ldx(bpfSizeDW, r5, r5, offs["bio_sector"]))
	a.jump(bpfJNE|bpfSrcX, r5, r7, 0, "overflow")

	a.label("single")

	// r8 = last region, r7 = first region
	a.emit(rshImm(r4, 9)) // sectors
	a.emit(movReg(r8, r7))
	a.emit(addReg(r8, r4))
	a.emit(subImm(r8, 1))
	a.emit(rshImm(r8, int32(shift-9)))
	a.emit(rshImm(r7, int32(shift-9)))

	// a request wider than the unroll cannot be marked completely: count it
	// so the next pass fails closed
	a.emit(movReg(r2, r8))
	a.emit(subReg(r2, r7))
	a.jump(bpfJGT|bpfSrcK, r2, 0, trackUnroll-1, "overflow")

	for i := 0; i < trackUnroll; i++ {
		a.jump(bpfJGT|bpfSrcX, r7, r8, 0, "exit")
		a.emit(stx(bpfSizeW, r10, r7, -4))
		m := ldMapFD(r1, regionMapFD)
		a.emit(m[0], m[1])
		a.emit(movReg(r2, r10))
		a.emit(addImm(r2, -4))
		a.emit(callHelper(bpfFuncMapLookupElem))
		a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, "exit")
		a.emit(movImm(r1, 1))
		a.emit(xaddDW(r0, r1, 0))
		a.emit(addImm(r7, 1))
	}
	a.jump(bpfJA, 0, 0, 0, "exit")

	a.label("overflow")
	a.emit(stImm(bpfSizeW, r10, -4, statOverflow))
	s := ldMapFD(r1, statsMapFD)
	a.emit(s[0], s[1])
	a.emit(movReg(r2, r10))
	a.emit(addImm(r2, -4))
	a.emit(callHelper(bpfFuncMapLookupElem))
	a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, "exit")
	a.emit(movImm(r1, 1))
	a.emit(xaddDW(r0, r1, 0))

	a.label("exit")
	a.emit(movImm(r0, 0))
	a.emit(exitInsn())
	return a.resolve()
}

