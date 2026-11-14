// SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
//
// the CloudHound block change tracker's mark-mode ebpf programs. this file is
// offered under either the BSD 2-Clause licence or the GNU General Public
// License version 2, at the recipient's option, so that the programs the
// kernel loads are GPL-compatible. the same file is published at
// https://github.com/cloudhound/blocktrack-bpf with both licence texts; keep
// the two copies identical. everything else in the agent is proprietary and
// is not covered by this notice
//
//go:build linux && (amd64 || arm64)

package replication

import (
	"fmt"

	"golang.org/x/sys/unix"
)

// mark mode counts a write when it is issued to the device, not when it
// completes, and keeps every issued write in a map until it completes. a
// cloud snapshot taken between two marks then holds every write completed
// before the first and may hold any write issued before the second, so the
// regions counted between the marks plus the writes in flight at the first
// cover everything the second snapshot can hold that the first does not

// helper numbers of bpf_map_update_elem and bpf_map_delete_elem
const (
	bpfFuncMapUpdateElem = 2
	bpfFuncMapDeleteElem = 3
)

// the update flag that refuses to replace an existing entry
const bpfNoExist = 1

// mark-mode stats map slots. any of them moving between two marks means a
// write may have gone unrecorded, which ends the tracking session
const (
	markStatWideFull    = 0 // a wide range could be placed neither in the wide map nor in the fold tree
	markStatInflightFull = 1 // the in-flight map refused an insert
	markStatPassthrough = 2 // an operation on the disk that is neither a read nor a known write
	markStatBios        = 3 // a discard carried more bios than the walk covers
	markStatSlots       = 4
)

// a discard's ranges are read from its bio chain, at most this many
const markDiscardBios = 64

// a wide range the wide map cannot take (it is full) is counted in the fold
// tree instead: an array of markFoldLeaves*2 counters laid out as a binary
// segment tree over cells of 1 << foldShift regions, node 1 the whole disk and
// node markFoldLeaves+c cell c. a range adds one to the at most two nodes per
// level that cover its cells exactly, so each count only grows and a mark
// compares them as it does regions. a cell is wider than a region on any disk
// of more than markFoldLeaves regions, so a range counted there is reported
// rounded out to whole cells: more is read, never less
const (
	markFoldLeaves = 4096
	markFoldLevels = 13 // log2(markFoldLeaves) + 1
	markFoldNodes  = 2 * markFoldLeaves
)

// the smallest cell, in regions as a power of two, that puts the whole disk
// in markFoldLeaves cells
func markFoldShift(regions uint32) uint {
	s := uint(0)
	for (uint64(regions)+(1<<s)-1)>>s > markFoldLeaves {
		s++
	}
	return s
}

// the regions node n of the fold tree covers, first and last inclusive, or
// false for a node that is not in the tree
func markFoldNodeRegions(n uint32, foldShift uint) (uint64, uint64, bool) {
	if n == 0 || n >= markFoldNodes {
		return 0, 0, false
	}
	level := uint(0)
	for (n >> (level + 1)) != 0 {
		level++
	}
	h := uint(markFoldLevels-1) - level
	first := uint64(n-(1<<level)) << h
	last := first + (1 << h) - 1
	return first << foldShift, ((last + 1) << foldShift) - 1, true
}

// the in-flight map value: the request's hardware queue and driver tag (the
// census matches them against blk-mq's own view) and the sectors it covers,
// end exclusive
const markInflightValueSize = 24

// what the mark programs need from the kernel's type
// information on top of the completion program's layout
type markLayout struct {
	trackLayout
	issueBTFID    uint32
	issueRqArg    int      // which tracepoint argument of block_rq_issue is the request
	completeRqArg int      // the same for block_rq_complete
	dataLen       btfField // struct request .__data_len (u32): bytes still to transfer
	tag           btfField // struct request .tag (int): the driver tag
	mqHctx        btfField // struct request .mq_hctx (pointer)
	queueNum      btfField // struct blk_mq_hw_ctx .queue_num (u32)
	readOps       []int32  // operations that never change data
	plainOps      []int32  // write-class operations with one contiguous range
	rangeOps      []int32  // discard-class operations whose ranges come from the bios
	scsiQ         markSCSILayout
}

// where an sd request's command is when block_rq_issue fires, for letting the
// queries on markSCSIRules through. the zero value turns the check off, and a
// disk that is not an sd disk on blk-mq always has it off. struct scsi_cmnd's
// own cmnd and cmd_len cannot be used before 5.18: on 4.18 the scsi layer
// clears them, starts the request (the tracepoint) and only then points them
// at the command, so the issuer's struct scsi_request (scsi_cmnd's first
// member there, which the clear leaves) is read instead. from 5.18 the issuer
// writes the bytes into scsi_cmnd.cmnd itself
type markSCSILayout struct {
	probeRead int32  // helper id of bpf_probe_read_kernel
	pdu       uint32 // sizeof(struct request): the struct scsi_cmnd after it
	cmdLen    uint32 // offset in struct scsi_cmnd of the command length (u16)
	cmd       uint32 // offset in struct scsi_cmnd of the command bytes, or of the pointer to them
	inline    bool   // cmd holds the bytes (5.18+), else it points to them
	own       uint32 // pointer form: offset in struct scsi_cmnd of the issuer's own 16 command bytes
}

func (s markSCSILayout) usable() bool { return s.probeRead != 0 }

// one command a mark program lets through on an sd disk: its opcode, its
// exact length, whether it moves data from the device (__data_len != 0) or
// none, and the bytes that must hold the given value under the mask
type markSCSIRule struct {
	name   string
	op     uint8
	len    uint8
	dataIn bool
	tests  []markSCSIByte
}

type markSCSIByte struct{ at, mask, val uint8 }

// the commands that cannot change the media, matched exactly. the ones with
// no data phase are TEST UNIT READY (udev's re-probe, multipathd's tur
// checker, smartd) and ATA PASS-THROUGH with the non-data protocol carrying
// CHECK POWER MODE or SMART RETURN STATUS (smartd on sata). the ones that
// move data from the device are the queries udev's scsi_id, smartctl, smartd,
// sg_inq, sg_readcap and the kernel's own revalidation send (captured on
// compute engine kernels 4.18 to 7.0), plus READ for multipathd's
// readsector0 checker. the data phase must match: a FORMAT UNIT, SANITIZE or
// WRITE SAME sent with a buffer from the device still counts, as does any
// command off the list, any length but the one given and any read of the
// command that fails
var markSCSIRules = []markSCSIRule{
	{name: "test unit ready", op: 0x00, len: 6},
	{name: "ata pass-through(16) check power mode", op: 0x85, len: 16, tests: []markSCSIByte{{1, ataProtoMask, ataProtoNonData}, {14, 0xff, ataCheckPower}}},
	{name: "ata pass-through(16) smart return status", op: 0x85, len: 16, tests: []markSCSIByte{{1, ataProtoMask, ataProtoNonData}, {14, 0xff, ataSMART}, {4, 0xff, ataSMARTStatus}, {3, 0xff, 0}}},
	{name: "ata pass-through(12) check power mode", op: 0xa1, len: 12, tests: []markSCSIByte{{1, ataProtoMask, ataProtoNonData}, {9, 0xff, ataCheckPower}}},
	{name: "ata pass-through(12) smart return status", op: 0xa1, len: 12, tests: []markSCSIByte{{1, ataProtoMask, ataProtoNonData}, {9, 0xff, ataSMART}, {3, 0xff, ataSMARTStatus}}},
	{name: "inquiry", op: 0x12, len: 6, dataIn: true},
	{name: "request sense", op: 0x03, len: 6, dataIn: true},
	{name: "mode sense(6)", op: 0x1a, len: 6, dataIn: true},
	{name: "mode sense(10)", op: 0x5a, len: 10, dataIn: true},
	{name: "read capacity(10)", op: 0x25, len: 10, dataIn: true},
	{name: "read capacity(16)", op: 0x9e, len: 16, dataIn: true, tests: []markSCSIByte{{1, 0x1f, 0x10}}},
	{name: "get lba status", op: 0x9e, len: 16, dataIn: true, tests: []markSCSIByte{{1, 0x1f, 0x12}}},
	{name: "persistent reserve in", op: 0x5e, len: 10, dataIn: true},
	{name: "report luns", op: 0xa0, len: 12, dataIn: true},
	{name: "log sense", op: 0x4d, len: 10, dataIn: true},
	{name: "receive diagnostic results", op: 0x1c, len: 6, dataIn: true},
	{name: "report target port groups", op: 0xa3, len: 12, dataIn: true, tests: []markSCSIByte{{1, 0x1f, 0x0a}}},
	{name: "report supported operation codes", op: 0xa3, len: 12, dataIn: true, tests: []markSCSIByte{{1, 0x1f, 0x0c}}},
	{name: "report supported task management functions", op: 0xa3, len: 12, dataIn: true, tests: []markSCSIByte{{1, 0x1f, 0x0d}}},
	{name: "read defect data(10)", op: 0x37, len: 10, dataIn: true},
	{name: "read defect data(12)", op: 0xb7, len: 12, dataIn: true},
	{name: "read(10)", op: 0x28, len: 10, dataIn: true},
	{name: "read(16)", op: 0x88, len: 16, dataIn: true},
}

// whether a mark program lets a command through: the same table the program
// is built from, for blktrace-free tests and the shared-table proof
func markSCSIAllowed(cdb []byte, cmdLen int, dataIn bool) bool {
	if len(cdb) == 0 {
		return false
	}
	for _, r := range markSCSIRules {
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
// command is on markSCSIRules leaves for exit; anything else, and any read of
// the command that fails, falls through to scsiother at the end of the block,
// where the caller counts it. the command is read with bpf_probe_read_kernel,
// since the verifier bounds btf loads to struct request. a failed read
// zero-fills its buffer and zero is TEST UNIT READY's opcode, so every read's
// result is checked and the length must match a rule exactly. r6 is the
// request. uses the stack slots the write path does, which nothing on this
// path has written. clobbers r0-r5 and r7-r9
func emitMarkSCSIPass(a *bpfAsm, s markSCSILayout, dataLenOff int16) {
	a.label("scsiquery")
	a.emit(movReg(r7, r6))
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
	a.emit(ldx(bpfSizeW, r7, r6, dataLenOff))      // r7 = the bytes it moves
	for i, r := range markSCSIRules {
		next := fmt.Sprintf("scsirule%d", i+1)
		if i == len(markSCSIRules)-1 {
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

// the operations a mark program lets through as reads. anything on the disk
// that is in none of the three sets moves the passthrough witness: driver
// passthrough (DRV_IN/DRV_OUT), zone management and any code a newer kernel
// adds
var markReadOpNames = []string{"REQ_OP_READ", "REQ_OP_FLUSH"}

var markPlainOpNames = []string{"REQ_OP_WRITE", "REQ_OP_WRITE_SAME", "REQ_OP_WRITE_ZEROES"}

var markRangeOpNames = []string{"REQ_OP_DISCARD", "REQ_OP_SECURE_ERASE"}

// matches the disk the request's queue belongs to, never the disk the
// request names (before 5.16 sg and bsg name the sg node's disk or none):
// the queue's gendisk from 5.15, before it the device that owns the queue's
// backing_dev_info. leaves for exit when that disk is not dev, or for nodisk
// when the queue has no disk (a controller's own queue, such as nvme admin).
// rq is the register holding the request; r0, r1 and r9 are clobbered
func emitDiskMatch(a *bpfAsm, l trackLayout, offs map[string]int16, rq uint8, dev uint32, exit, nodisk string) {
	a.emit(ldx(bpfSizeDW, r9, rq, offs["q"]))
	if l.qDisk.size != 0 {
		a.emit(ldx(bpfSizeDW, r9, r9, offs["q_disk"]))
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, nodisk)
		a.emit(ldx(bpfSizeW, r0, r9, offs["major"]))
		a.emit(lshImm(r0, 20))
		a.emit(ldx(bpfSizeW, r1, r9, offs["first_minor"]))
		a.emit(orReg(r0, r1))
	} else {
		a.emit(ldx(bpfSizeDW, r9, r9, offs["q_bdi"]))
		a.emit(ldx(bpfSizeDW, r9, r9, offs["bdi_owner"]))
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, nodisk)
		a.emit(ldx(bpfSizeW, r0, r9, offs["devt"]))
	}
	a.jump(bpfJNE|bpfSrcK, r0, 0, int32(dev), exit)
}

// adds one to a stats slot and jumps to next. clobbers r0-r5
func emitStat(a *bpfAsm, statsMapFD int, slot int32, next string) {
	a.emit(stImm(bpfSizeW, r10, -4, slot))
	m := ldMapFD(r1, statsMapFD)
	a.emit(m[0], m[1])
	a.emit(movReg(r2, r10))
	a.emit(addImm(r2, -4))
	a.emit(callHelper(bpfFuncMapLookupElem))
	a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, next)
	a.emit(movImm(r1, 1))
	a.emit(xaddDW(r0, r1, 0))
	a.jump(bpfJA, 0, 0, 0, next)
}

// records sectors [r7, r8) and continues at next. a range of
// at most trackUnroll regions adds one to each region's counter; a wider one
// adds one to its own counter in the wide map, keyed by its first and last
// region, so the same range issued again is the same entry with a higher
// count. userspace never deletes from the wide map, so a count can only
// grow and a mark compares counts exactly as it does for regions. a wide
// range the map cannot take widens the fold hull at fp-64 (lowest first
// region) and fp-72 (highest last region) and continues at full, where the
// hull is counted in the fold tree once per request (emitMarkFold). an empty
// range records nothing. r7 and r8 are clobbered, as are r0-r5
func emitMarkRange(a *bpfAsm, p string, shift uint, regionMapFD, wideMapFD int, next, full string) {
	a.jump(bpfJGT|bpfSrcX, r8, r7, 0, p+"nonempty")
	a.jump(bpfJA, 0, 0, 0, next)
	a.label(p + "nonempty")
	a.emit(subImm(r8, 1))
	a.emit(rshImm(r8, int32(shift-9)))
	a.emit(rshImm(r7, int32(shift-9)))
	a.emit(movReg(r2, r8))
	a.emit(subReg(r2, r7))
	a.jump(bpfJGT|bpfSrcK, r2, 0, trackUnroll-1, p+"wide")
	for i := 0; i < trackUnroll; i++ {
		a.jump(bpfJGT|bpfSrcX, r7, r8, 0, next)
		a.emit(stx(bpfSizeW, r10, r7, -4))
		m := ldMapFD(r1, regionMapFD)
		a.emit(m[0], m[1])
		a.emit(movReg(r2, r10))
		a.emit(addImm(r2, -4))
		a.emit(callHelper(bpfFuncMapLookupElem))
		// a region past the map is beyond the disk the tracker was sized
		// for: the size check on every mark turns that into a full read
		a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, next)
		a.emit(movImm(r1, 1))
		a.emit(xaddDW(r0, r1, 0))
		a.emit(addImm(r7, 1))
	}
	a.jump(bpfJA, 0, 0, 0, next)

	a.label(p + "wide")
	// key = first << 32 | last; regions fit in 32 bits (maxRegions)
	a.emit(lshImm(r7, 32))
	a.emit(orReg(r7, r8))
	a.emit(stx(bpfSizeDW, r10, r7, -48))
	emitWideLookup(a, wideMapFD)
	a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, p+"insert")
	a.emit(movImm(r1, 1))
	a.emit(xaddDW(r0, r1, 0))
	a.jump(bpfJA, 0, 0, 0, next)
	// new range: insert it with a count of one, only if no other cpu has
	// just done the same, in which case that entry is counted instead
	a.label(p + "insert")
	a.emit(stImm(bpfSizeDW, r10, -56, 1))
	w := ldMapFD(r1, wideMapFD)
	a.emit(w[0], w[1])
	a.emit(movReg(r2, r10))
	a.emit(addImm(r2, -48))
	a.emit(movReg(r3, r10))
	a.emit(addImm(r3, -56))
	a.emit(movImm(r4, bpfNoExist))
	a.emit(callHelper(bpfFuncMapUpdateElem))
	a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, next)
	a.jump(bpfJNE|bpfSrcK, r0, 0, -int32(unix.EEXIST), p+"full")
	emitWideLookup(a, wideMapFD)
	a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, p+"full")
	a.emit(movImm(r1, 1))
	a.emit(xaddDW(r0, r1, 0))
	a.jump(bpfJA, 0, 0, 0, next)
	a.label(p + "full")
	a.emit(ldx(bpfSizeDW, r1, r10, -48))
	a.emit(movReg(r2, r1))
	a.emit(rshImm(r2, 32)) // first
	a.emit(lshImm(r1, 32))
	a.emit(rshImm(r1, 32)) // last
	a.emit(ldx(bpfSizeDW, r3, r10, -64))
	a.jump(bpfJGT|bpfSrcX, r2, r3, 0, p+"keeplo")
	a.emit(stx(bpfSizeDW, r10, r2, -64))
	a.label(p + "keeplo")
	a.emit(ldx(bpfSizeDW, r3, r10, -72))
	a.jump(bpfJGT|bpfSrcX, r3, r1, 0, p+"keephi")
	a.emit(stx(bpfSizeDW, r10, r1, -72))
	a.label(p + "keephi")
	a.jump(bpfJA, 0, 0, 0, full)
}

// counts the fold hull at fp-64/fp-72 in the fold tree and leaves for exit.
// an empty hull (lowest above highest, as set before counting) counts
// nothing. a hull past the tree's last cell is a range beyond the disk the
// tracker was sized for and moves the wide witness; the size check on every
// mark catches a grown disk the same way. r7-r9 and r0-r5 are clobbered
func emitMarkFold(a *bpfAsm, foldShift uint, foldMapFD, statsMapFD int, exit string) {
	a.label("fold")
	a.emit(ldx(bpfSizeDW, r7, r10, -64))
	a.emit(ldx(bpfSizeDW, r8, r10, -72))
	a.jump(bpfJGT|bpfSrcX, r7, r8, 0, exit)
	if foldShift > 0 {
		a.emit(rshImm(r7, int32(foldShift)))
		a.emit(rshImm(r8, int32(foldShift)))
	}
	a.jump(bpfJGT|bpfSrcK, r8, 0, markFoldLeaves-1, "foldpast")
	// l = first leaf, r = one past the last leaf, walked up a level at a time
	a.emit(addImm(r7, markFoldLeaves))
	a.emit(addImm(r8, markFoldLeaves+1))
	// adds r9 (0 or 1) to node reg. the add of 0 keeps the walk free of
	// branches, so the verifier sees one path per level
	add := func(reg uint8) {
		a.emit(stx(bpfSizeW, r10, reg, -4))
		m := ldMapFD(r1, foldMapFD)
		a.emit(m[0], m[1])
		a.emit(movReg(r2, r10))
		a.emit(addImm(r2, -4))
		a.emit(callHelper(bpfFuncMapLookupElem))
		a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, "foldpast")
		a.emit(xaddDW(r0, r9, 0))
	}
	for i := 0; i < markFoldLevels; i++ {
		p := fmt.Sprintf("f%d", i)
		a.jump(bpfJGT|bpfSrcX, r8, r7, 0, p+"go")
		a.jump(bpfJA, 0, 0, 0, exit)
		a.label(p + "go")
		// an odd l is a right child: count it and step past it
		a.emit(movReg(r9, r7))
		a.emit(andImm(r9, 1))
		add(r7)
		a.emit(addReg(r7, r9))
		// an odd r means r-1 is a left child: count it and step before it.
		// the node r-1 is in the tree whatever r is, so the add of 0 lands
		// inside it
		a.emit(movReg(r9, r8))
		a.emit(andImm(r9, 1))
		a.emit(subImm(r8, 1))
		add(r8)
		a.emit(addImm(r8, 1))
		a.emit(subReg(r8, r9))
		a.emit(rshImm(r7, 1))
		a.emit(rshImm(r8, 1))
	}
	a.jump(bpfJA, 0, 0, 0, exit)
	// a node lookup cannot fail inside the tree; one that does is counted
	// as a range that was not placed
	a.label("foldpast")
	emitStat(a, statsMapFD, markStatWideFull, exit)
}

// looks up the wide map entry keyed at fp-48 into r0
func emitWideLookup(a *bpfAsm, wideMapFD int) {
	w := ldMapFD(r1, wideMapFD)
	a.emit(w[0], w[1])
	a.emit(movReg(r2, r10))
	a.emit(addImm(r2, -48))
	a.emit(callHelper(bpfFuncMapLookupElem))
}

func markOffsets(l markLayout) (map[string]int16, error) {
	offs := map[string]int16{}
	for name, f := range map[string]btfField{"sector": l.sector, "cmd_flags": l.cmdFlags, "bio": l.bio, "bio_sector": l.bioSector, "q": l.q, "q_disk": l.qDisk, "q_bdi": l.qBDI, "bdi_owner": l.bdiOwner, "devt": l.devT, "major": l.major, "first_minor": l.firstMinor, "data_len": l.dataLen, "tag": l.tag, "mq_hctx": l.mqHctx, "queue_num": l.queueNum, "bio_next": l.bioNext, "bio_size": l.bioSize} {
		o, err := loadOff(f)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		offs[name] = o
	}
	return offs, nil
}

// assembles the block_rq_issue program for one disk.
// for a write-class request on the disk it first records the request in the
// in-flight map (keyed by the request's address, which is the same at its
// completion), then counts its ranges. the order matters: userspace reads the
// counters before the in-flight map, so a write counted before the counter
// read is either still in the map when the map is read or already complete.
// stack: -8 key (request address), -32..-9 in-flight value, -48 wide key, -56
// wide value, -64/-72 fold hull, -4 map index. registers: r6 = request, r7/r8
// = range, r9 = the bio being walked (and the disk during the match)
func buildMarkIssueProgram(l markLayout, dev uint32, shift, foldShift uint, regionMapFD, wideMapFD, inflightMapFD, statsMapFD, foldMapFD int) ([]bpfInsn, error) {
	if dev >= 1<<31 {
		return nil, fmt.Errorf("device number %#x cannot be compared by the program", dev)
	}
	if len(l.plainOps) == 0 || len(l.rangeOps) == 0 || len(l.readOps) == 0 {
		return nil, fmt.Errorf("operation codes missing")
	}
	if l.issueRqArg < 0 || l.issueRqArg > 3 {
		return nil, fmt.Errorf("request argument %d", l.issueRqArg)
	}
	if foldShift > 20 {
		return nil, fmt.Errorf("fold cell shift %d", foldShift)
	}
	offs, err := markOffsets(l)
	if err != nil {
		return nil, err
	}
	a := newBpfAsm()
	a.emit(ldx(bpfSizeDW, r6, r1, int16(8*l.issueRqArg))) // r6 = rq
	// every request on this disk's queue is judged as this disk's, whatever
	// disk it names. a queue with no disk (an nvme controller's admin queue,
	// which carries format and sanitize) can still change this disk's data,
	// so anything but a read or a proven nvme admin query there moves the
	// witness
	emitDiskMatch(a, l.trackLayout, offs, r6, dev, "exit", "nodisk")
	a.emit(ldx(bpfSizeW, r3, r6, offs["cmd_flags"]))
	a.emit(andImm(r3, reqOpMask))
	for _, op := range l.readOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "exit")
	}
	for _, op := range l.plainOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "plain")
	}
	for _, op := range l.rangeOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "ranges")
	}
	// on an sd disk a read-direction passthrough (the code a no-data command
	// takes too) carrying a command on markSCSIRules leaves as well. reads,
	// writes and ranges have gone by now, so they run none of it
	scsi := l.scsiQ.usable() && len(l.dataInOps) > 0
	if scsi {
		for _, op := range l.dataInOps {
			a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "scsiquery")
		}
	}
	emitStat(a, statsMapFD, markStatPassthrough, "exit")
	if scsi {
		emitMarkSCSIPass(a, l.scsiQ, offs["data_len"])
		emitStat(a, statsMapFD, markStatPassthrough, "exit")
	}

	// plain write: one range, __sector for __data_len bytes
	a.label("plain")
	a.emit(ldx(bpfSizeDW, r7, r6, offs["sector"]))
	a.emit(ldx(bpfSizeW, r8, r6, offs["data_len"]))
	a.emit(rshImm(r8, 9))
	a.jump(bpfJEQ|bpfSrcK, r8, 0, 0, "exit")
	a.emit(addReg(r8, r7))
	a.emit(stx(bpfSizeDW, r10, r7, -24))
	a.emit(stx(bpfSizeDW, r10, r8, -16))
	a.jump(bpfJA, 0, 0, 0, "inflight")

	// discard: the hull of its bios goes in the map. a chain longer than the
	// walk moves the witness and records nothing more
	a.label("ranges")
	a.emit(ldx(bpfSizeDW, r9, r6, offs["bio"]))
	a.emit(movImm(r7, -1)) // lowest start
	a.emit(movImm(r8, 0))  // highest end
	for i := 0; i < markDiscardBios; i++ {
		p := fmt.Sprintf("h%d", i)
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "hulled")
		a.emit(ldx(bpfSizeDW, r4, r9, offs["bio_sector"]))
		a.emit(ldx(bpfSizeW, r5, r9, offs["bio_size"]))
		a.emit(rshImm(r5, 9))
		a.emit(addReg(r5, r4))
		a.jump(bpfJGT|bpfSrcX, r4, r7, 0, p+"lo")
		a.emit(movReg(r7, r4))
		a.label(p + "lo")
		a.jump(bpfJGT|bpfSrcX, r8, r5, 0, p+"hi")
		a.emit(movReg(r8, r5))
		a.label(p + "hi")
		a.emit(ldx(bpfSizeDW, r9, r9, offs["bio_next"]))
	}
	a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "hulled")
	emitStat(a, statsMapFD, markStatBios, "exit")
	// range request with no bio range to walk cannot be placed
	a.label("hulled")
	a.jump(bpfJGT|bpfSrcX, r8, r7, 0, "hullok")
	emitStat(a, statsMapFD, markStatPassthrough, "exit")
	a.label("hullok")
	a.emit(stx(bpfSizeDW, r10, r7, -24))
	a.emit(stx(bpfSizeDW, r10, r8, -16))

	// the in-flight entry, before anything is counted
	a.label("inflight")
	a.emit(ldx(bpfSizeDW, r1, r6, offs["mq_hctx"]))
	a.emit(ldx(bpfSizeW, r2, r1, offs["queue_num"]))
	a.emit(stx(bpfSizeW, r10, r2, -32))
	a.emit(ldx(bpfSizeW, r2, r6, offs["tag"]))
	a.emit(stx(bpfSizeW, r10, r2, -28))
	a.emit(stx(bpfSizeDW, r10, r6, -8))
	f := ldMapFD(r1, inflightMapFD)
	a.emit(f[0], f[1])
	a.emit(movReg(r2, r10))
	a.emit(addImm(r2, -8))
	a.emit(movReg(r3, r10))
	a.emit(addImm(r3, -32))
	a.emit(movImm(r4, 0))
	a.emit(callHelper(bpfFuncMapUpdateElem))
	a.jump(bpfJEQ|bpfSrcK, r0, 0, 0, "count")
	emitStat(a, statsMapFD, markStatInflightFull, "exit")

	// count: a plain write is its one range, a discard each of its bios
	a.label("count")
	// the fold hull starts empty: lowest all ones, highest zero. both are
	// spilled from a register, as the hull's updates are, so the verifier
	// compares the slots as scalars on every path
	a.emit(movImm(r1, -1))
	a.emit(stx(bpfSizeDW, r10, r1, -64))
	a.emit(movImm(r1, 0))
	a.emit(stx(bpfSizeDW, r10, r1, -72))
	a.emit(ldx(bpfSizeW, r3, r6, offs["cmd_flags"]))
	a.emit(andImm(r3, reqOpMask))
	for _, op := range l.rangeOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "bios")
	}
	a.emit(ldx(bpfSizeDW, r7, r10, -24))
	a.emit(ldx(bpfSizeDW, r8, r10, -16))
	emitMarkRange(a, "p", shift, regionMapFD, wideMapFD, "exit", "fold")

	// every bio of a discard is placed first; the ones the wide map could
	// not take are counted in the fold tree together at the end
	a.label("bios")
	a.emit(ldx(bpfSizeDW, r9, r6, offs["bio"]))
	for i := 0; i < markDiscardBios; i++ {
		p := fmt.Sprintf("b%d", i)
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "fold")
		a.emit(ldx(bpfSizeDW, r7, r9, offs["bio_sector"]))
		a.emit(ldx(bpfSizeW, r8, r9, offs["bio_size"]))
		a.emit(rshImm(r8, 9))
		a.emit(addReg(r8, r7))
		emitMarkRange(a, p, shift, regionMapFD, wideMapFD, p+"next", p+"next")
		a.label(p + "next")
		a.emit(ldx(bpfSizeDW, r9, r9, offs["bio_next"]))
	}
	a.jump(bpfJA, 0, 0, 0, "fold")
	emitMarkFold(a, foldShift, foldMapFD, statsMapFD, "exit")

	a.label("nodisk")
	a.emit(ldx(bpfSizeW, r3, r6, offs["cmd_flags"]))
	a.emit(andImm(r3, reqOpMask))
	for _, op := range l.readOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "exit")
	}
	// an nvme controller's queries (hwmon threshold reads, smart and identify
	// from monitoring tools) leave too. the check uses the stack slots the
	// write path does, which nothing on this path has written
	if l.nvme.usable() {
		emitNVMeAdminPass(a, l.nvme, offs["q"], movReg(r7, r6))
	}
	emitStat(a, statsMapFD, markStatPassthrough, "exit")

	a.label("exit")
	a.emit(movImm(r0, 0))
	a.emit(exitInsn())
	return a.resolve()
}

// assembles the block_rq_complete program for one
// disk: once a write-class request on the disk has completed in full (the
// bytes completed reach the bytes still to transfer) its in-flight entry is
// removed. it never counts anything; a partial completion leaves the entry
func buildMarkCompleteProgram(l markLayout, dev uint32, inflightMapFD int) ([]bpfInsn, error) {
	if dev >= 1<<31 {
		return nil, fmt.Errorf("device number %#x cannot be compared by the program", dev)
	}
	if l.completeRqArg != 0 {
		// nr_bytes is read at its fixed place after (rq, error); a different
		// argument order is a kernel this program was not written for
		return nil, fmt.Errorf("block_rq_complete request argument %d", l.completeRqArg)
	}
	offs, err := markOffsets(l)
	if err != nil {
		return nil, err
	}
	a := newBpfAsm()
	a.emit(movReg(r7, r1))
	a.emit(ldx(bpfSizeDW, r6, r7, 0)) // r6 = rq
	emitDiskMatch(a, l.trackLayout, offs, r6, dev, "exit", "exit")
	a.emit(ldx(bpfSizeW, r3, r6, offs["cmd_flags"]))
	a.emit(andImm(r3, reqOpMask))
	for _, op := range append(append([]int32{}, l.plainOps...), l.rangeOps...) {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "write")
	}
	a.jump(bpfJA, 0, 0, 0, "exit")
	a.label("write")
	a.emit(ldx(bpfSizeDW, r4, r7, 16))               // nr_bytes completed
	a.emit(ldx(bpfSizeW, r5, r6, offs["data_len"])) // bytes still to transfer
	a.jump(bpfJGT|bpfSrcX, r5, r4, 0, "exit")
	a.emit(stx(bpfSizeDW, r10, r6, -8))
	m := ldMapFD(r1, inflightMapFD)
	a.emit(m[0], m[1])
	a.emit(movReg(r2, r10))
	a.emit(addImm(r2, -8))
	a.emit(callHelper(bpfFuncMapDeleteElem))
	a.label("exit")
	a.emit(movImm(r0, 0))
	a.emit(exitInsn())
	return a.resolve()
}
