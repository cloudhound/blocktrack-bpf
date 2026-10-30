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
	markStatWideFull    = 0 // the wide range map refused an insert
	markStatInflightFull = 1 // the in-flight map refused an insert
	markStatPassthrough = 2 // an operation on the disk that is neither a read nor a known write
	markStatBios        = 3 // a discard carried more bios than the walk covers
	markStatSlots       = 4
)

// a discard's ranges are read from its bio chain, at most this many
const markDiscardBios = 64

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
	bioNext       btfField // struct bio .bi_next (pointer)
	bioSize       btfField // struct bio .bi_iter.bi_size (u32)
	readOps       []int32  // operations that never change data
	plainOps      []int32  // write-class operations with one contiguous range
	rangeOps      []int32  // discard-class operations whose ranges come from the bios
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
// grow and a mark compares counts exactly as it does for regions. an empty
// range records nothing. r7 and r8 are clobbered, as are r0-r5
func emitMarkRange(a *bpfAsm, p string, shift uint, regionMapFD, wideMapFD, statsMapFD int, next string) {
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
	emitStat(a, statsMapFD, markStatWideFull, next)
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
// wide value, -4 map index. registers: r6 = request, r7/r8 = range, r9 = the
// bio being walked (and the disk during the match)
func buildMarkIssueProgram(l markLayout, dev uint32, shift uint, regionMapFD, wideMapFD, inflightMapFD, statsMapFD int) ([]bpfInsn, error) {
	if dev >= 1<<31 {
		return nil, fmt.Errorf("device number %#x cannot be compared by the program", dev)
	}
	if len(l.plainOps) == 0 || len(l.rangeOps) == 0 || len(l.readOps) == 0 {
		return nil, fmt.Errorf("operation codes missing")
	}
	if l.issueRqArg < 0 || l.issueRqArg > 3 {
		return nil, fmt.Errorf("request argument %d", l.issueRqArg)
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
	// so anything but a read there moves the witness
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
	emitStat(a, statsMapFD, markStatPassthrough, "exit")

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
	a.emit(ldx(bpfSizeW, r3, r6, offs["cmd_flags"]))
	a.emit(andImm(r3, reqOpMask))
	for _, op := range l.rangeOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "bios")
	}
	a.emit(ldx(bpfSizeDW, r7, r10, -24))
	a.emit(ldx(bpfSizeDW, r8, r10, -16))
	emitMarkRange(a, "p", shift, regionMapFD, wideMapFD, statsMapFD, "exit")

	a.label("bios")
	a.emit(ldx(bpfSizeDW, r9, r6, offs["bio"]))
	for i := 0; i < markDiscardBios; i++ {
		p := fmt.Sprintf("b%d", i)
		a.jump(bpfJEQ|bpfSrcK, r9, 0, 0, "exit")
		a.emit(ldx(bpfSizeDW, r7, r9, offs["bio_sector"]))
		a.emit(ldx(bpfSizeW, r8, r9, offs["bio_size"]))
		a.emit(rshImm(r8, 9))
		a.emit(addReg(r8, r7))
		emitMarkRange(a, p, shift, regionMapFD, wideMapFD, statsMapFD, p+"next")
		a.label(p + "next")
		a.emit(ldx(bpfSizeDW, r9, r9, offs["bio_next"]))
	}
	a.jump(bpfJA, 0, 0, 0, "exit")

	a.label("nodisk")
	a.emit(ldx(bpfSizeW, r3, r6, offs["cmd_flags"]))
	a.emit(andImm(r3, reqOpMask))
	for _, op := range l.readOps {
		a.jump(bpfJEQ|bpfSrcK, r3, 0, op, "exit")
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
