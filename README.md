# blocktrack-bpf

The eBPF program the CloudHound discovery agent loads to track which regions of
a physical server's disk change between replication passes, and the small
assembler that emits it.

The program attaches to the kernel's `block_rq_complete` tracepoint as a
BTF-typed raw tracepoint program. For every completed write or discard on one
disk it adds one to an atomic counter for each fixed-size region the request
covers; requests it cannot mark completely (wider than its unrolled loop, or a
discard merged from several separate ranges) are counted separately so the
agent falls back to a full read. It reads the request structure directly, with
field offsets taken from the running kernel's BTF, so it needs no kernel
headers, no compiler on the host and no per-kernel build.

A driver passthrough (a command sent through sg, bsg or the nvme ioctls) is
counted as a possible change unless the disk's driver and the command prove it
cannot change the media. On a SCSI disk (the `sd` driver on blk-mq) the
programs read the command bytes with the kernel's probe read helper and let a
command through only when every read succeeds and the command matches one
entry of `scsiPassRules` exactly: its length, its data phase and every tested
byte. The table holds read-only queries such as TEST UNIT READY, INQUIRY,
READ CAPACITY, MODE SENSE, LOG SENSE, PERSISTENT RESERVE IN and the ATA
PASS-THROUGH reads smartd and smartctl send (IDENTIFY, SMART READ DATA,
THRESHOLDS and LOG, READ LOG EXT). A command is never let through because of
how many bytes it moved, so a FORMAT UNIT, SANITIZE or WRITE SAME sent with a
data-in buffer still counts. On virtio_blk the driver's own serial number
query passes. On every other disk, and when the command cannot be read, every
passthrough counts. On a queue with no disk only an nvme controller's admin
query (Get Log Page, Identify, Get Features and the like, proven by the
controller naming the queue as its admin queue) passes.

On kernel 5.4, which has vmlinux BTF but no typed `block_rq_complete`
tracepoint and no mmap-able maps, `buildTrackProgramClassic` emits a classic
raw tracepoint program instead. It reads every kernel field with
`bpf_probe_read` at the BTF offsets and treats any failed read as a request it
cannot mark, so the agent reads the whole disk. It sets one byte per changed
region in one of two buffers; the agent switches buffers and waits for a
`membarrier` grace period before it reads the old one.

`blocktrack_mark_prog_linux.go` holds the mark-mode programs the agent uses
while a cloud disk is replicated from snapshots. One program on
`block_rq_issue` counts each write-class request's regions and records it as
in flight; one on `block_rq_complete` removes it once it has completed in full.
A request the programs cannot account for (a driver passthrough the same rules
do not let through, a full map, a bio chain past its walk limit) moves a
witness counter, and the agent then reads the whole disk.

## Licence

`blocktrack_prog_linux.go` and `blocktrack_mark_prog_linux.go` are offered under either the BSD 2-Clause licence
(`LICENSE-BSD`) or the GNU General Public License version 2 (`LICENSE-GPL`),
at your option. The kernel requires a GPL-compatible licence for a BPF program
that reads kernel structures, and the program declares itself to the kernel as
`Dual BSD/GPL`.

This repository contains only the program and its assembler. The CloudHound
agent that loads it, and the rest of CloudHound, are proprietary and are not
covered by these licences.

## Layout

Each file is a verbatim copy of the file of the same name under
`agent/replication/` in the CloudHound agent. They are written against Go 1.19 and depend only on the
standard library. `buildTrackProgram` takes the struct offsets and operation
codes the agent resolved from the kernel's BTF, the disk's device number, the
region size and the two map file descriptors, and returns the instructions.
