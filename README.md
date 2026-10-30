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

`blocktrack_mark_prog_linux.go` holds the mark-mode programs the agent uses
while a cloud disk is replicated from snapshots. One program on
`block_rq_issue` counts each write-class request's regions and records it as
in flight; one on `block_rq_complete` removes it once it has completed in full.
A request the programs cannot account for (a driver passthrough that may
change data, a full map, a bio chain past its walk limit) moves a witness
counter, and the agent then reads the whole disk.

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
