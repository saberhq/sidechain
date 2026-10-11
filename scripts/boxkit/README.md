# boxkit: small tools for runs on rented boxes

Tools that every box run uses, whatever the task. Each is tested in `tests/test_boxkit.py`.

| file | what it does |
|---|---|
| `box_machine.py` | writes what a box really is (CPU model and vendor, cores, threads, memory, disks, cards) as JSON; run it first |
| `box_profile.sh` | one-minute sampler: load, free memory, the 60 largest processes, one line per card, into `procs.log` |
| `box_profile_report.py` | summarises a `procs.log`: load, memory, card use, and per kind of process the count, wall time, CPU, cores and peak memory |
| `box_scorer_caps.sh` | goes in front of `cell-eval2`: caps every thread pool of a scoring step (default 4) so that a step on a many-CPU box does not spread over all of them |

The planner that prices a queue on a machine before it is rented is `private/research/protocol/box_plan.py` (private repo).
