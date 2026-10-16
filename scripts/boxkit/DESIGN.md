# boxkit: how the pieces fit, and what is not built yet

## Built (each tested in `tests/test_boxkit.py`)

| tool | where it runs | what it does |
|---|---|---|
| `box_machine.py` | the box, first command | CPU model and vendor, cores, threads, memory, disks, cards, as JSON |
| `box_profile.sh` / `box_profile_report.py` | the box / anywhere | a one-minute sample of load, memory, the 60 largest processes and each card; a summary per kind of process |
| `box_scorer_caps.sh` | in front of `cell-eval2` | caps every thread pool of a scoring step; adds `gather_threads` to `run` |
| `box_scorer_gate.sh` | in front of `cell-eval2` | a `run` step waits for a free card and enough free memory, then runs under the caps; a hang guard kills a step that has run for `BOX_SCORER_TIMEOUT_S` (default 5400) and nothing else; a stop file never ends a waiting step |
| `box_order.py` | where the queue is made | longest estimate first inside each dependency layer |
| `box_done.py` | the Mac | reads a kit's done marker over ssh and prints the commands that end the box; never runs them |

The planner that prices a queue on a machine before it is rented is `private/research/protocol/box_plan.py`; the ledger takes
`box start --predict-hours --predict-usd`, `box pause` / `resume` for a stoppable box, and `box join` for a second task on a running box.

## Not built: the generic worker (K7) and shared directories (K8)

Two task kits exist (T98's and T111's second pass); the second is the first to reserve memory by step. Their worker scripts hold the queue
loop, the admission rule, claims with an owner, markers, and a pack-register-done chain, mixed with each task's recipe flags and checks.
The split worth making:

- **common** (a library, sourced): `pop` and `put_back` on a queue file under a lock, `reserve` (admission on what a line holds, oldest waiter
  first), claims with an `owner` file, the marker vocabulary (`*_ok`, `*_fail`, `line_done`), the packer and the done marker, `STOP`.
- **per task** (hook files in the task's run directory): the recipe flags for a line, the fold-to-source map, the arm table, the checks and
  reference comparisons, the memory table (read from measured peaks: `time -v`, not sampled medians).
- **shared state** under `$BOX_STATE` (default `~/box_state`): `resv/`, `gpu/`, `scorerwait/`, each reservation named for its session,
  and sessions named `<tid>_w<N>` so two tasks' workers never collide; the `pgrep` and session-name checks of `run_all.sh` become
  per-task prefixes.
- **a second task's lines** go into the owner's queue file with the task id in the line's name; the owner records `ledger.py box join`.
- **one work directory, named once:** every script reads it from the same variable (`BOX_WORK`) with no per-task default. T111's
  `realised_energy.py` fell back to `~/t111` when `T111_WORK` was unset, so a renamed copy of the kit failed it silently (T102, 2026-10-15).

It is not built because a library with one instance guesses its interface. The first customer is the next queue that needs a box (T114's
step 3 is waiting on a GO); extract the common part from T111's kit while building that task's hooks, and keep T111's kit as the test.
