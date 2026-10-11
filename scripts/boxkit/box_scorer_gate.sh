#!/bin/bash
# The scoring gate: a wrapper in front of cell-eval2 that makes `cell-eval2 run` wait for a free card and enough free memory, then runs it
# with its thread pools capped (box_scorer_caps.sh). Everything but `run` passes straight through.
#
#   in Python:  sidechain.eval.mirror2026.CE2 = ["bash", ".../box_scorer_gate.sh"] + sidechain.eval.mirror2026.CE2
#
# Why: an arm holds a few GiB for hours and its scoring step holds tens of GiB for minutes. A step is admitted on its own memory and a card,
# separately from the arm's working reservation, so a queue is not sized by its peaks. Descended from T111's second-pass gate; what differs:
#   * no taskset pin (it never applied: `nproc` reads 1 under OMP_NUM_THREADS=1); the thread caps replace it;
#   * STOP never ends a step that is already waiting: its arm has emitted for hours and a stop only blocks NEW starts;
#   * the lock is a mkdir, so the gate runs on a Mac as well as on a box (no flock), and the timeout is a watcher (no `timeout` binary);
#   * state lives under $BOX_STATE (default ~/box_state), not under one task's directory.
#
# Environment (all optional except BOX_SCORER_GIB):
#   BOX_SCORER_GIB       GiB this step holds at its peak (the arm's fold decides it)
#   BOX_STATE            state directory: resv/ (reservations: a number in a file), gpu/ (one file per busy card), scorerwait/, log
#   BOX_BUDGET_GIB       memory the box may reserve (default MemTotal - 14)
#   BOX_FREE_MARGIN_GIB  GiB that must really be free beyond the step's own (default 6)
#   BOX_CARDS            number of cards (default: nvidia-smi, at least 1)
#   BOX_SCORER_TIMEOUT_S a step still running after this long is hung, not working: killed, logged SCORER_TIMEOUT (default 5400)
#   BOX_SCORER_WEDGE_S   a step that waited this long with no step live anywhere is admitted on the accounting alone (default 1800)
#   BOX_POLL_S           seconds between admission attempts (default 20)
#   BOX_LIVENESS         tmux (a reservation file named for a tmux session is live while the session is; default) or none
export PATH="$HOME/.local/bin:$PATH"
set -uo pipefail
T=${BOX_STATE:-$HOME/box_state}
GLOG=$T/scorer_gate.log
mkdir -p "$T/resv" "$T/gpu" "$T/scorerwait"
[ $# -ge 1 ] || { echo "box_scorer_gate.sh: no cell-eval2 path" >> "$GLOG"; exit 64; }
HERE=$(cd "$(dirname "$0")" && pwd)
CE=$1; shift
if [ "${1:-}" != run ]; then exec "$CE" "$@"; fi

GIB=${BOX_SCORER_GIB:-0}
case $GIB in ''|*[!0-9]*) GIB=0;; esac
memfree()  { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo 2>/dev/null || echo 9999; }
memtotal() { awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo 2>/dev/null || echo 9999; }
BUDGET=${BOX_BUDGET_GIB:-$(( $(memtotal) - 14 ))}
MARGIN=${BOX_FREE_MARGIN_GIB:-6}
TIMEOUT_S=${BOX_SCORER_TIMEOUT_S:-5400}
WEDGE_S=${BOX_SCORER_WEDGE_S:-1800}
POLL=${BOX_POLL_S:-20}
LIVENESS=${BOX_LIVENESS:-tmux}
NCARD=${BOX_CARDS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')}
case $NCARD in ''|*[!0-9]*) NCARD=1;; esac
[ "$NCARD" -ge 1 ] || NCARD=1
CARD=""
WAITFILE=$T/scorerwait/$$
LOCK=$T/resv.lock.d
t0=$(date +%s)
say() { echo "$(date -u +%FT%TZ) pid $$ gib $GIB card ${2:--} waited $(( $(date +%s) - t0 ))s $1" >> "$GLOG"; }
HAVELOCK=0
cleanup() { rm -f "$T/resv/s.$$" "$WAITFILE"; [ -n "$CARD" ] && rm -f "$T/gpu/card$CARD"; [ "$HAVELOCK" = 1 ] && rmdir "$LOCK" 2>/dev/null; return 0; }
trap 'cleanup' EXIT
trap 'cleanup; exit 143' TERM INT HUP

# a lock without flock: mkdir is atomic. A lock older than 60 s was left by a killed process and is broken.
lock() {
  local n=0
  until mkdir "$LOCK" 2>/dev/null; do
    n=$((n + 1))
    if [ $((n % 50)) -eq 0 ] && [ -d "$LOCK" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || stat -f %m "$LOCK" 2>/dev/null || date +%s) ))
      [ "$age" -gt 60 ] && rmdir "$LOCK" 2>/dev/null
    fi
    sleep 0.1
  done
  HAVELOCK=1
}
unlock() { HAVELOCK=0; rmdir "$LOCK" 2>/dev/null; return 0; }
numeric() { [ -n "${1:-}" ] && [ -z "${1//[0-9]/}" ]; }

printf '%s\n' "$GIB $t0" > "$T/scorerwait/.$$.tmp" && mv "$T/scorerwait/.$$.tmp" "$WAITFILE"
say "waiting (budget ${BUDGET}G, memfree $(memfree)G, cards $NCARD, timeout ${TIMEOUT_S}s)"
STOPSAID=no
WEDGED=no
while true; do
  if [ -f "$T/STOP" ] && [ "$STOPSAID" = no ]; then say "STOP seen: this step belongs to a running arm and waits its turn"; STOPSAID=yes; fi
  lock
  waited=$(( $(date +%s) - t0 ))
  used=0; nlive_s=0
  if [ "$LIVENESS" = tmux ]; then sess=" $(tmux list-sessions -F '#S' 2>/dev/null | tr '\n' ' ') "; srcok=$(tmux list-sessions >/dev/null 2>&1; echo $?); else srcok=1; fi
  for f in "$T"/resv/*; do
    [ -f "$f" ] || continue
    b=${f##*/}
    if [ "$b" != "${b#s.}" ]; then
      kill -0 "${b#s.}" 2>/dev/null || { rm -f "$f"; continue; }
      nlive_s=$((nlive_s + 1))
    elif [ "$srcok" -eq 0 ]; then
      rest=${sess#*" $b "}
      if [ "$rest" = "$sess" ]; then rm -f "$f"; continue; fi
    fi
    v=$(cat "$f" 2>/dev/null)
    numeric "$v" || v=0
    used=$((used + v))
  done
  free=""; k=0
  while [ "$k" -lt "$NCARD" ]; do
    g=$T/gpu/card$k
    if [ ! -f "$g" ]; then free=$k; break; fi
    p=$(cat "$g" 2>/dev/null)
    if ! numeric "$p"; then rm -f "$g"; free=$k; break; fi
    kill -0 "$p" 2>/dev/null || { rm -f "$g"; free=$k; break; }
    k=$((k + 1))
  done
  mf=$(memfree)
  older=no
  for f in "$T"/scorerwait/*; do               # the oldest waiting step that would fit goes first, so a big step cannot starve
    [ -f "$f" ] || continue
    b=${f##*/}
    [ "$b" = "$$" ] && continue
    numeric "$b" || { rm -f "$f"; continue; }
    kill -0 "$b" 2>/dev/null || { rm -f "$f"; continue; }
    read -r og os < "$f"
    numeric "${og:-}" && numeric "${os:-}" || continue
    if [ "$os" -lt "$t0" ] || { [ "$os" -eq "$t0" ] && [ "$b" -lt "$$" ]; }; then
      if [ $((used + og)) -le "$BUDGET" ] && [ "$mf" -ge $((og + MARGIN)) ]; then older=yes; fi
    fi
  done
  ok=no
  if [ -n "$free" ] && [ "$older" = no ] && [ $((used + GIB)) -le "$BUDGET" ] && [ "$mf" -ge $((GIB + MARGIN)) ]; then
    ok=yes
  elif [ "$waited" -ge "$WEDGE_S" ] && [ "$nlive_s" -eq 0 ] && [ -n "$free" ] && [ $((used + GIB)) -le "$BUDGET" ] && [ "$mf" -ge $((GIB + 2)) ]; then
    ok=yes; WEDGED=yes                           # nothing live anywhere can release memory: break the wedge, loudly
  fi
  if [ "$ok" = yes ]; then
    printf '%s\n' "$GIB" > "$T/resv/.s.$$.tmp" && mv "$T/resv/.s.$$.tmp" "$T/resv/s.$$"
    printf '%s\n' "$$" > "$T/gpu/.card$free.tmp" && mv "$T/gpu/.card$free.tmp" "$T/gpu/card$free"
    rm -f "$WAITFILE"
    CARD=$free
    unlock
    break
  fi
  unlock
  sleep "$POLL"
done
if [ "$WEDGED" = yes ]; then say "ADMITTED_BY_THE_WEDGE_RULE (free-memory margin relaxed to 2G)" "$CARD"; else say ADMITTED "$CARD"; fi

# run it: every thread pool capped, the card chosen, a hang guard that only ever applies to a step (never to an arm)
flag=$T/.timedout.$$
rm -f "$flag"
CUDA_VISIBLE_DEVICES=$CARD bash "$HERE/box_scorer_caps.sh" "$CE" "$@" &
child=$!
( sleep "$TIMEOUT_S" & sp=$!; trap 'kill $sp 2>/dev/null; exit 0' TERM
  wait $sp 2>/dev/null; touch "$flag"; kill -TERM "$child" 2>/dev/null
  sleep 60 & sp=$!; wait $sp 2>/dev/null; kill -KILL "$child" 2>/dev/null ) &
watcher=$!
wait "$child"; rc=$?
kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
if [ -f "$flag" ]; then rc=124; rm -f "$flag"; say "SCORER_TIMEOUT after ${TIMEOUT_S}s: the card and the reservation are given back; loco reports a failed scorer" "$CARD"
else say "FINISHED rc $rc" "$CARD"; fi
cleanup
exit $rc
