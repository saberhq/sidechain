#!/bin/bash
# One-minute sampler for a box run: load, free memory, the 60 largest processes and one line per GPU card, appended to <out_dir>/procs.log
# until <out_dir>/STOP_PROFILE exists. Read it with box_profile_report.py.
#   bash box_profile.sh ~/run/profile 60 &
out=${1:?usage: box_profile.sh <out_dir> [seconds]}; every=${2:-60}
mkdir -p "$out"
while [ ! -e "$out/STOP_PROFILE" ]; do
  {
    echo "=== $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg) memavail_GiB $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)"
    ps -eo pid,ppid,rss,pcpu,etimes,args --sort=-rss | head -60 | cut -c1-600
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null |
      awk -F', ' '{print "card " $1 ", " $2 " MiB, " $3 " MiB, " $4 " %, " $5 " %"}'
  } >> "$out/procs.log"
  sleep "$every"
done
