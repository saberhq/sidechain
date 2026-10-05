#!/usr/bin/env bash
# Mirror one `Host <instance>` block from ~/.brev/ssh_config into ~/.ssh/config, between markers,
# optionally under a different alias.
#
#   scripts/brev_ssh_mirror.sh sidechain-gpu-t84 --as sidechain-gpu   # after every `brev create`
#   scripts/brev_ssh_mirror.sh sidechain-gpu                           # same name in, same name out
#
# Why: ~/.ssh/config already `Include`s brev's file and plain ssh follows that -- but Claude
# Science's Compute > "Add SSH host" dialog parses ~/.ssh/config itself and refuses an alias that
# is not literally defined there ("alias 'sidechain-gpu' must be defined in ~/.ssh/config",
# 2026-09-14). Hostname and port change with every instance, which is why this runs once per box.
#
# Why `--as`: the Science app knows ONE host, `sidechain-gpu`, and since 2026-09-28 every box is
# created per task (`sidechain-gpu-<tid>`, `sidechain-gpu-sci-<tid>`), so the app's alias was left
# pointing at a deleted box's port and every probe failed ("Connection closed by <ip> port <p>",
# 2026-10-05). Mirroring the new box's block UNDER the app's alias re-points the existing entry; the
# app only has to re-probe, nothing is re-added. The block is written AFTER the Include line, and
# ssh takes the first value it finds per option, so brev's own entry stays authoritative when the
# names coincide; when they differ there is nothing to collide with -- unless brev's file still
# carries a `Host <alias>` block of its own, which this script warns about (it would win).
# Idempotent: the marked block for the alias is replaced, nothing else moves.
set -euo pipefail

usage="usage: scripts/brev_ssh_mirror.sh <brev-instance> [--as <alias>]   (e.g. sidechain-gpu-t84 --as sidechain-gpu)"
INSTANCE="${1:?$usage}"; shift
ALIAS="$INSTANCE"
while [ $# -gt 0 ]; do
  case "$1" in
    --as) ALIAS="${2:?--as needs an alias}"; shift 2 ;;
    *) echo "$usage" >&2; exit 2 ;;
  esac
done
SRC="${BREV_SSH_CONFIG:-$HOME/.brev/ssh_config}"
DST="${SSH_CONFIG:-$HOME/.ssh/config}"

[ -f "$SRC" ] || { echo "no $SRC -- has a box been created?" >&2; exit 1; }
block=$(awk -v a="$INSTANCE" '/^Host /{p=($2==a)} p' "$SRC")
[ -n "$block" ] || { echo "no 'Host $INSTANCE' block in $SRC (brev ls to see what exists)" >&2; exit 1; }
if [ "$ALIAS" != "$INSTANCE" ]; then
  block=$(printf '%s\n' "$block" | awk -v a="$ALIAS" 'NR==1 && /^Host /{print "Host " a; next} {print}')
  if awk -v a="$ALIAS" '/^Host /{if ($2==a) f=1} END{exit !f}' "$SRC"; then
    echo "warning: $SRC also defines 'Host $ALIAS'; ssh takes the first match, so that block wins over this mirror" >&2
  fi
fi

begin="# >>> brev_ssh_mirror: $ALIAS (refreshed by scripts/brev_ssh_mirror.sh; do not edit) >>>"
end="# <<< brev_ssh_mirror: $ALIAS <<<"
touch "$DST"; chmod 600 "$DST"
tmp="$DST.brev_ssh_mirror.$$"; trap 'rm -f "$tmp"' EXIT
awk -v b="$begin" -v e="$end" '$0==b{skip=1} !skip{print} $0==e{skip=0}' "$DST" > "$tmp"
{
  cat "$tmp"
  [ -s "$tmp" ] && [ -n "$(tail -c1 "$tmp")" ] && echo
  echo "$begin"; echo "$block"; echo "$end"
} > "$DST"
rm -f "$tmp"
echo "mirrored 'Host $INSTANCE' as 'Host $ALIAS' into $DST: $(echo "$block" | awk 'tolower($1)=="hostname"{h=$2} tolower($1)=="port"{p=$2} END{print h":"p}')"
