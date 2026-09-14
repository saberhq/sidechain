#!/usr/bin/env bash
# Mirror one `Host <alias>` block from ~/.brev/ssh_config into ~/.ssh/config, between markers.
#
#   scripts/brev_ssh_mirror.sh sidechain-gpu      # after every `brev create`
#
# Why: ~/.ssh/config already `Include`s brev's file and plain ssh follows that -- but Claude
# Science's Compute > "Add SSH host" dialog parses ~/.ssh/config itself and refuses an alias that
# is not literally defined there ("alias 'sidechain-gpu' must be defined in ~/.ssh/config",
# 2026-09-14). The mirror is appended AFTER the Include line, and ssh takes the first value it
# finds per option, so brev's own entry stays authoritative for ssh; the mirror only has to
# satisfy the dialog. Hostname and port change with every instance, which is why this runs once
# per box and not once per machine. Idempotent: the marked block is replaced, nothing else moves.
set -euo pipefail

ALIAS="${1:?usage: scripts/brev_ssh_mirror.sh <brev-instance>   (e.g. sidechain-gpu)}"
SRC="${BREV_SSH_CONFIG:-$HOME/.brev/ssh_config}"
DST="${SSH_CONFIG:-$HOME/.ssh/config}"

[ -f "$SRC" ] || { echo "no $SRC -- has a box been created?" >&2; exit 1; }
block=$(awk -v a="$ALIAS" '/^Host /{p=($2==a)} p' "$SRC")
[ -n "$block" ] || { echo "no 'Host $ALIAS' block in $SRC" >&2; exit 1; }

begin="# >>> brev_ssh_mirror: $ALIAS (refreshed by scripts/brev_ssh_mirror.sh; do not edit) >>>"
end="# <<< brev_ssh_mirror: $ALIAS <<<"
touch "$DST"; chmod 600 "$DST"
tmp=$(mktemp)
awk -v b="$begin" -v e="$end" '$0==b{skip=1} !skip{print} $0==e{skip=0}' "$DST" > "$tmp"
{
  cat "$tmp"
  [ -s "$tmp" ] && [ -n "$(tail -c1 "$tmp")" ] && echo
  echo "$begin"; echo "$block"; echo "$end"
} > "$DST"
rm -f "$tmp"
echo "mirrored 'Host $ALIAS' into $DST: $(echo "$block" | awk 'tolower($1)=="hostname"{h=$2} tolower($1)=="port"{p=$2} END{print h":"p}')"
