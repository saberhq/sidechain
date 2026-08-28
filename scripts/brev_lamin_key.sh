#!/usr/bin/env bash
# Ship the Lamin API key to a Brev box. ADR 0007 §4. Run FROM THE MAC, once per box,
# before `brev_bootstrap.sh`:
#
#   scripts/brev_lamin_key.sh sidechain-gpu
#
# THERE IS ONE KEY, AND IT IS ALREADY MADE. Saber created it once at
# https://lamin.ai/settings; it is a long-lived credential, and `lamin login` stored it
# on this Mac in ~/.lamin/current_user.env. This script reads it from there. Standing up
# a new box therefore costs this one command -- no trip to the hub, no new key, nothing
# to remember. You only ever touch the hub again to ROTATE the key (revoke on the
# settings page, `lamin login` here with the new one, and every future box picks it up).
#
# Why a file and not `brev exec BOX -- 'LAMIN_API_KEY=... bash bootstrap.sh'`: an env var
# passed that way becomes argv, and argv is visible in the box's process table, in this
# Mac's shell history, and in the agent transcript that logged the command. A 0600 file
# copied over keeps the value out of all three. It is not airtight -- after login the key
# sits in the box's own ~/.lamin/current_user.env in plaintext, as it does here -- but the
# box's disk dies with the instance, and a transcript does not.
#
# The key is never echoed, never written into the repo, and never committed. If you are
# debugging this script, do NOT add `set -x`.
set -euo pipefail

BOX="${1:?usage: scripts/brev_lamin_key.sh <brev-instance>   (e.g. sidechain-gpu)}"
SRC="${LAMIN_USER_ENV:-$HOME/.lamin/current_user.env}"

# A BOX SHOULD GET ITS OWN KEY. The hub allows five named keys per account, each with its
# own expiry and revocable on its own, and there is no read-only or scoped key -- so a
# separate key is the only blast-radius control there is. Make one at
# https://lamin.ai/settings, then:
#     LAMIN_API_KEY=<paste it here> scripts/brev_lamin_key.sh sidechain-gpu
# Revoking that one later costs the other boxes nothing and this Mac nothing. With the
# variable unset we fall back to the Mac's own stored key, which works and is one step
# less -- at the price of one credential shared by the laptop and every box.
if [ -n "${LAMIN_API_KEY:-}" ]; then
  KEY="$LAMIN_API_KEY"
  SOURCE="the LAMIN_API_KEY environment variable"
else
  [ -f "$SRC" ] || {
    echo "no $SRC, and LAMIN_API_KEY is unset." >&2
    echo "Either log in here (uv run lamin login) or pass a box key:" >&2
    echo "  LAMIN_API_KEY=<key> $0 $BOX" >&2
    exit 1
  }
  KEY="$(sed -n 's/^lamin_user_api_key=//p' "$SRC" | head -1)"
  SOURCE="$SRC"
fi

case "$KEY" in
  ""|null)
    echo "no usable API key in $SOURCE (found '${KEY:-empty}')." >&2
    echo "Create one at https://lamin.ai/settings, then: uv run lamin login" >&2
    exit 1
    ;;
esac

echo "shipping the key from $SOURCE to $BOX"

TMP="$(umask 077 && mktemp "${TMPDIR:-/tmp}/lamin_env.XXXXXX")"
trap 'rm -f "$TMP"' EXIT INT TERM
printf 'export LAMIN_API_KEY=%s\n' "$KEY" > "$TMP"

brev copy "$TMP" "$BOX:~/.lamin_env"
brev exec "$BOX" -- 'chmod 600 ~/.lamin_env && echo "lamin key placed at ~/.lamin_env"'

echo "next: brev exec $BOX -- 'bash ~/brev_bootstrap.sh'   # it sources ~/.lamin_env and logs in"
