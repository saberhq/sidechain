#!/usr/bin/env bash
# Keep the boxes' own Lamin API key in the login Keychain, and ship it to a box as a 0600 ~/.lamin_env.
#   scripts/brev_lamin_key.sh --store --expires YYYY-MM-DD   # in your own terminal: the box key, at a hidden prompt
#   scripts/brev_lamin_key.sh --check                        # stored, in date, not the Mac's key, accepted by the hub
#   scripts/brev_lamin_key.sh <ssh host>                     # from the Mac, once per box, before brev_bootstrap.sh
# Reads the Keychain item sidechain-lamin-box-key (LAMIN_API_KEY wins when set); ~/.lamin/current_user.env only to
# refuse the Mac's own key. Writes ~/.lamin_env on the box. Never prints the key: do not add `set -x`.
# Notes: private/code-notes/scripts/brev_lamin_key.sh.md
set -euo pipefail

SERVICE="sidechain-lamin-box-key"
ACCOUNT="${USER:-$(id -un)}"
MAC_ENV="${LAMIN_USER_ENV:-$HOME/.lamin/current_user.env}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

die() { echo "brev_lamin_key: $*" >&2; exit 1; }

read_key() {
  if [ -n "${LAMIN_API_KEY:-}" ]; then printf '%s' "$LAMIN_API_KEY"; return; fi
  security find-generic-password -a "$ACCOUNT" -s "$SERVICE" -w 2>/dev/null || true
}

read_expiry() {
  local d
  d="$(security find-generic-password -a "$ACCOUNT" -s "$SERVICE" 2>/dev/null \
        | sed -n 's/.*"icmt"<blob>="expires \([0-9-]*\)".*/\1/p' | head -1)" || true
  echo "${d:-unknown}"
}

mac_key() { if [ -f "$MAC_ENV" ]; then sed -n 's/^lamin_user_api_key=//p' "$MAC_ENV" | head -1; fi; }

need_key() {
  KEY="$(read_key)"
  case "$KEY" in
    ""|null) die "no box key stored. Make one at https://lamin.ai/settings, then: $0 --store --expires YYYY-MM-DD" ;;
  esac
  [ "$KEY" != "$(mac_key)" ] || die "the box key is this Mac's own Lamin key; make a separate key for boxes and --store it"
  if [ -z "${LAMIN_API_KEY:-}" ]; then
    local exp; exp="$(read_expiry)"
    if [ "$exp" != unknown ] && [[ "$exp" < "$(date +%F)" ]]; then
      die "the box key expired on $exp; make a new one at https://lamin.ai/settings, then --store"
    fi
  fi
}

case "${1:-}" in
  ""|-h|--help)
    sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
    ;;

  --store)
    shift
    EXPIRES=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --expires) EXPIRES="${2:?--expires needs a date, YYYY-MM-DD}"; shift 2 ;;
        *) die "unknown option for --store: $1" ;;
      esac
    done
    case "$EXPIRES" in
      [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
      *) die "--store needs --expires YYYY-MM-DD, the expiry date chosen at lamin.ai/settings" ;;
    esac
    [ -t 0 ] || die "--store reads the key at a hidden prompt: run it in your own terminal, not from a session"
    echo "Paste the boxes' Lamin API key twice (Keychain asks for it, then to retype it). Nothing is shown."
    security add-generic-password -U -a "$ACCOUNT" -s "$SERVICE" -j "expires $EXPIRES" -w
    if [ -n "$(read_key)" ] && [ "$(read_key)" = "$(mac_key)" ]; then
      security delete-generic-password -a "$ACCOUNT" -s "$SERVICE" >/dev/null 2>&1 || true
      die "that is this Mac's own Lamin key; nothing stored. Make a separate key for boxes"
    fi
    [ -n "$(read_key)" ] || die "nothing stored"
    echo "stored in the login Keychain as \`$SERVICE\`, expires $EXPIRES. Next: $0 --check"
    ;;

  --check)
    need_key
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/lamin_check.XXXXXX")"
    trap 'rm -rf "$tmp"' EXIT INT TERM
    if (cd "$ROOT" && HOME="$tmp" LAMIN_API_KEY="$KEY" .venv/bin/lamin login </dev/null >/dev/null 2>&1); then
      handle="$(cd "$ROOT" && HOME="$tmp" .venv/bin/python -I -c \
        'import lamindb_setup as s; print(s.settings.user.handle)' 2>/dev/null || true)"
      echo "OK: the hub accepts the box key (user ${handle:-unknown}); expires $(read_expiry); not the Mac's key"
    else
      die "the hub refused the box key, or was unreachable; check the key at https://lamin.ai/settings"
    fi
    ;;

  -*)
    die "unknown option: $1 (--store, --check, or an ssh host)"
    ;;

  *)
    BOX="$1"
    need_key
    echo "shipping the box key to $BOX"
    TMP="$(umask 077 && mktemp "${TMPDIR:-/tmp}/lamin_env.XXXXXX")"
    trap 'rm -f "$TMP"' EXIT INT TERM
    printf 'export LAMIN_API_KEY=%s\n' "$KEY" > "$TMP"
    scp -q -o BatchMode=yes "$TMP" "$BOX:.lamin_env"
    ssh -o BatchMode=yes -T "$BOX" 'chmod 600 ~/.lamin_env && echo "lamin key placed at ~/.lamin_env"'
    echo "next: ssh $BOX 'bash ~/brev_bootstrap.sh'"
    ;;
esac
