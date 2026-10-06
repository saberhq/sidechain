#!/usr/bin/env bash
# Run brev on the long-lived Brev API key, so `brev ls / stop / delete` keep answering after
# the ~7 h `brev login` has died (trap 12 in /sidechain-brev). Run FROM THE MAC:
#
#   scripts/brev_key.sh --store --expires 2026-12-31   # Saber, once per key, in his own terminal
#   scripts/brev_key.sh --check                        # stored, in date, and answering on its own?
#   scripts/brev_key.sh ls                             # any brev call, authenticated by the key
#   scripts/brev_key.sh delete sidechain-gpu-t84       # the call that ends the bill
#
# THE KEY IS MADE ONCE, BY SABER. Brev console -> Settings -> API Keys -> Create API Key:
# access "Read & Write" (a delete is a write), an expiry date of his choosing. The console
# shows it once; `--store` takes it at a hidden prompt and puts it in the macOS login Keychain
# (service `sidechain-brev-api-key`), with the expiry date beside it so the ledger's pre-flight
# can read the date without reading the key. To ROTATE: revoke in the console, `--store` again.
#
# What the key replaces and what it does not. `brev login` writes a session into
# ~/.brev/credentials.json that the CLI refreshes for about seven hours and then loses; this
# script leaves that file alone and hands the key to ONE brev call through BREV_API_KEY, which
# the CLI reads before any saved login (brev-cli pkg/auth/auth.go, GetFreshAccessTokenOrNil).
# The interactive login is still how a box is CREATED and how its SSH entry is written: an
# API-key login "does not set up your user SSH credentials" (docs.nvidia.com/brev/guides/api-keys).
# So: create and bootstrap under `brev login`, as before; everything after rides plain SSH
# (`ssh sidechain-gpu-...`, which never needed the login) and this script.
#
# Why the environment and not `brev --api-key <key>`: a flag is argv, and argv is visible in
# the process table, the shell history and the agent transcript that logged the command.
#
# Needs brev >= 0.6.335. An older CLI ignores BREV_API_KEY without a word and answers from the
# saved login instead, which looks like success until the login dies -- so this refuses it.
# When the brev on PATH is that old, a newer copy kept at ~/.brev/brev-0.6.335 is used in its
# place, for these calls only, so the shared CLI need not be upgraded under running boxes.
#
# The key is never echoed, never written into the repo, and never committed. If you are
# debugging this script, do NOT add `set -x`.
set -euo pipefail

SERVICE="sidechain-brev-api-key"
ACCOUNT="${USER:-$(id -un)}"
BREV="${BREV_BIN:-brev}"
MIN_VERSION="0.6.335"
SIDE_COPY="$HOME/.brev/brev-$MIN_VERSION"   # a release binary beside the installed CLI

die() { echo "brev_key: $*" >&2; exit 1; }

# The stored key on stdout, or nothing. BREV_API_KEY in the environment wins, unstored.
read_key() {
  if [ -n "${BREV_API_KEY:-}" ]; then printf '%s' "$BREV_API_KEY"; return; fi
  security find-generic-password -a "$ACCOUNT" -s "$SERVICE" -w 2>/dev/null || true
}

# The expiry date written beside the key at --store, or `unknown`.
read_expiry() {
  local d
  d="$(security find-generic-password -a "$ACCOUNT" -s "$SERVICE" 2>/dev/null \
        | sed -n 's/.*"icmt"<blob>="expires \([0-9-]*\)".*/\1/p' | head -1)" || true
  echo "${d:-unknown}"
}

# The version of the brev at $1, or nothing.
version_of() {
  "$1" --version --no-check-latest 2>/dev/null | sed -n 's/.*Current Version: v\([0-9.]*\).*/\1/p' | head -1
}

# Is the version $1 at least MIN_VERSION?
new_enough() {
  [ -n "$1" ] && [ "$(printf '%s\n%s\n' "$MIN_VERSION" "$1" | sort -t. -k1,1n -k2,2n -k3,3n | head -1)" = "$MIN_VERSION" ]
}

# Refuse a CLI that would ignore the key. With no BREV_BIN given, an old brev on PATH gives
# way to the side copy when there is one.
need_version() {
  command -v "$BREV" >/dev/null 2>&1 || die "no \`$BREV\` on PATH"
  local have
  have="$(version_of "$BREV")"
  [ -n "$have" ] || die "could not read the version from \`$BREV --version\`"
  new_enough "$have" && return 0
  if [ -z "${BREV_BIN:-}" ] && [ -x "$SIDE_COPY" ] && new_enough "$(version_of "$SIDE_COPY")"; then
    BREV="$SIDE_COPY"
    return 0
  fi
  die "brev $have ignores BREV_API_KEY and would answer from the saved login; needs >= $MIN_VERSION (\`brew upgrade brev\`, or a release binary at $SIDE_COPY)"
}

need_key() {
  KEY="$(read_key)"
  [ -n "$KEY" ] || die "no key stored. Saber: scripts/brev_key.sh --store --expires YYYY-MM-DD"
  case "$KEY" in bak-*) ;; *) die "the stored value is not a Brev API key (no \`bak-\` prefix); --store again" ;; esac
}

case "${1:-}" in
  ""|-h|--help)
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
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
      *) die "--store needs --expires YYYY-MM-DD, the expiry date chosen in the Brev console" ;;
    esac
    [ -t 0 ] || die "--store reads the key at a hidden prompt: run it in your own terminal, not from a session"
    echo "Paste the Brev API key (it starts with bak-). Nothing is shown as you type."
    # `-w` last, with no value: security prompts for it, so the key never becomes argv.
    security add-generic-password -U -a "$ACCOUNT" -s "$SERVICE" -j "expires $EXPIRES" -w
    case "$(read_key)" in
      bak-*) echo "stored in the login Keychain as \`$SERVICE\`, expires $EXPIRES. Next: scripts/brev_key.sh --check" ;;
      *) security delete-generic-password -a "$ACCOUNT" -s "$SERVICE" >/dev/null 2>&1 || true
         die "that was not a Brev API key (no \`bak-\` prefix); nothing stored" ;;
    esac
    ;;

  --expires)
    # For `ledger.py box preflight`: the expiry date when a key is stored and this CLI can use
    # it, exit 1 with the reason otherwise. No network, and the key itself is never printed.
    need_version
    need_key
    read_expiry
    ;;

  --check)
    need_version
    need_key
    EXPIRES="$(read_expiry)"
    TODAY="$(date +%Y-%m-%d)"
    if [ "$EXPIRES" != "unknown" ] && [ "$TODAY" \> "$EXPIRES" ]; then
      die "the stored key expired on $EXPIRES: make a new one in the Brev console, then --store"
    fi
    # An empty HOME has no saved login in it, so an answer here is the key's own.
    EMPTY="$(mktemp -d "${TMPDIR:-/tmp}/brev_key.XXXXXX")"
    trap 'rm -rf "$EMPTY"' EXIT
    set +e
    OUT="$(HOME="$EMPTY" BREV_API_KEY="$KEY" "$BREV" ls --no-check-latest </dev/null 2>&1)"
    RC=$?
    set -e
    if printf '%s' "$OUT" | grep -q -i -E 'unauthorized|forbidden|logged out'; then
      printf '%s\n' "$OUT" | grep -v -E '^(/go/|github\.com/|: \[error\])' | head -5 >&2
      die "Brev refused the key (above). Expired or revoked? Check the Brev console"
    fi
    # Anything else that failed is the call, not the key: a TLS or network error passes on a retry.
    if [ $RC -ne 0 ] || printf '%s' "$OUT" | grep -q -E 'RESTY|\[error\]'; then
      printf '%s\n' "$OUT" | grep -v -E '^(/go/|github\.com/|: \[error\])' | cut -c1-240 | head -5 >&2
      die "\`brev ls\` did not get an answer (above), which says nothing about the key: run --check again"
    fi
    echo "OK -- the key answers \`brev ls\` with no login behind it; expires $EXPIRES"
    printf '%s\n' "$OUT"
    ;;

  --*)
    die "unknown option: $1 (--store, --check, --expires, or a brev command)"
    ;;

  login|logout)
    die "\`brev $1\` rewrites ~/.brev/credentials.json, the login every session shares: run it yourself, without this script"
    ;;

  *)
    need_version
    need_key
    BREV_API_KEY="$KEY" exec "$BREV" "$@"
    ;;
esac
