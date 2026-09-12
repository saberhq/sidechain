#!/usr/bin/env bash
# Preview both public surfaces locally, before anything is pushed.
#
#   scripts/preview.sh              # site on :1414, README on :1415
#   scripts/preview.sh --site       # the Hugo site only
#   scripts/preview.sh --readme     # the README only
#
# Two surfaces change when the standings are regenerated and only one of them is a website,
# so "does it build" is not the same question as "does it look right". This serves both.
#
# The README is rendered by GITHUB'S OWN markdown endpoint (`gh api /markdown`), not a local
# approximation, so the table, the alignment and the sanitiser's opinion of the `<picture>`
# wordmark are what github.com will actually show. The Hugo side live-reloads; the README is
# rendered once per run, so re-run it after editing README.md.
#
# Ctrl-C stops both. Ports are overridable: PORT_SITE=8080 scripts/preview.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT_SITE="${PORT_SITE:-1414}"
PORT_README="${PORT_README:-1415}"
REPO="${REPO:-saberhq/sidechain}"

want_site=1
want_readme=1
case "${1:-}" in
  --site)   want_readme=0 ;;
  --readme) want_site=0 ;;
  "")       ;;
  *) echo "usage: $(basename "$0") [--site|--readme]" >&2; exit 2 ;;
esac

TMP="$(mktemp -d)"
cleanup() { trap - EXIT INT TERM; kill 0 2>/dev/null || true; rm -rf "$TMP"; }
trap cleanup EXIT INT TERM

if [ "$want_readme" = 1 ]; then
  command -v gh >/dev/null || { echo "preview: gh is not installed -- brew install gh" >&2; exit 1; }
  gh api --method POST /markdown \
     -f mode=gfm -f context="$REPO" -f text="$(cat "$ROOT/README.md")" > "$TMP/_body.html"
  # README images are relative (assets/wordmark-*.svg), so serve them from beside the page.
  ln -s "$ROOT/assets" "$TMP/assets"
  {
    echo '<!doctype html><meta charset="utf-8"><title>README — '"$REPO"'</title>'
    echo '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.css">'
    echo '<style>body{margin:0;background:var(--color-canvas-default,#fff)}'
    echo '.markdown-body{box-sizing:border-box;max-width:1012px;margin:0 auto;padding:45px}</style>'
    echo '<article class="markdown-body">'
    cat "$TMP/_body.html"
    echo '</article>'
  } > "$TMP/index.html"
  python3 -m http.server "$PORT_README" --directory "$TMP" >/dev/null 2>&1 &
fi

if [ "$want_site" = 1 ]; then
  command -v hugo >/dev/null || { echo "preview: hugo is not installed -- brew install hugo" >&2; exit 1; }
  hugo server --source "$ROOT/site" --port "$PORT_SITE" >"$TMP/hugo.log" 2>&1 &
fi

sleep 2
echo
[ "$want_site" = 1 ]   && echo "  site    http://localhost:$PORT_SITE/sidechain/          (live-reloads on edit)"
[ "$want_readme" = 1 ] && echo "  README  http://localhost:$PORT_README/                    (re-run to re-render)"
echo
echo "  Ctrl-C to stop."
wait
