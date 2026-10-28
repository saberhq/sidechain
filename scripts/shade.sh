#!/usr/bin/env bash
# Rent and end a Shadeform box from the Mac: the API key in the macOS Keychain, Shadeform's REST API through curl,
# one ssh host entry per box, so the kits reach it as `ssh <name>` exactly as they reach a Brev box. Run FROM THE MAC:
#
#   scripts/shade.sh --store                    # Saber, once per key, in his own terminal: the key at a hidden prompt
#   scripts/shade.sh --check                    # stored, and answering?
#   scripts/shade.sh types A100_80G 2           # on-demand types with stock now, cheapest first (gpu type, gpu count)
#   scripts/shade.sh sshkey                     # make ~/.ssh/sidechain_shade_ed25519 if absent, register it, make it the default
#   scripts/shade.sh create sidechain-gpu-t125 hyperstack A100_80Gx2 canada-1      # SHADE_OS=<image> to override the OS
#   scripts/shade.sh typeinfo massedcompute L40Sx2  # one type's raw entry: its OS images, CPU and memory
#   scripts/shade.sh wait sidechain-gpu-t125    # until active, then writes its host entry: `ssh sidechain-gpu-t125`
#   scripts/shade.sh ls [--json] | info NAME | delete NAME     # --json: the board's Box panel reads it
#
# The API: https://docs.shadeform.ai/api-reference (base https://api.shadeform.ai/v1, header X-API-KEY). The key reaches
# curl on stdin (`-K -`), never argv; it is never echoed, written into the repo or committed. Do not add `set -x`.
# Host entries go to ~/.ssh/sidechain_shade_config, which ~/.ssh/config includes (the line is added once). T118.
set -euo pipefail

SERVICE="sidechain-shadeform-api-key"
ACCOUNT="${USER:-$(id -un)}"
API="https://api.shadeform.ai/v1"
KEYFILE="$HOME/.ssh/sidechain_shade_ed25519"
HOSTS="$HOME/.ssh/sidechain_shade_config"
PY="${PYTHON:-/usr/bin/python3}"

die() { echo "shade: $*" >&2; exit 1; }
read_key() { security find-generic-password -a "$ACCOUNT" -s "$SERVICE" -w 2>/dev/null || true; }
need_key() { KEY="$(read_key)"; [ -n "$KEY" ] || die "no key stored. Saber: scripts/shade.sh --store"; }

# METHOD PATH [JSON] -> the body; dies with the status and the body's head on anything but 2xx.
call() {
  local out code
  if [ -n "${3:-}" ]; then
    out="$(printf 'header = "X-API-KEY: %s"\n' "$KEY" | curl -sS -m 60 -K - -X "$1" -H 'Content-Type: application/json' \
            --data "$3" -w '\n%{http_code}' "$API$2")" || die "$1 $2: no answer (network)"
  else
    out="$(printf 'header = "X-API-KEY: %s"\n' "$KEY" | curl -sS -m 60 -K - -X "$1" -w '\n%{http_code}' "$API$2")" \
      || die "$1 $2: no answer (network)"
  fi
  code="${out##*$'\n'}"
  out="${out%$'\n'*}"
  case "$code" in
    2??) printf '%s' "$out" ;;
    *) die "$1 $2 answered HTTP $code: $(printf '%s' "$out" | head -c 400)" ;;
  esac
}

# METHOD PATH JSON-or-"" PYTHON-CODE [ARGS...]: the body is fetched first and parsed after, so a refused call stops
# with its own reason (an HTTP 409 OUT_OF_STOCK) instead of a JSON error from an empty pipe.
callpy() {
  local m="$1" p="$2" d="$3" code="$4" out
  shift 4
  out="$(call "$m" "$p" "$d")" || exit 1
  printf '%s' "$out" | "$PY" -c "$code" "$@"
}

# The JSON readers (python3 from the OS: no project env needed on the Mac side).
read -r -d '' PY_LIVE <<'EOF' || true
import json, sys
name, want = sys.argv[1], sys.argv[2]
live = [i for i in json.load(sys.stdin).get("instances", []) if i.get("name") == name and i.get("status") != "deleted"]
if want == "exists":
    sys.exit(0 if live else 1)
if len(live) != 1:
    sys.exit("shade: %d live instances named %r" % (len(live), name))
print(live[0]["id"])
EOF
read -r -d '' PY_INFO <<'EOF' || true
import json, sys
i = json.load(sys.stdin)
p = i.get("hourly_price")
print(i.get("status"), i.get("ip") or "-", i.get("ssh_user") or "-", i.get("ssh_port") or 22,
      "%.2f" % (p / 100.0) if p is not None else "-", i.get("cost_estimate") or "0",
      (i.get("status_details") or "-").replace(" ", "_"))
EOF
read -r -d '' PY_TYPES <<'EOF' || true
import json, sys
for t in json.load(sys.stdin).get("instance_types", []):
    regions = [a["region"] for a in t.get("availability", [])
               if a.get("available") and a.get("rental_type", "on_demand") == "on_demand"]
    if not regions:
        continue
    c = t.get("configuration", {})
    print("{:15s} {:24s} {}x{:10s} vcpu={:<4} ram={:<5} disk={:<5} ${:.2f}/h  regions={}".format(
        t["cloud"], t["shade_instance_type"], c.get("num_gpus"), str(c.get("gpu_type")), c.get("vcpus"),
        c.get("memory_in_gb"), c.get("storage_in_gb"), t["hourly_price"] / 100.0, ",".join(regions)))
EOF
read -r -d '' PY_LS <<'EOF' || true
import json, sys
for i in json.load(sys.stdin).get("instances", []):
    p = i.get("hourly_price")
    print("{:26s} {:16s} {:13s} {:20s} {:16s} {:9s} spent ${} {}".format(
        i.get("name", ""), i.get("status", ""), i.get("cloud", ""), i.get("shade_instance_type", ""), i.get("ip") or "-",
        "$%.2f/h" % (p / 100.0) if p is not None else "", i.get("cost_estimate") or 0, i["id"]))
EOF
read -r -d '' PY_LSJSON <<'EOF' || true
import json, sys
out = []
for i in json.load(sys.stdin).get("instances", []):
    p = i.get("hourly_price")
    out.append({"name": i.get("name"), "status": i.get("status"), "cloud": i.get("cloud"),
                "type": i.get("shade_instance_type"), "hourly_usd": p / 100.0 if p is not None else None,
                "cost_usd": i.get("cost_estimate"), "id": i.get("id"), "created_at": i.get("created_at")})
print(json.dumps(out))
EOF
read -r -d '' PY_HOST <<'EOF' || true
import os, re, sys
path, name, ip, user, port, key = sys.argv[1:]
text = open(path).read() if os.path.exists(path) else ""
text = re.sub(r"(?m)^Host " + re.escape(name) + r"\n(?:[ \t]+.*\n?)*", "", text)
text += ("Host %s\n  HostName %s\n  User %s\n  Port %s\n  IdentityFile %s\n  IdentitiesOnly yes\n"
         "  StrictHostKeyChecking accept-new\n  ServerAliveInterval 60\n") % (name, ip, user, port, key)
tmp = path + ".tmp"
with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
    f.write(text)
os.replace(tmp, path)
EOF
read -r -d '' PY_UNHOST <<'EOF' || true
import os, re, sys
path, name = sys.argv[1:]
if os.path.exists(path):
    text = re.sub(r"(?m)^Host " + re.escape(name) + r"\n(?:[ \t]+.*\n?)*", "", open(path).read())
    tmp = path + ".tmp"
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write(text)
    os.replace(tmp, path)
EOF
read -r -d '' PY_INCLUDE <<'EOF' || true
import os, sys
path, line = sys.argv[1:]
old = open(path).read() if os.path.exists(path) else ""
tmp = path + ".tmp"
with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
    f.write(line + "\n" + old)
os.replace(tmp, path)
EOF

# NAME or id -> the id of the one live instance by that name.
id_of() {
  if [[ "$1" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then echo "$1"; return; fi
  callpy GET /instances "" "$PY_LIVE" "$1" id
}

# NAME IP USER PORT -> its host block in $HOSTS (replacing an older one), and the Include line in ~/.ssh/config once.
write_host() {
  "$PY" -c "$PY_HOST" "$HOSTS" "$1" "$2" "$3" "$4" "$KEYFILE"
  local line="Include \"$HOSTS\""
  if ! grep -qxF "$line" "$HOME/.ssh/config" 2>/dev/null; then
    "$PY" -c "$PY_INCLUDE" "$HOME/.ssh/config" "$line"
    echo "added to the top of ~/.ssh/config: $line"
  fi
  ssh-keygen -R "$2" >/dev/null 2>&1 || true   # a recycled IP must not trip an old host key
}

case "${1:-}" in
  ""|-h|--help)
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    ;;

  --store)
    [ -t 0 ] || die "--store reads the key at a hidden prompt: run it in your own terminal, not from a session"
    echo "Paste the Shadeform API key (platform.shadeform.ai -> Settings -> API). Nothing is shown as you type."
    # `-w` last, with no value: security prompts for it, so the key never becomes argv.
    security add-generic-password -U -a "$ACCOUNT" -s "$SERVICE" -w
    [ -n "$(read_key)" ] || die "nothing stored"
    echo "stored in the login Keychain as \`$SERVICE\`. Next: scripts/shade.sh --check"
    ;;

  --check)
    need_key
    n="$(callpy GET /instances "" 'import json,sys; print(len(json.load(sys.stdin).get("instances", [])))')"
    echo "OK -- the key answers; $n instance(s) on the account"
    ;;

  types)
    need_key
    q="available=true&sort=price"
    [ -n "${2:-}" ] && q="$q&gpu_type=$2"
    [ -n "${3:-}" ] && q="$q&num_gpus=$3"
    callpy GET "/instances/types?$q" "" "$PY_TYPES"
    ;;

  typeinfo)
    need_key
    callpy GET "/instances/types?cloud=${2:?usage: typeinfo CLOUD SHADE_TYPE}" "" '
import json, sys
for t in json.load(sys.stdin).get("instance_types", []):
    if t.get("shade_instance_type") == sys.argv[1]: print(json.dumps(t, indent=1))' "${3:?shade type}"
    ;;

  sshkey)
    need_key
    [ -f "$KEYFILE" ] || ssh-keygen -q -t ed25519 -N "" -C "sidechain-shadeform" -f "$KEYFILE"
    pub="$(cat "$KEYFILE.pub")"
    have="$(callpy GET /sshkeys "" '
import json, sys
pub = sys.argv[1].split()[1]
print(next((k["id"] for k in json.load(sys.stdin).get("ssh_keys", []) if pub in k.get("public_key", "")), ""))' "$pub")"
    if [ -z "$have" ]; then
      body="$("$PY" -c 'import json,sys; print(json.dumps({"name": "sidechain-mac", "public_key": sys.argv[1]}))' "$pub")"
      have="$(callpy POST /sshkeys/add "$body" 'import json,sys; print(json.load(sys.stdin)["id"])')"
      echo "registered $KEYFILE.pub as $have"
    else
      echo "already registered as $have"
    fi
    call POST "/sshkeys/$have/setdefault" >/dev/null
    echo "default key: $have"
    ;;

  create)
    need_key
    name="${2:?usage: create NAME CLOUD SHADE_TYPE REGION}"; cloud="${3:?cloud}"; stype="${4:?shade instance type}"
    region="${5:?region}"
    [ -f "$KEYFILE" ] || die "no $KEYFILE: run \`scripts/shade.sh sshkey\` first"
    live="$(call GET /instances)" || exit 1
    if printf '%s' "$live" | "$PY" -c "$PY_LIVE" "$name" exists; then die "an instance named $name is already live"; fi
    body="$("$PY" -c '
import json, sys
n, c, t, r, o = sys.argv[1:]
print(json.dumps({"cloud": c, "region": r, "shade_instance_type": t, "shade_cloud": True, "name": n,
                  "os": o}))' "$name" "$cloud" "$stype" "$region" "${SHADE_OS:-ubuntu22.04_cuda12.2_shade_os}")"
    id="$(callpy POST /instances/create "$body" 'import json,sys; print(json.load(sys.stdin)["id"])')"
    echo "$name $id creating ($cloud $stype $region). Next: scripts/shade.sh wait $name"
    ;;

  wait)
    need_key
    name="${2:?usage: wait NAME [minutes, default 20]}"; mins="${3:-20}"
    id="$(id_of "$name")"
    st="unknown"; details="-"
    for _ in $(seq 1 $((mins * 4))); do
      line="$(info_line_out="$(call GET "/instances/$id/info")" && printf '%s' "$info_line_out" | "$PY" -c "$PY_INFO")" || line=""
      read -r st ip user port usd cost details <<<"${line:-unknown - - 22 - 0 -}"
      case "$st" in
        active)
          if [ "$ip" != "-" ]; then
            write_host "$name" "$ip" "$user" "$port"
            echo "$name active at $ip as $user, \$$usd/h; host entry written: ssh $name"
            exit 0
          fi ;;
        error|deleted|deleting) die "$name is $st ($details)" ;;
      esac
      sleep 15
    done
    die "$name not active after $mins minutes (last: $st $details)"
    ;;

  ls)
    need_key
    if [ "${2:-}" = "--json" ]; then callpy GET /instances "" "$PY_LSJSON"; else callpy GET /instances "" "$PY_LS"; fi
    ;;

  info)
    need_key
    id="$(id_of "${2:?usage: info NAME}")"
    line="$(callpy GET "/instances/$id/info" "" "$PY_INFO")"
    echo "$2 $id $line"
    ;;

  delete)
    need_key
    name="${2:?usage: delete NAME}"
    id="$(id_of "$name")"
    call POST "/instances/$id/delete" >/dev/null
    "$PY" -c "$PY_UNHOST" "$HOSTS" "$name"
    echo "$name $id delete requested, its host entry removed; \`scripts/shade.sh info $id\` reads deleted when it is gone"
    ;;

  *)
    die "unknown command: $1 (--store, --check, types, sshkey, create, wait, ls, info, delete)"
    ;;
esac
