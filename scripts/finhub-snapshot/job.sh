#!/bin/bash
# job.sh — install / control the daily finhub snapshot job (macOS launchd)
#
#   job.sh start       turn the schedule ON (installs first if needed; also re-enables after `stop`)
#   job.sh stop        turn the schedule OFF now and keep it off at login (data and install are kept)
#   job.sh install     create ~/Developer/finhub-snapshot (venv + code), install the LaunchAgent
#   job.sh run         record a snapshot now (ignores the time guards; for testing)
#   job.sh status      is the agent loaded, last rows recorded, recent log
#   job.sh logs [-f]   show / follow the log
#   job.sh sync        copy the recorded CSVs into this repo's data/macro/snapshots/
#   job.sh uninstall   remove the LaunchAgent (keeps the recorded data)
#
# WHY ~/Developer: macOS blocks launchd jobs from reading ~/Desktop, ~/Documents and ~/Downloads
# (Python hangs silently). So the venv, the script and the data live outside Desktop. `sync` is run
# by you in Terminal, where Desktop access is allowed.
#
# Schedule: Mon-Fri at 15:10, 16:30 and 20:00 (VN time). The extra runs catch a Mac that was asleep;
# snapshot.py does nothing if today is already recorded.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FINHUB_SRC="${FINHUB_SRC:-$(cd "$REPO/../finhub-mcp" 2>/dev/null && pwd)}"
BASE="${SNAPSHOT_HOME:-$HOME/Developer/finhub-snapshot}"
PY="$BASE/.venv/bin/python"
LABEL="com.finhub.snapshot"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGDIR="$HOME/Library/Logs/finhub-snapshot"
TARGET="gui/$(id -u)/$LABEL"
say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

render_plist() {
  local entries=""
  for wd in 1 2 3 4 5; do
    for hm in "15 10" "16 30" "20 0"; do
      set -- $hm
      entries="$entries
    <dict><key>Weekday</key><integer>$wd</integer><key>Hour</key><integer>$1</integer><key>Minute</key><integer>$2</integer></dict>"
    done
  done
  cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$PY</string><string>$BASE/snapshot.py</string></array>
  <key>WorkingDirectory</key><string>$BASE</string>
  <key>StartCalendarInterval</key>
  <array>$entries
  </array>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>$LOGDIR/snapshot.log</string>
  <key>StandardErrorPath</key><string>$LOGDIR/snapshot.log</string>
  <key>EnvironmentVariables</key><dict><key>PYTHONUNBUFFERED</key><string>1</string></dict>
</dict>
</plist>
EOF
}

cmd_install() {
  [ "$(uname -s)" = Darwin ] || die "launchd is macOS only"
  [ -n "$FINHUB_SRC" ] && [ -f "$FINHUB_SRC/pyproject.toml" ] || die "finhub-mcp not found (set FINHUB_SRC=/path/to/finhub-mcp)"
  case "$BASE" in "$HOME/Desktop"*|"$HOME/Documents"*|"$HOME/Downloads"*) die "SNAPSHOT_HOME must be outside Desktop/Documents/Downloads";; esac
  mkdir -p "$BASE/data" "$LOGDIR" "$HOME/Library/LaunchAgents"
  if [ ! -x "$PY" ]; then
    PYBIN="$(command -v python3.14 || command -v python3.13 || command -v python3.12 || command -v python3.11)" || die "need Python 3.11+"
    say "creating venv with $PYBIN"; "$PYBIN" -m venv "$BASE/.venv" || die "venv failed"
  fi
  say "installing dependencies (a minute or two)..."
  "$PY" -m pip install -q --upgrade pip >/dev/null 2>&1
  "$PY" -m pip install -q -r "$FINHUB_SRC/requirements.txt" -r "$FINHUB_SRC/docker/requirements-vnstock-deps.txt" || die "pip install failed"
  # non-editable copy so launchd never needs to read the Desktop folder; re-run install to update
  "$PY" -m pip install -q --force-reinstall --no-deps "$FINHUB_SRC" || die "finhub-mcp install failed"
  SP="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
  if ! "$PY" -c 'import vnstock' >/dev/null 2>&1; then
    TAR="$FINHUB_SRC/docker/vendor/vnstock-site-packages.tar.gz"
    [ -f "$TAR" ] || die "vnstock 4.x is not installable from PyPI and $TAR is missing (see finhub-mcp README: freeze_vnstock.py)"
    tar -xzf "$TAR" -C "$SP" && say "vnstock restored from vendored copy"
  fi
  "$PY" -c 'import vnstock, finhub_mcp' >/dev/null 2>&1 || die "imports still failing in $BASE/.venv"
  cp "$(dirname "${BASH_SOURCE[0]}")/snapshot.py" "$BASE/snapshot.py"
  render_plist > "$PLIST"
  launchctl bootout "$TARGET" >/dev/null 2>&1
  launchctl enable "$TARGET" >/dev/null 2>&1     # clear a 'disabled' override left by `stop`
  launchctl bootstrap "gui/$(id -u)" "$PLIST" || die "launchctl bootstrap failed"
  say "installed: $LABEL (Mon-Fri 15:10, 16:30, 20:00). Data: $BASE/data  Log: $LOGDIR/snapshot.log"
}

cmd_start() {
  if [ ! -x "$PY" ] || [ ! -f "$PLIST" ] || [ ! -f "$BASE/snapshot.py" ]; then cmd_install; return; fi
  launchctl enable "$TARGET" >/dev/null 2>&1
  if launchctl print "$TARGET" >/dev/null 2>&1; then say "already on ($LABEL)"
  else launchctl bootstrap "gui/$(id -u)" "$PLIST" || die "launchctl bootstrap failed"; say "schedule ON (Mon-Fri 15:10, 16:30, 20:00)"; fi
}
cmd_stop() {
  launchctl bootout "$TARGET" >/dev/null 2>&1
  launchctl disable "$TARGET" >/dev/null 2>&1
  say "schedule OFF — stays off at login. Data kept in $BASE/data. Turn on again with: job.sh start"
}
cmd_run()    { "$PY" "$BASE/snapshot.py" --force "$@"; }
cmd_logs()   { if [ "${1:-}" = -f ]; then tail -f "$LOGDIR/snapshot.log"; else tail -n 40 "$LOGDIR/snapshot.log"; fi; }
cmd_status() {
  if launchctl print "$TARGET" >/dev/null 2>&1; then say "schedule: ON ($LABEL loaded)"
  elif launchctl print-disabled "gui/$(id -u)" 2>/dev/null | grep -Eq "\"$LABEL\" => (disabled|true)"; then say "schedule: OFF (stopped; run job.sh start)"
  else say "schedule: not installed / not loaded"; fi
  for f in vn_foreign_daily vn_breadth_daily macro_daily; do
    p="$BASE/data/$f.csv"; [ -f "$p" ] && say "$f.csv: $(($(wc -l < "$p")-1)) rows, last date $(tail -1 "$p" | cut -d, -f1)" || say "$f.csv: (none yet)"
  done
  [ -f "$LOGDIR/snapshot.log" ] && { say "--- log tail ---"; tail -n 8 "$LOGDIR/snapshot.log"; }
}
cmd_sync() {
  mkdir -p "$REPO/data/macro/snapshots"
  cp "$BASE"/data/*.csv "$REPO/data/macro/snapshots/" 2>/dev/null && say "synced -> $REPO/data/macro/snapshots/" || say "nothing to sync yet"
}
cmd_uninstall() { launchctl bootout "$TARGET" >/dev/null 2>&1; rm -f "$PLIST"; say "agent removed (data kept in $BASE/data)"; }

case "${1:-}" in
  install) cmd_install;; start) cmd_start;; stop) cmd_stop;; run) shift; cmd_run "$@";; status) cmd_status;; logs) shift; cmd_logs "$@";;
  sync) cmd_sync;; uninstall) cmd_uninstall;;
  *) sed -n 2,15p "${BASH_SOURCE[0]}";;
esac
