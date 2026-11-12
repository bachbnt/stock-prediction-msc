#!/bin/bash
# Double-click this file in Finder to run: scripts/finhub-snapshot/job.sh status
cd "$(dirname "$0")" || exit 1
./job.sh status
code=$?
echo
read -n 1 -s -r -p "Press any key to close this window..."
echo
exit $code
