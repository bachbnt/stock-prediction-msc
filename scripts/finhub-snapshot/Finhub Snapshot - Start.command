#!/bin/bash
# Double-click this file in Finder to run: scripts/finhub-snapshot/job.sh start
cd "$(dirname "$0")" || exit 1
./job.sh start
code=$?
echo
read -n 1 -s -r -p "Press any key to close this window..."
echo
exit $code
