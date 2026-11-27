#!/usr/bin/env bash
# Run a command; if it fails, post the end of its output as one annotation on the run
# (readable on the run's page without downloading the logs), then fail the step.
# Usage: .github/report.sh <title> <command> [args...]
title=$1
shift
out=$(mktemp)
if "$@" >"$out" 2>&1; then
    tail -n 5 "$out"
    exit 0
fi
cat "$out"
msg=$(grep -E "^(FAILED|ERROR)|Error|error:|would reformat|Traceback" "$out" | head -n 25)
msg="$msg
--- last lines ---
$(tail -n 25 "$out")"
echo "::error title=$title failed::$(printf '%s' "$msg" | sed 's/%/%25/g' | awk '{printf "%s%%0A", $0}')"
exit 1
