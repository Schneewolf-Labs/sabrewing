#!/usr/bin/env bash
# upstream-sync — keep sabrewing's shared substrate in step with colibrì.
#
# sabrewing was hand-imported (fresh history), so we don't merge upstream —
# we vendor specific files from it. Files fall into three classes:
#
#   VENDORED  taken verbatim from upstream; sync pulls upstream's version.
#   MERGED    we changed them AND (often) have a pending upstream PR; sync
#             only shows the diff — you merge by hand. When the PR lands,
#             adopt upstream's version (it will contain our change).
#   OWNED     sabrewing-only; sync never touches them.
#
# Usage:
#   tools/upstream-sync.sh check        # report drift (default, no changes)
#   tools/upstream-sync.sh pull         # adopt upstream for VENDORED files
#   tools/upstream-sync.sh diff <file>  # show a MERGED file's drift vs upstream
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE=colibri
BRANCH=dev

VENDORED=(
  c/glm.c c/olmoe.c c/st.h c/json.h c/tok_unicode.h c/compat.h c/tier.h
  c/grammar.h c/schema_gbnf.h c/decode_batch.h c/backend_cuda.cu c/backend_cuda.h
  c/backend_metal.h c/backend_metal.mm c/backend_loader.c c/iobench.c
  c/coli c/doctor.py c/resource_plan.py
)
# we modified these; a pending upstream PR may supersede our change
MERGED=( c/tok.h c/Makefile c/openai_server.py )
# note: MERGED files with upstream PRs — tok.h→#330. When merged upstream,
# move the file to VENDORED and pull.

git remote get-url "$REMOTE" >/dev/null 2>&1 || {
  echo "adding $REMOTE remote"; git remote add "$REMOTE" https://github.com/JustVugg/colibri.git; }
git fetch -q "$REMOTE" "$BRANCH"
REF="$REMOTE/$BRANCH"

cmd="${1:-check}"

case "$cmd" in
  check)
    echo "== VENDORED (sync pulls upstream) =="
    for f in "${VENDORED[@]}"; do
      if ! git cat-file -e "$REF:$f" 2>/dev/null; then echo "  gone   $f (removed upstream?)"; continue; fi
      git diff --quiet "$REF:$f" "HEAD:$f" 2>/dev/null && echo "  ok     $f" || echo "  UPDATE $f"
    done
    echo "== MERGED (review by hand: tools/upstream-sync.sh diff <file>) =="
    for f in "${MERGED[@]}"; do
      git diff --quiet "$REF:$f" "HEAD:$f" 2>/dev/null && echo "  ok     $f" || echo "  DRIFT  $f"
    done
    ;;
  pull)
    changed=0
    for f in "${VENDORED[@]}"; do
      git cat-file -e "$REF:$f" 2>/dev/null || continue
      if ! git diff --quiet "$REF:$f" "HEAD:$f" 2>/dev/null; then
        git checkout "$REF" -- "$f" && echo "  pulled $f" && changed=1
      fi
    done
    [ "$changed" = 1 ] && echo "review, build (make -C c inkling && make -C c check), then commit" || echo "nothing to pull"
    ;;
  diff)
    f="${2:?usage: upstream-sync.sh diff <file>}"
    git diff "$REF:$f" "HEAD:$f"
    ;;
  *) echo "usage: $0 {check|pull|diff <file>}"; exit 1 ;;
esac
