# Staying in step with colibrì

sabrewing is a hand-imported fork with its own history, so we don't merge from
upstream — we **vendor** specific files. The substrate (GLM engine, safetensors
reader, tokenizer, CUDA kernels, tiering) is colibrì's, and we want its fixes;
the Inkling engine and everything around it is ours.

## The three file classes

**VENDORED** — taken verbatim from upstream, no local changes. `glm.c`, `st.h`,
`json.h`, `tok_unicode.h`, `compat.h`, `tier.h`, `backend_cuda.*`,
`backend_loader.c`, `coli`, and friends. `tools/upstream-sync.sh pull` adopts
upstream's version.

**MERGED** — we changed them, usually with a matching upstream PR open:
`Makefile` (inkling targets), `openai_server.py` (inkling arch + serve). Sync
only *reports* drift; you merge by hand. **When the upstream PR lands, move the
file to VENDORED and pull** — the upstream version then contains our change and
the divergence disappears. (`tok.h` followed this path: o200k landed upstream
as colibri PR #330, so it is VENDORED now.)

**OWNED** — sabrewing-only, sync never touches: `inkling.c`,
`backend_cuda_ink.*`, `tok_unicode_o200k.h`, `swing`,
`tools/convert_inkling_int4.py`, `tools/make_tiny_inkling.py`, `docs/`
(except `ENVIRONMENT.md` and `grammar-draft.md`, which document the vendored
substrate and are pulled from upstream), `README.md`, `NOTICE`, `web/`
(rewritten in Vue), `.github/workflows/ci.yml`.

## Routine

```sh
tools/upstream-sync.sh check          # what drifted?
tools/upstream-sync.sh pull           # adopt upstream for VENDORED files
make -C c inkling && make -C c check  # rebuild + oracle before committing
tools/upstream-sync.sh diff c/tok.h   # inspect a MERGED file's drift
```

Do this on a branch, run the oracle (`SNAP=./glm_tiny TF=1 ./c/glm 64 16 16`
and the tiny-inkling test) before merging — a vendored change to `st.h` or
`tok.h` can affect both engines.

## The virtuous loop

Improvements that belong upstream go upstream first (PRs #232, #312, #330, #331).
Once merged, sabrewing pulls them back through the VENDORED path. Net effect: the
shared substrate stays a single codebase with two front doors, and sabrewing
never carries a permanent private patch to a file colibrì also owns.
