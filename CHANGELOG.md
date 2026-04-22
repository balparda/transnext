<!-- SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Changelog

All notable changes to this project will be documented in this file.

- [Changelog](#changelog)
  - [1.1.0 - 2026-04-22](#110---2026-04-22)
  - [1.0.0 - 2026-04-17](#100---2026-04-17)

This project follows a pragmatic versioning approach:

- **Patch**: bug fixes / docs / small improvements.
- **Minor**: new features or non-breaking changes.
- **Major**: breaking changes (e.g., DB schema changes, CLI flag renames).

## 1.1.0 - 2026-04-22

- Added
  - `experiment` CLI app (console script `experiment`) with `new` and `markdown` commands
  - `experiment new` command — systematic XY grid generation experiments that vary one or more axes (CFG scale, sampler, model, positive/negative prompt) across a set of seed values, producing labeled XY grid images for visual comparison of results
  - `experiment markdown` command — auto-generate `experiment.md` CLI documentation
  - `core/newton.py` module — `Experiments` and `Experiment` classes managing experiment lifecycle: axis definition, Cartesian product of keys, result tracking per seed, and `Grid()` for building labeled XY grid images
  - `--seeds` option for `experiment new`: pipe-separated seed values (e.g., `"42|-1|999"`); `-1` generates a random seed each time
  - `--axis` option (repeatable) for `experiment new`: axis definitions in `"KEY:VAL1|VAL2|..."` format; valid keys: `cfg_scale`, `sampler`, `model_hash`, `positive`, `negative`
  - `--redo`/`--no-redo` flag added to `gen make`, `gen sync`, and `experiment new` commands to force re-generation/re-sync even when the output is already in the DB
  - Sidecar model options added as global flags to both `gen` and `experiment` apps:
    - `--sidecar`/`--no-sidecar`: save/load `.transnext.json` sidecar files alongside model files for per-model metadata (default: `--sidecar`)
    - `--respect-vae`/`--no-respect-vae`: apply VAE override from model sidecar (default: `--respect-vae`)
    - `--respect-pony`/`--no-respect-pony`: add Pony quality prefix to positive prompt when model sidecar says the model is a Pony model (default: `--respect-pony`)
    - `--respect-clip2`/`--no-respect-clip2`: automatically set CLIP skip to 20 when model sidecar says the model uses CLIP2 (default: `--respect-clip2`)
  - AutoV3 hash support for LoRA/LyCORIS model entries (compatible with A1111/kohya-ss `addnet_hash_safetensors` format)
  - Embedding detection and tracking: `p_embeddings` and `n_embeddings` fields in `AIMetaType` hold sorted lists of embedding names detected in positive/negative prompts; embedding inventory in the DB
  - New helper scripts:
    - `scripts/show_errors.py` — explore and triage DB parse errors interactively (read-only)
    - `scripts/clean_db_leave_models.py` — safety-guarded script to wipe all images/experiments from the DB while retaining the model and lora inventory

- Changed
  - Test suite expanded from 181 to 299 tests (+65%) across 10 test files, adding `tests/core/newton_test.py` (65 tests) and `tests/cli/cliexperiment_test.py` (15 tests)
  - `CHANGELOG.md` versioning description updated to align with the project's actual scheme
  - Both `gen` and `experiment` apps now share the same full set of global flags (host, port, db, sidecar, out)
  - `experiment.md` auto-generated CLI docs added alongside the existing `gen.md`

- Fixed
  - Wrong docstring in `scripts/clean_db_leave_models.py` (was copy-pasted from `show_errors.py`)

## 1.0.0 - 2026-04-17

Initial release.

- Repo is live, with `gen` application.
- `gen make` command for text-to-image generation via the SDNext API
- `gen sync` command for scanning directories, importing images, parsing embedded PNG metadata (SDNext/A1111 formats), tracking duplicates, and detecting deleted files
- `gen markdown` command for auto-generating CLI documentation
- Core modules: `core/base.py` (constants, enums, CLI option definitions), `core/db.py` (`AIDatabase`, TypedDict schemas, import/sync/metadata parsing), `core/sdnapi.py` (SDNext HTTP API client)
- Full `make` command flags: `--negative`, `--iterations`, `--seed`, `--vseed`, `--vstrength`, `--width`, `--height`, `--sampler` (50+ samplers), `--parser`, `--model`, `--clip`, `--cfg`, `--cfg-end`, `--cfg-rescale`, `--sigma`, `--spacing`, `--beta`, `--prediction`, `--freeu`/`--no-freeu`, `--b1`/`--b2`/`--s1`/`--s2`, `--backup`/`--no-backup`
- `sync` command `--force-api`/`--no-force-api` flag for optional API connectivity
- SDNext API version detection at startup with mixed API version support
- Encrypted local database (`AIDatabase`) with safe-save concurrent write protection
- Image origin detection (SDNext, A1111, or unknown) and version tracking
- Metadata parsing for: positive/negative prompts, seed, variation seed/strength, steps, sampler, parser, CFG scale/end/rescale, CLIP skip, model hash, image dimensions, LoRA/LyCORIS references, scheduler options (sigma/spacing/beta/prediction type), NGMS, CFG skip, FreeU parameters, and img2img denoising info
- Duplicate image tracking by SHA-256 hash with alternate path support
- Model/LoRA/LyCORIS inventory management from SDNext API with hash-based lookup and substring name search
- Real test images in `tests/data/images/` for end-to-end metadata parsing validation
- `gen reproduce` command to re-generate an existing DB image by its SHA-256 hash or original file path, preserving all original generation parameters (model, prompt, seed, sampler, etc.)
- 181 unit tests across 8 test files (3217+ LOC): `base_test.py` (31), `db_test.py` (67), `sdnapi_test.py` (55), `make_test.py` (9), `sync_test.py` (5), `reproduce_test.py` (7), `gen_test.py` (4), `semi_integration_test.py` (1)
