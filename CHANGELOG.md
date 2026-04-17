<!-- SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Changelog

All notable changes to this project will be documented in this file.

- [Changelog](#changelog)
  - [V.V.V - YYYY-MM-DD - Placeholder](#vvv---yyyy-mm-dd---placeholder)
  - [1.1.0 - 2026-04-17](#110---2026-04-17)
  - [1.0.0 - 2026-04-10](#100---2026-04-10)

This project follows a pragmatic versioning approach:

- **Patch**: bug fixes / docs / small improvements.
- **Minor**: new template features or non-breaking developer workflow changes.
- **Major**: breaking template changes (e.g., required file/command renames).

## V.V.V - YYYY-MM-DD - Placeholder

- Added
  - Placeholder for future changes.

- Changed
  - Placeholder for future changes.

- Fixed
  - Placeholder for future changes.

## 1.1.0 - 2026-04-17

Substantial feature expansion of both generation and sync capabilities.

- Added
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
  - 186 unit tests across 6 test files (3217 LOC): `base_test.py` (46), `db_test.py` (65), `sdnapi_test.py` (58), `make_test.py` (8), `sync_test.py` (5), `gen_test.py` (4)

- Changed
  - Default image dimensions now 1024x1024 (SDXL native resolution) instead of 512x512
  - Seed range expanded to full 64-bit: 1 to 18446744073709551615
  - Query normalization moved to the database layer (consistent across API and import paths)
  - All integer storage uses multiplied values for lossless precision (CFG ×10, CFG rescale ×100, CLIP skip ×10, FreeU ×100, etc.)

## 1.0.0 - 2026-04-10

Initial release.

- Repo is live, with `gen` application.
