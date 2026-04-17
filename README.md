<!-- SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# TransNext

SDXL helper, searcher, maker, based on [SDNext](https://github.com/vladmandic/sdnext) API.

- **Primary use case:** Generate SDXL/Stable Diffusion images via the SDNext API, track them in an encrypted local database, and sync image collections from multiple directories
- **Works with:** Local SDNext server, local filesystem image directories, encrypted JSON database
- **Status:** Stable
- **License:** Apache-2.0

Since version 1.0.0 it is a PyPI package: <https://pypi.org/project/transnext/>

## Table of contents

<!-- TOC is auto-generated, do not edit -->
- [TransNext](#transnext)
  - [Table of contents](#table-of-contents)
  - [License](#license)
    - [Third-party notices](#third-party-notices)
  - [Installation](#installation)
    - [Supported platforms](#supported-platforms)
    - [Known dependencies (Prerequisites)](#known-dependencies-prerequisites)
  - [Context](#context)
    - [What this tool is](#what-this-tool-is)
    - [What this tool is not](#what-this-tool-is-not)
    - [Key concepts and terminology](#key-concepts-and-terminology)
    - [Inputs and outputs](#inputs-and-outputs)
      - [Inputs](#inputs)
      - [Outputs](#outputs)
  - [Design assumptions and disclaimers](#design-assumptions-and-disclaimers)
    - [Assumptions](#assumptions)
    - [Known limitations](#known-limitations)
    - [Privacy and telemetry](#privacy-and-telemetry)
  - [CLI Interface](#cli-interface)
    - [Quick start](#quick-start)
    - [Common workflows](#common-workflows)
      - [Generate an image](#generate-an-image)
      - [Sync images from directories](#sync-images-from-directories)
      - [Reproduce an existing image](#reproduce-an-existing-image)
      - [Generate CLI documentation](#generate-cli-documentation)
    - [Command structure](#command-structure)
    - [Global flags](#global-flags)
    - [`make` command flags](#make-command-flags)
    - [`reproduce` command](#reproduce-command)
    - [`sync` command](#sync-command)
    - [CLI Commands Documentation](#cli-commands-documentation)
    - [Configuration](#configuration)
      - [Config file locations](#config-file-locations)
      - [Environment variables](#environment-variables)
    - [Color and formatting](#color-and-formatting)
    - [Exit codes](#exit-codes)
    - [Logging](#logging)
  - [Project Design](#project-design)
    - [Architecture overview](#architecture-overview)
    - [Modules and packages](#modules-and-packages)
    - [Data flow](#data-flow)
      - [Image generation (`make`)](#image-generation-make)
      - [Image sync (`sync`)](#image-sync-sync)
      - [Image reproduce (`reproduce`)](#image-reproduce-reproduce)
    - [Error handling](#error-handling)
    - [Security model](#security-model)
  - [Development Instructions](#development-instructions)
    - [File structure](#file-structure)
    - [Development Setup](#development-setup)
      - [Install Python](#install-python)
      - [Install Poetry (recommended: `pipx`)](#install-poetry-recommended-pipx)
      - [Make sure `.venv` is local](#make-sure-venv-is-local)
      - [Get the repository](#get-the-repository)
      - [Create environment and install dependencies](#create-environment-and-install-dependencies)
      - [Optional: VSCode setup](#optional-vscode-setup)
    - [Build and run](#build-and-run)
    - [Testing](#testing)
      - [Unit tests / Coverage](#unit-tests--coverage)
      - [Instrumenting your code](#instrumenting-your-code)
      - [Integration / e2e tests](#integration--e2e-tests)
    - [Linting / formatting / static analysis](#linting--formatting--static-analysis)
      - [Type checking](#type-checking)
    - [Documentation updates](#documentation-updates)
    - [Versioning and releases](#versioning-and-releases)
      - [Versioning scheme](#versioning-scheme)
      - [Updating versions](#updating-versions)
        - [Bump project version (patch/minor/major)](#bump-project-version-patchminormajor)
        - [Update dependency versions](#update-dependency-versions)
        - [Exporting the `requirements.txt` file](#exporting-the-requirementstxt-file)
        - [CI and docs](#ci-and-docs)
        - [Git tag and commit](#git-tag-and-commit)
        - [Publish to PyPI](#publish-to-pypi)
  - [Security](#security)
  - [Troubleshooting](#troubleshooting)
    - [Enable debug output](#enable-debug-output)
    - [Common issues](#common-issues)
  - [Glossary](#glossary)

## License

Copyright 2026 Daniel Balparda & BellaKeri <balparda@github.com>

Licensed under the ***Apache License, Version 2.0*** (the "License"); you may not use this file except in compliance with the License. You may obtain a [copy of the License here](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

### Third-party notices

This project depends on the following key third-party software:

- [SDNext](https://github.com/vladmandic/sdnext) — The Stable Diffusion server this tool communicates with (not bundled)
- [transai](https://pypi.org/project/transai/) — AI utilities library
- [transcrypto](https://github.com/balparda/transcrypto) — CLI modules, encryption, hashing, config management
- [Pillow (PIL)](https://pypi.org/project/pillow/) — Image processing (via transai)
- [Rich](https://pypi.org/project/rich/) — Terminal output formatting (via transai)
- [Typer](https://pypi.org/project/typer/) — CLI framework (via transai)
- [Requests](https://pypi.org/project/requests/) — HTTP client for API communication

## Installation

To install from PyPI:

```sh
pip3 install transnext
```

For development, see [Development Setup](#development-setup).

### Supported platforms

- **OS:** Linux, macOS (any platform that runs Python 3.13+ and can reach an SDNext server)
- **Architectures:** x86_64, arm64
- **Python:** >= 3.13

### Known dependencies (Prerequisites)

- **[Python 3.13+](https://python.org/)** — [documentation](https://docs.python.org/3.13/)
- **[transai 1.2+](https://pypi.org/project/transai/)** — AI utilities (brings in `typer`, `rich`, `transcrypto`, `Pillow`, etc.)
- **[requests 2.33+](https://pypi.org/project/requests/)** — HTTP client for SDNext API communication
- **[transcrypto 2.1+](https://pypi.org/project/transcrypto/)** — CLI modules, logging, humanization, crypto, random, hash, serialization, config management — [documentation](https://github.com/balparda/transcrypto)
- A running **[SDNext](https://github.com/vladmandic/sdnext)** server with API enabled (default: `http://127.0.0.1:7860`)

## Context

### What this tool is

TransNext is a CLI tool for generating AI images using Stable Diffusion SDXL models via the [SDNext](https://github.com/vladmandic/sdnext) API. It provides rich control over generation parameters (prompts, sampler, CFG scale/end/rescale, seed, variation seed, dimensions, CLIP skip, scheduler options, FreeU scaling, LoRA tracking, etc.), stores every generated image along with its full generation metadata in an optionally-encrypted local database, and can sync/import existing image collections from multiple directories — detecting duplicates, parsing embedded SDNext/A1111 PNG metadata, and keeping track of file locations.

### What this tool is not

- Not a Stable Diffusion server — it is a **client** for SDNext; you must run SDNext separately
- Not a web UI — it is a command-line tool
- Not a general image editor or converter

### Key concepts and terminology

- **SDNext API** — The HTTP API exposed by the [SDNext](https://github.com/vladmandic/sdnext) Stable Diffusion server (based on Automatic1111's API)
- **AIDatabase** — TransNext's local encrypted database that tracks all generated and imported images with their metadata
- **Image hash** — SHA-256 of the image file on disk, used as the primary key in the database
- **Raw hash** — SHA-256 of the decoded RGBA pixel data, format-agnostic for cross-format duplicate detection
- **Model hash** — SHA-256 of the model weights file on disk, used to identify which model generated an image
- **CFG scale** — Classifier-Free Guidance scale; controls how strongly the image follows the prompt (1.0-30.0)
- **CFG end** — Fraction of total steps at which CFG guidance stops being applied (0.0-1.0)
- **CFG rescale** — Reduces overexposure/oversaturation tendency at high CFG values (0.0-1.0)
- **CLIP skip** — Number of CLIP layers to skip during prompt encoding (1-12)
- **Sampler** — The diffusion sampling algorithm (50+ options; e.g., Euler, DPM++ SDE, UniPC, Heun, LCM)
- **Parser** — The prompt attention/weighting parser (native, compel, xhinker, a1111, fixed)
- **Scheduler options** — Controls for the noise schedule: sigma type, spacing, beta schedule, and prediction type
- **FreeU** — Backbone and skip-connection feature scaling that can improve generation quality without retraining (b1, b2, s1, s2 parameters)
- **Variation seed** — A secondary seed that can be blended with the primary seed at a configurable strength to produce subtle variations of an image
- **LoRA / LyCORIS** — Lightweight model adaptation layers; detected and tracked from both API model inventory and embedded PNG metadata

### Inputs and outputs

#### Inputs

- **stdin:** not supported
- **Network/API:** SDNext server at configurable host:port (default `http://127.0.0.1:7860`)
- **Files:** Image files (PNG, JPEG, GIF) from local directories for sync/import
- **Environment variables:** `SDAPI_URL` overrides default host:port; `NO_COLOR` disables colored output
- **Config:** Encrypted database file managed by `transcrypto.utils.config` at OS-native locations

#### Outputs

- **stdout:** Human-readable Rich-formatted output (colored by default)
- **stderr:** Logging at configurable verbosity levels
- **Files:** Generated PNG images saved to date-based subdirectories (`YYYY-MM-DD/`) under the output root; encrypted database file

## Design assumptions and disclaimers

### Assumptions

- A running SDNext server is accessible at the configured host:port
- The user has filesystem read/write access to the output directory and image source directories
- Encoding: UTF-8
- Image dimensions must be multiples of 8 (default: 1024x1024 for SDXL)

### Known limitations

- Only supports text-to-image generation (`txt2img`); no img2img, inpainting, or other modes
- Single image per generation call (batch size = 1)
- API calls use `verify=False` for TLS (SDNext often runs on localhost without certificates)
- SDNext API timeout is 300 seconds per call

### Privacy and telemetry

- **Telemetry:** None. TransNext collects no data and makes no network calls except to the configured SDNext server.
- All data is stored locally in an optionally-encrypted database file

## CLI Interface

### Quick start

Generate an image with default settings:

```sh
poetry run gen --out ~/my-images make "a beautiful sunset over mountains"
```

Generate with more control:

```sh
poetry run gen -vv --out ~/my-images make "dark knight in the rain" \
  -n "batman, comic" --cfg 7.5 -m SDXL_model -i 30 --sampler "Euler a" -w 800 -h 800
```

Sync existing images into the database:

```sh
poetry run gen sync ~/path/to/existing/images
```

### Common workflows

#### Generate an image

```sh
poetry run gen -vv --out ~/foo/bar make "dark knight" -n batman \
  --cfg 7.5 -m SDXL_model_1234 -i 30 --sampler "Euler a"
```

This will:

1. Connect to the SDNext API
2. Look up the model by name (fetching from the server if not in the DB)
3. Generate a 1024x1024 image with the given parameters
4. Save the PNG to `~/foo/bar/YYYY-MM-DD/<hash>-<timestamp>-<model>-<cfg>-<steps>-<w>-<h>-<seed>-<img-hash>.png`
5. Store the full metadata in the encrypted database

#### Sync images from directories

```sh
poetry run gen sync                       # sync all known directories
poetry run gen sync ~/foo/bar/new/dir     # add a new directory and sync everything
```

Sync scans all known image directories, detects new and deleted images, parses embedded SDNext/A1111 metadata from PNGs, tracks duplicates (same hash at multiple paths), and updates the database.

#### Reproduce an existing image

```sh
poetry run gen reproduce abc123def456              # reproduce by image hash
poetry run gen reproduce ~/foo/bar/image.png       # reproduce by file path (resolved to hash via DB)
```

Looks up the image by hash (or resolves a path to its hash), then calls the SDNext API again with the exact same generation parameters stored in the DB. The reproduced image is stored as a new entry in the database. Requires DB access (`--db`).

#### Generate CLI documentation

```sh
poetry run gen markdown > gen.md
```

### Command structure

```sh
gen [global flags] <command> [command flags] [args]
```

### Global flags

| Flag | Description | Default |
| --- | --- | --- |
| `--help` | Show help | off |
| `--version` | Show version and exit | off |
| `-v`, `-vv`, `-vvv`, `--verbose` | Verbosity (nothing=*ERROR*, `-v`=*WARNING*, `-vv`=*INFO*, `-vvv`=*DEBUG*) | *ERROR* |
| `--color`/`--no-color` | Force enable/disable colored output (respects `NO_COLOR` env var) | `--color` |
| `--host` | SDNext API host URL | `http://127.0.0.1` |
| `-p`, `--port` | SDNext API port (0-65535) | `7860` |
| `--db`/`--no-db` | Use/update internal database | `--db` |
| `-o`, `--out` | Output root directory for generated images (creates `YYYY-MM-DD` sub-dirs) | last used (with `--db`) |

### `make` command flags

| Flag | Description | Default |
| --- | --- | --- |
| (argument) `POSITIVE_PROMPT` | Positive prompt string (required) | — |
| `-n`, `--negative` | Negative prompt string | none |
| `-i`, `--iterations` | Number of generation steps (1-200) | `20` |
| `-s`, `--seed` | Random seed (1-18446744073709551615); omit for random | random |
| `--vseed` | Variation seed (1-18446744073709551615); omit to disable variation | none |
| `--vstrength` | Variation strength, how much to mix variation seed with base seed (0.0-1.0) | `0.5` |
| `-w`, `--width` | Image width in pixels (16-4096, must be multiple of 8) | `1024` |
| `-h`, `--height` | Image height in pixels (16-4096, must be multiple of 8) | `1024` |
| `--sampler` | Sampler method (50+ options; e.g., `DPM++ SDE`, `Euler`, `UniPC`, `Heun`, `LCM`, etc.) | `DPM++ SDE` |
| `--parser` | Query parser: native, compel, xhinker, a1111, fixed | `a1111` |
| `-m`, `--model` | Model name (substring match against known models) | `XLB_v10` |
| `--clip` | CLIP skip value (1-12) | `1` |
| `-g`, `--cfg` | CFG scale / guidance scale (1.0-30.0) | `6.0` |
| `--cfg-end` | CFG guidance end fraction (0.0-1.0) | `0.8` |
| `--cfg-rescale` | CFG rescale to reduce overexposure at high CFG (0.0-1.0) | `0.0` |
| `--sigma` | Scheduler sigma schedule: default, karras, betas, exponential, lambdas, flowmatch | none (SDNext default) |
| `--spacing` | Scheduler spacing: default, linspace, leading, trailing | none (SDNext default) |
| `--beta` | Scheduler beta schedule: default, linear, scaled, cosine, sigmoid, laplace | none (SDNext default) |
| `--prediction` | Scheduler prediction type: default, epsilon, sample, v_prediction, flow_prediction | none (SDNext default) |
| `--freeu`/`--no-freeu` | Enable/disable FreeU backbone and skip feature scaling | `--freeu` |
| `--b1` | FreeU b1 backbone feature scale (0.0-3.0) | `1.05` |
| `--b2` | FreeU b2 backbone feature scale (0.0-3.0) | `1.1` |
| `--s1` | FreeU s1 skip feature scale (0.0-3.0) | `0.75` |
| `--s2` | FreeU s2 skip feature scale (0.0-3.0) | `0.65` |
| `--backup`/`--no-backup` | Also save a backup on the SDNext server | `--no-backup` |

### `reproduce` command

| Argument/Flag | Description | Default |
| --- | --- | --- |
| (argument) `HASH_OR_PATH` | Image hash (hex string) or file path to reproduce (required) | — |
| `--backup`/`--no-backup` | Also save a backup on the SDNext server | `--no-backup` |

Requires `--db` (the default); will fail with `--no-db`.

### `sync` command

| Argument/Flag | Description | Default |
| --- | --- | --- |
| `[ADD_DIR]` | Optional directory to add and sync | none (sync known dirs only) |
| `--force-api`/`--no-force-api` | Require SDNext API connection; if `--no-force-api` (default), will try to connect but proceed standalone if unavailable | `--no-force-api` |

### CLI Commands Documentation

This software auto-generates docs for CLI apps:

- [**`gen`** documentation](gen.md)

### Configuration

#### Config file locations

TransNext uses `transcrypto.utils.config` for configuration management. The database file (`config.bin`) is stored in OS-native locations:

- **macOS:** `~/Library/Application Support/transnext/config.bin`
- **Linux:** `~/.config/transnext/config.bin`
- **Windows:** `C:\Users\<user>\AppData\Local\transnext\config.bin`

The database is optionally encrypted with AES and stores:

- All known image metadata (hashes, paths, dimensions, generation parameters)
- Available AI model inventory (hashes, names, paths, types)
- Known image source directories for sync
- Current output directory setting

#### Environment variables

| Variable | Description | Default |
| --- | --- | --- |
| `SDAPI_URL` | Override SDNext API URL (e.g., `http://192.168.1.100:8000`); parsed into host + port | `http://127.0.0.1:7860` |
| `NO_COLOR` | Disable colored output (any value) | unset |

### Color and formatting

Rich provides colored terminal output. The app:

- Respects `NO_COLOR` environment variable
- Has `--no-color` / `--color` flag: if given, overrides the `NO_COLOR` environment variable
- If there is no environment variable and no flag is given, defaults to having color

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Generic failure (API error, DB error, invalid input) |
| 2 | CLI usage error (missing required arguments, invalid flag combinations) |

### Logging

TransNext uses Python's standard `logging` module with Rich formatting:

- Default: only *ERROR* messages
- `-v`: *WARNING* messages
- `-vv`: *INFO* messages (recommended for seeing generation progress)
- `-vvv`: *DEBUG* messages (full API payloads, DB operations)

## Project Design

### Architecture overview

```txt
CLI (Typer)  ->  gen.py (Main callback + config)
                   |-- cli/make.py       ->  core/db.py (AIDatabase.Txt2Img)    ->  core/sdnapi.py (API.Txt2Img)
                   |-- cli/reproduce.py  ->  core/db.py (AIDatabase.Reproduce)  ->  core/sdnapi.py (API.Txt2Img)
                   +-- cli/sync.py       ->  core/db.py (AIDatabase.Sync)       ->  filesystem scan + metadata parse
```

The CLI layer (`gen.py` + `cli/`) handles argument parsing and wiring. The core layer (`core/`) contains all business logic: the database (`db.py`), API client (`sdnapi.py`), and shared constants/types (`base.py`).

### Modules and packages

| Component | Responsibility |
| --- | --- |
| `gen.py` | Typer app definition, global callback, `GenConfig` dataclass, `markdown` command |
| `cli/make.py` | `make` command — image generation via SDNext API |
| `cli/reproduce.py` | `reproduce` command — re-generate an existing DB image by hash or path |
| `cli/sync.py` | `sync` command — directory scanning and DB import |
| `core/base.py` | Constants, enums (`Sampler`, `QueryParser`), CLI option definitions, `TransNextConfig` base class |
| `core/db.py` | `AIDatabase` class, TypedDict schemas (`DBImageType`, `AIMetaType`, `AIModelType`), image import/metadata parsing |
| `core/sdnapi.py` | SDNext API HTTP client (`API` class), image generation, model management |

### Data flow

#### Image generation (`make`)

1. Parse CLI args, create `GenConfig`
2. Open `sdnapi.API` connection to SDNext server
3. Open `AIDatabase` (load or create encrypted DB file)
4. Look up model by name in DB; if not found, fetch all models from API and store them
5. Check if an identical image already exists in DB (same `AIMetaType`); if so, return it
6. Call SDNext `/sdapi/v1/txt2img` with generation parameters
7. Validate response (dimensions, format), compute hashes
8. Save PNG to disk under `<output>/<YYYY-MM-DD>/<filename>.png`
9. Store `DBImageType` entry in DB, save DB

#### Image sync (`sync`)

1. Open `AIDatabase`
2. Optionally add new directory to known sources
3. Scan all known source directories recursively for image files (PNG, JPEG, GIF)
4. For each image: compute SHA-256 hash
   - If known: update path tracking (main path, alt paths for duplicates)
   - If new: open with PIL, extract SDNext/A1111 metadata from PNG info tags, parse metadata, create `DBImageType`, add to DB
5. Check for deleted paths (paths in DB that no longer exist on disk) and clean up
6. Save DB

#### Image reproduce (`reproduce`)

1. Parse CLI args, create `GenConfig`
2. Open `sdnapi.API` connection to SDNext server
3. Open `AIDatabase` (load encrypted DB file; requires `--db`)
4. If a file path was given, resolve it to an image hash via the DB path index
5. Look up the existing `DBImageType` entry by hash and retrieve its `AIMetaType`
6. Verify the model referenced in `AIMetaType` is still in the DB
7. Call SDNext `/sdapi/v1/txt2img` with the exact same `AIMetaType` parameters
8. Store the new `DBImageType` entry in DB, save DB

### Error handling

- All errors flow through a hierarchy: `base.Error` -> `db.Error` / `sdnapi.Error`
- CLI commands are wrapped with `@clibase.CLIErrorGuard` for clean error reporting
- `AIDatabase` is a context manager; on exception, the database is **not** saved to prevent corruption
- Safe save mode (default): reads the on-disk DB before writing to detect concurrent modifications

### Security model

- Database is optionally AES-encrypted via `transcrypto.core.aes`
- No secrets are logged (API responses are logged at DEBUG level but contain no credentials)
- SDNext API is called with `verify=False` because it typically runs on localhost without TLS certificates
- No telemetry, no external network calls except to the configured SDNext server

## Development Instructions

### File structure

```txt
.
├── CHANGELOG.md                  ⟸ latest changes/releases
├── LICENSE
├── Makefile
├── gen.md                        ⟸ auto-generated CLI doc (by `make docs` or `make ci`)
├── poetry.lock                   ⟸ maintained by Poetry, do not manually edit
├── pyproject.toml                ⟸ most important configurations live here
├── README.md                     ⟸ this documentation
├── SECURITY.md                   ⟸ security policy
├── requirements.txt
├── .editorconfig
├── .gitignore
├── .pre-commit-config.yaml       ⟸ pre-commit configs
├── .github/
│   ├── copilot-instructions.md   ⟸ GitHub Copilot project-specific instructions
│   ├── dependabot.yaml           ⟸ Github dependency update pipeline
│   └── workflows/
│       ├── ci.yaml               ⟸ Github CI pipeline
│       └── codeql.yaml           ⟸ Github security scans and code quality pipeline
├── .vscode/
│   ├── extensions.json
│   └── settings.json             ⟸ VSCode configs
├── scripts/
│   └── template.py               ⟸ Template for executable standalone scripts
├── src/
│   └── transnext/
│       ├── __init__.py           ⟸ Version (`__version__`)
│       ├── __main__.py
│       ├── gen.py                ⟸ Main CLI app entry point (GenConfig, Main callback)
│       ├── py.typed
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── make.py           ⟸ `gen make` command (image generation)
│       │   ├── reproduce.py      ⟸ `gen reproduce` command (re-generate image by hash/path)
│       │   └── sync.py           ⟸ `gen sync` command (directory sync)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── base.py           ⟸ Constants, enums, CLI option definitions
│       │   ├── db.py             ⟸ AIDatabase, TypedDict schemas, import/sync logic
│       │   └── sdnapi.py         ⟸ SDNext API client
│       └── utils/
│           ├── __init__.py
│           └── template.py       ⟸ Template for new modules
├── tests/                        ⟸ Unit tests (mirrors src/ structure)
│   ├── gen_test.py
│   ├── cli/
│   ├── core/
│   ├── data/
│   │   └── images/               ⟸ Real test images with embedded metadata
│   └── utils/
└── tests_integration/
    └── test_installed_cli.py     ⟸ Integration tests (wheel build + install + smoke)
```

### Development Setup

#### Install Python

On **Linux**:

```sh
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git python3 python3-dev python3-venv build-essential software-properties-common

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.13
```

On **macOS**:

```sh
brew update
brew upgrade
brew cleanup -s

brew install git python@3.13
```

#### Install Poetry (recommended: `pipx`)

[Poetry reference.](https://python-poetry.org/docs/cli/)

Install `pipx` (if you don't have it):

```sh
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

If you previously had **Poetry** installed, but ***not*** through `pipx` make sure to remove it first: `brew uninstall poetry` (mac) / `sudo apt-get remove python3-poetry` (linux). You should install Poetry with `pipx` and configure poetry to create `.venv/` locally. This keeps Poetry isolated from project virtual environments and python for the environments is isolated from python for Poetry. Do:

```sh
pipx install poetry
poetry --version
```

If you will use [PyPI](https://pypi.org/) to publish:

```sh
poetry config pypi-token.pypi <TOKEN>  # add your personal PyPI project token, if any
```

#### Make sure `.venv` is local

This project expects a project-local virtual environment at `./.venv` (VSCode settings assume it).

```sh
poetry config virtualenvs.in-project true
```

#### Get the repository

```sh
git clone https://github.com/balparda/transnext.git transnext
cd transnext
```

#### Create environment and install dependencies

From the repository root:

```sh
poetry env use python3.13  # creates the .venv with the correct Python version
poetry sync                # sync env to project's poetry.lock file
poetry env info            # no-op: just to check that environment looks good
poetry check               # no-op: make sure all pyproject.toml fields are being used correctly

poetry run gen --help      # simple test if everything loaded OK
make ci                    # should pass OK on clean repo
```

To activate and use the environment do:

```sh
poetry env activate        # (optional) will print activation command for environment, but you can just use:
source .venv/bin/activate  # because .venv SHOULD BE LOCAL
...
pytest -vvv  # for example, or other commands you want to execute in-environment
...
deactivate  # to close environment
```

#### Optional: VSCode setup

This repo ships a `.vscode/settings.json` configured to:

- use `./.venv/bin/python`
- run `pytest`
- use **Ruff** as formatter
- disable deprecated pylint/flake8 integrations
- configure Google-style docstrings via **autoDocstring**
- use **Code Spell Checker**

Recommended VSCode extensions:

- Python (`ms-python.python`)
- Python Environments (`ms-python.vscode-python-envs`)
- Python Debugger (`ms-python.debugpy`)
- Pylance (`ms-python.vscode-pylance`)
- Mypy Type Checker (`ms-python.mypy-type-checker`)
- Ruff (`charliermarsh.ruff`)
- autoDocstring — Python Docstring Generator (`njpwerner.autodocstring`)
- Code Spell Checker (`streetsidesoftware.code-spell-checker`)
- markdownlint (`davidanson.vscode-markdownlint`)
- Markdown All in One (`yzhang.markdown-all-in-one`) - helps maintain this `README.md` table of contents
- Markdown Preview Enhanced (`shd101wyy.markdown-preview-enhanced`, optional)
- GitHub Copilot (`github.copilot`) - AI assistant; reads `.github/copilot-instructions.md` for project-specific coding conventions (indentation, naming, workflow)

### Build and run

Build a wheel:

```sh
poetry build
```

Run from source:

```sh
poetry run gen --help
poetry run gen -vv --out ~/my-images make "a sunset" -i 30 --cfg 7.0
poetry run gen sync ~/existing/images
```

### Testing

#### Unit tests / Coverage

```sh
make test               # plain test run, no integration tests
make integration        # run the integration tests
poetry run pytest -vvv  # verbose test run, includes integration tests

make cov  # coverage run with typeguard, equivalent to:
          # poetry run pytest --typeguard-packages=transnext --cov=src --cov-report=term-missing -q tests
```

A test can be marked with a "tag" by just adding a decorator:

```py
@pytest.mark.slow
def test_foo_method() -> None:
  """Test."""
  ...
```

These tags, like `slow` above are defined in `pyproject.toml`, in section `[tool.pytest.ini_options.markers]`, and you can define your own there. The ones already defined are:

| Tag | Meaning |
| --- | --- |
| `slow` | test is slow (> 1s) |
| `flaky` | AVOID! - test is known to be flaky |
| `stochastic` | test is capable of failing (even if very unlikely) |

You can use them to filter tests, for example:

```sh
poetry run pytest -vvv -m slow  # run only the slow tests
```

You can find the slowest tests by running:

```sh
poetry run pytest -vvv -q --durations=20
poetry run pytest -vvv -q --durations=20 -m "not slow"  # find unknown slow methods
```

You can search for flaky tests by running `make flakes`, which runs all tests 100 times:

```sh
make flakes  # equivalent to: poetry run pytest --flake-finder --flake-runs=100 -q tests
```

#### Instrumenting your code

You can instrument your code to find bottlenecks:

```sh
$ source .venv/bin/activate
$ which gen
/path/to/.venv/bin/gen  # <== place this in the command below:
$ pyinstrument -r html -o output1.html -- /path/to/.venv/bin/gen -vv --out ~/foo make "test prompt"
$ deactivate
```

This will save a file `output1.html` to the project directory with the timings for all method calls. Make sure to **cleanup** these html files later.

#### Integration / e2e tests

Integration tests validate packaging and the installed console script by:

- building a wheel from the repository
- installing that wheel into a fresh temporary virtualenv
- running the installed `gen` console script to verify behavior (`--version` and basic commands)

The canonical integration test is [tests_integration/test_installed_cli.py](tests_integration/test_installed_cli.py). Tests in this suite are marked with `pytest.mark.integration`.

Run the integration tests with:

```sh
make integration  # or: poetry run pytest -m integration -q
```

### Linting / formatting / static analysis

```sh
make lint  # equivalent to: poetry run ruff check .
make fmt   # equivalent to: poetry run ruff format .
```

To check formatting without rewriting:

```sh
poetry run ruff format --check .
```

#### Type checking

```sh
make type  # equivalent to: poetry run mypy src tests tests_integration
```

(Pyright is primarily for editor-time; MyPy is what CI enforces.)

### Documentation updates

CLI reference documentation (`gen.md`) is auto-generated from the Typer app:

```sh
make docs  # or: poetry run gen markdown > gen.md
```

Always run `make docs` (or `make ci`) before committing to keep the CLI docs in sync.

### Versioning and releases

#### Versioning scheme

This project follows a pragmatic versioning approach:

- **Patch**: bug fixes / docs / small improvements.
- **Minor**: new features or non-breaking changes.
- **Major**: breaking changes (e.g., DB schema changes, CLI flag renames).

See: [CHANGELOG.md](CHANGELOG.md)

#### Updating versions

##### Bump project version (patch/minor/major)

Poetry can bump versions:

```sh
poetry version minor  # updates 1.0.0 to 1.1.0, for example
# or:
poetry version patch  # updates 1.0.0 to 1.0.1
# or:
poetry version <version-number>
```

This updates `[project].version` in `pyproject.toml`. **Remember to also update `src/transnext/__init__.py` to match (this repo gets/prints `__version__` from there)!**

##### Update dependency versions

The project has a [**dependabot**](https://docs.github.com/en/code-security/tutorials/secure-your-dependencies/dependabot-quickstart-guide) config file in `.github/dependabot.yaml` that weekly (defaulting to Tuesdays) scans both Github actions and the project dependencies and creates PRs to update them.

To update `poetry.lock` file to more current versions do `poetry update`, it will ignore the current lock, update, and rewrite the `poetry.lock` file. If you have cache problems `poetry cache clear PyPI --all` will clean it.

To add a new dependency you should do:

```sh
poetry add "pkg>=1.2.3"  # regenerates lock, updates env (adds dep to prod code)
poetry add -G dev "pkg>=1.2.3"  # adds dep to dev code ("group" dev)
# also remember: "pkg@^1.2.3" = latest 1.* ; "pkg@~1.2.3" = latest 1.2.* ; "pkg@1.2.3" exact
```

##### Exporting the `requirements.txt` file

Poetry uses `poetry.lock` as the primary lockfile. If you need a `requirements.txt` for Docker/legacy tooling:

```sh
make req  # or: poetry export --format requirements.txt --without-hashes --output requirements.txt
```

##### CI and docs

Make sure to run `make docs` or even better `make ci`. Both will update the CLI markdown docs and `requirements.txt` automatically.

##### Git tag and commit

Publish to GIT, including a TAG:

```sh
git commit -a -m "release version 1.1.0"
git tag 1.1.0
git push
git push --tags
```

##### Publish to PyPI

If you already have your PyPI token registered with Poetry (see [Install Poetry](#install-poetry-recommended-pipx)) then just:

```sh
poetry build
poetry publish
```

Remember to update [CHANGELOG.md](CHANGELOG.md).

## Security

Please refer to the security policy in [SECURITY.md](SECURITY.md) for supported versions and how to report vulnerabilities.

The project has a [**codeql**](https://codeql.github.com/docs/) config file in `.github/workflows/codeql.yaml` that weekly (defaulting to Fridays) scans the project for code quality and security issues. It will also run on all commits. Github security issues will be opened in the project if anything is found.

## Troubleshooting

### Enable debug output

```sh
poetry run gen -vvv --out ~/my-images make "test prompt"
```

The `-vvv` flag enables DEBUG-level logging, which will show full API payloads, database operations, and image processing details.

### Common issues

- **`Failed to connect to SDNext API`**: Make sure the SDNext server is running and accessible at the configured host:port. Check with `curl http://127.0.0.1:7860/sdapi/v1/sd-models`.
- **`Model with name "..." not found`**: The model name is matched as a substring against known model names. Run with `-vv` to see available models after they are fetched from the API. Use a more specific substring.
- **`With --no-db you must specify --out`**: When running without the database (`--no-db`), you must provide an output directory via `--out`.
- **`DB on disk ... differs from loaded DB ...`**: Two processes tried to write the DB simultaneously. The safe-save mechanism prevents data loss. Retry the operation.

## Glossary

- **A1111** — [Automatic1111 web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui), the original Stable Diffusion web UI; SDNext is a fork/successor
- **CFG** — Classifier-Free Guidance; the mechanism that steers generation toward the prompt
- **CLIP** — Contrastive Language-Image Pre-training; the text encoder used by Stable Diffusion
- **FreeU** — A technique for improving generation quality by scaling backbone and skip-connection features without additional training
- **LoRA** — Low-Rank Adaptation; lightweight fine-tuning layers that modify model behavior
- **LyCORIS** — A more flexible variant of LoRA with additional network module types
- **SDXL** — Stable Diffusion XL; a larger, higher-quality Stable Diffusion model architecture
- **SDNext** — [SD.Next](https://github.com/vladmandic/sdnext); the Stable Diffusion server this tool communicates with
- **txt2img** — Text-to-image generation; creating an image from a text prompt

---

*Thanks!* — Daniel Balparda & BellaKeri
