# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the whole stack minus the SDNext part.

With an EMPTY database and transnext.cli.make._DEBUG_RECORD==True i ran:

```bash
poetry run gen --port 7861 make \
    "dark knight in moody rain <lora:XL-CLR-colorful-fractal:1.33>" -n "batman, comic, text" \
    -i 40 -s 666 --vseed 999 --vstrength 0.3 -w 512 -h 256 --sampler "DPM SDE" --parser a1111 \
    -m XLB --clip 2 --cfg 8 --cfg-end 0.9 --cfg-rescale 0.8 \
    --freeu --b1 1.1 --b2 1.15 --s1 0.7 --s2 0.6 \
    --sigma karras --spacing linspace --beta linear --prediction epsilon
```

The resulting JSON call log was moved to `tests/data/json/call_record_1.json`.
The resulting image is in
`tests/data/images/6db2ba7302bd-20260417102202-e6bb9ea8-80-40-512-256-666-db088cdca097.png`.

In this test we mock the *minimal*:

- transnext.core.sdnapi._Call()  (replays recorded HTTP responses from the JSON)
- transcrypto.utils.timer.Now()  (fixed timestamp for deterministic file naming)
- pathlib.Path.exists() and pathlib.Path.read_bytes()  (fake model/lora files on disk)

And run the full CLI command through the real stack.
"""

from __future__ import annotations

import base64
import json
import pathlib
from collections import abc
from typing import cast
from unittest import mock

import pytest
from click import testing as click_testing
from transcrypto.core import hashes
from transcrypto.utils import base as tbase
from transcrypto.utils import config as app_config
from transcrypto.utils import logging as cli_logging
from typer import testing

from transnext import gen
from transnext.core import base

# ─── test data paths ─────────────────────────────────────────────────────────

_DATA_DIR: pathlib.Path = pathlib.Path(__file__).parent / 'data'
_CALL_RECORD_PATH: pathlib.Path = _DATA_DIR / 'json' / 'call_record_1.json'

# ─── fixed time for deterministic output ──────────────────────────────────────

_FIXED_TIMESTAMP: int = 1776000000  # 2026-04-12 13:20:00 UTC

# ─── fake file paths from the recorded API responses ─────────────────────────

_MODEL_PATH: str = '/foo/bar/sdnext/models/Stable-diffusion/SDXL_00_XLB_v10VAEFix.safetensors'
_LORA_PATH: str = '/foo/bar/sdnext/models/Lora/XL-CLR-colorful-fractal.safetensors'
_FAKE_FILE_BYTES: bytes = b'fake-model-file-bytes-for-hashing'

# ─── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_cli() -> None:
  """Reset CLI singleton before each test."""
  cli_logging.ResetConsole()
  app_config.ResetConfig()


def _CallCLI(args: list[str]) -> click_testing.Result:
  """Call the CLI with args.

  Args:
    args (list[str]): CLI arguments.

  Returns:
    click_testing.Result: CLI result.

  """
  return testing.CliRunner().invoke(gen.app, args)


def _LoadCallRecord() -> list[tbase.JSONDict]:
  """Load the recorded API call/response pairs from the JSON fixture.

  Returns:
    list[tbase.JSONDict]: The list of recorded API calls.

  """
  with _CALL_RECORD_PATH.open('r', encoding='utf-8') as f:
    return cast('list[tbase.JSONDict]', json.load(f))


def _MakeCallReplay(
  records: list[tbase.JSONDict],
) -> abc.Callable[..., tbase.JSONValue]:
  """Build a side_effect function that replays recorded _Call() responses in order.

  Args:
    records (list[tbase.JSONDict]): Recorded API call/response pairs.

  Returns:
    abc.Callable[..., tbase.JSONValue]: A callable returning the next recorded response.

  """
  call_index: list[int] = [0]  # mutable counter for closure

  def _replay(
    _method: object,
    _sd_url: str,
    _endpoint: str,
    _payload: tbase.JSONDict | None = None,
    *,
    record_list: list[tbase.JSONDict] | None = None,  # noqa: ARG001
  ) -> tbase.JSONValue:
    """Return the next recorded response, advancing the counter.

    Args:
      _method: ignored (the HTTP method function).
      _sd_url: ignored (the base URL).
      _endpoint: ignored (the API endpoint).
      _payload: ignored (the JSON payload).
      record_list: ignored (recording list).

    Returns:
      tbase.JSONValue: The recorded response.

    Raises:
      IndexError: If all recorded responses have been consumed.

    """
    idx: int = call_index[0]
    if idx >= len(records):
      raise IndexError(
        f'Semi-integration replay exhausted: '
        f'expected at most {len(records)} calls, got call #{idx + 1}'
      )
    response: tbase.JSONValue = (records[idx]['response'],)
    call_index[0] += 1
    return response

  return _replay


# ─── path patching helpers ────────────────────────────────────────────────────

_REAL_EXISTS: abc.Callable[[pathlib.Path], bool] = pathlib.Path.exists
_REAL_READ_BYTES: abc.Callable[[pathlib.Path], bytes] = pathlib.Path.read_bytes
_REAL_IS_DIR: abc.Callable[[pathlib.Path], bool] = pathlib.Path.is_dir

_FAKE_PATHS: frozenset[str] = frozenset({_MODEL_PATH, _LORA_PATH})


def _patched_exists(self: pathlib.Path) -> bool:
  """Return True for fake model/lora file paths; delegate otherwise.

  Args:
    self: The Path instance.

  Returns:
    bool: Whether the path exists.

  """
  if str(self) in _FAKE_PATHS:
    return True
  return _REAL_EXISTS(self)


def _patched_read_bytes(self: pathlib.Path) -> bytes:
  """Return fake bytes for model/lora paths; delegate otherwise.

  Args:
    self: The Path instance.

  Returns:
    bytes: The file contents.

  """
  if str(self) in _FAKE_PATHS:
    return _FAKE_FILE_BYTES
  return _REAL_READ_BYTES(self)


def _patched_is_dir(self: pathlib.Path) -> bool:
  """Return False for fake model/lora file paths; delegate otherwise.

  Args:
    self: The Path instance.

  Returns:
    bool: Whether the path is a directory.

  """
  if str(self) in _FAKE_PATHS:
    return False
  return _REAL_IS_DIR(self)


# ─── CLI args for the recorded command ────────────────────────────────────────

_MAKE_CLI_ARGS: list[str] = [
  '-vvv',
  '--port',
  '7861',
  '--no-db',
  'make',
  'dark knight in moody rain <lora:XL-CLR-colorful-fractal:1.33>',
  '-n',
  'batman, comic, text',
  '-i',
  '40',
  '-s',
  '666',
  '--vseed',
  '999',
  '--vstrength',
  '0.3',
  '-w',
  '512',
  '-h',
  '256',
  '--sampler',
  'DPM SDE',
  '--parser',
  'a1111',
  '-m',
  'XLB',
  '--clip',
  '2',
  '--cfg',
  '8',
  '--cfg-end',
  '0.9',
  '--cfg-rescale',
  '0.8',
  '--freeu',
  '--b1',
  '1.1',
  '--b2',
  '1.15',
  '--s1',
  '0.7',
  '--s2',
  '0.6',
  '--sigma',
  'karras',
  '--spacing',
  'linspace',
  '--beta',
  'linear',
  '--prediction',
  'epsilon',
]

# ─── test ─────────────────────────────────────────────────────────────────────


@mock.patch('transnext.core.sdnapi._Call')
@mock.patch('transcrypto.utils.timer.Now', return_value=_FIXED_TIMESTAMP)
@mock.patch('transcrypto.utils.config.InitConfig')
def testMakeSemiIntegration(
  mock_init_config: mock.Mock,
  _mock_timer_now: mock.Mock,
  mock_call: mock.Mock,
  tmp_path: pathlib.Path,
) -> None:
  """Full CLI make command with recorded API responses replayed through _Call().

  This exercises the real stack: CLI parsing -> make command -> sdnapi.API ->
  db.AIDatabase -> image extraction and saving — with only the HTTP layer mocked.
  """
  # ── fresh DB config pointing to tmp_path ─────────────────────────────
  mock_init_config.return_value = app_config.AppConfig(
    'transnext',
    'config.bin',
    fixed_dir=tmp_path / 'db',
  )
  # ── load recorded data and wire up the replay ────────────────────────
  records: list[tbase.JSONDict] = _LoadCallRecord()
  mock_call.side_effect = _MakeCallReplay(records)

  # ── extract the expected image from the recording ────────────────────
  b64_image: str = records[-1]['response']['images'][0]  # type: ignore[call-overload, index, assignment]
  expected_img_bytes: bytes = base64.b64decode(b64_image)
  expected_img_hash: str = hashes.Hash256(expected_img_bytes).hex()

  # ── output directory ─────────────────────────────────────────────────
  out_dir: pathlib.Path = tmp_path / 'output'
  out_dir.mkdir()

  # ── run the CLI command with all the same args as the original ───────
  with (
    mock.patch.object(pathlib.Path, 'exists', _patched_exists),
    mock.patch.object(pathlib.Path, 'read_bytes', _patched_read_bytes),
    mock.patch.object(pathlib.Path, 'is_dir', _patched_is_dir),
  ):
    result: click_testing.Result = _CallCLI(
      ['--out', str(out_dir), *_MAKE_CLI_ARGS],
    )

  # ── CLI should succeed ──────────────────────────────────────────────
  assert not result.exit_code, f'CLI exited with code {result.exit_code}:\n{result.output}'

  # ── all 6 recorded _Call() invocations should have been consumed ────
  assert mock_call.call_count == 6

  # ── verify the generated image was saved to disk ────────────────────
  date_subdirs: list[pathlib.Path] = [p for p in out_dir.iterdir() if p.is_dir()]
  assert len(date_subdirs) == 1, f'Expected 1 date subdir, got {date_subdirs}'
  png_files: list[pathlib.Path] = list(date_subdirs[0].glob('*.png'))
  assert len(png_files) == 1, f'Expected 1 PNG file, got {png_files}'
  saved_file: pathlib.Path = png_files[0]

  # ── verify the saved file matches the expected image ────────────────
  saved_bytes: bytes = saved_file.read_bytes()
  assert hashes.Hash256(saved_bytes).hex() == expected_img_hash
  assert saved_bytes == expected_img_bytes

  # ── verify filename encodes the generation parameters ───────────────
  filename: str = saved_file.name
  prompt_hash: str = base.PromptHash(
    'dark knight in moody rain <lora:XL-CLR-colorful-fractal:1.33>',
    'batman, comic, text',
  )
  assert filename.startswith(prompt_hash)
  assert '-e6bb9ea8-' in filename  # model_hash[:8]
  assert '-80-40-512-256-666-' in filename  # cfg*10, steps, w, h, seed
  assert filename.endswith(f'{expected_img_hash[:12]}.png')
