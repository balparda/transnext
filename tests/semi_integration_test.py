# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the whole stack minus the SDNext part.

With an EMPTY database and transnext.cli.make._DEBUG_RECORD==True i ran:

```
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

- transnext.core.sdnapi._Call()
- time
- disk

And run the CLI command

"""

from __future__ import annotations

import pytest
from click import testing as click_testing
from transcrypto.utils import config as app_config
from transcrypto.utils import logging as cli_logging
from typer import testing

from transnext import gen


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


def testMakeNoBackupDefault() -> None:
  # TODO: mock only transnext.core.sdnapi._Call(), time, disk
  result: click_testing.Result = _CallCLI(
    ['--port', '7861', 'make', 'etc etc etc'],  # TODO: place args here
  )
  assert not result.exit_code
  _, kwargs = mock_api_class.call_args
  # TODO: make sure we got exactly the expected result
