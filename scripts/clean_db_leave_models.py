#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Shows interesting errors found in the database, example missing models.

This file is meant to be executed directly.

Usage
- ./scripts/show_errors.py [args]
- Or: poetry run show_errors [args]

"""

from __future__ import annotations

import pdb  # noqa: T100
import re

from rich.console import Console
from transcrypto.utils import config as app_config
from transcrypto.utils import logging as cli_logging

from transnext.core import db

_MODEL_NOT_FOUND: re.Pattern[str] = re.compile(r'model\s#(?P<hash>.*)/(?P<name>.*)\snot found')
_LORA_NOT_FOUND: re.Pattern[str] = re.compile(r'lora\s#(?P<hash>.*)/(?P<name>.*)\snot found')


_PORT: int = 7860


def Main() -> int:
  """Show parse errors for all images in the DB.

  Returns:
    int: Exit code.

  """
  console: Console = cli_logging.InitLogging(2, color=True)[0]
  config: app_config.AppConfig = app_config.InitConfig('transnext', 'config.bin')
  console.print(
    '[bold red]WARNING: This script will modify the database by removing all entries '
    'with missing models, but keeping the models themselves. '
    'Make a backup before proceeding![/bold red]'
  )
  pdb.set_trace()  # noqa: T100
  with db.AIDatabase(config, read_only=False) as ai_db:
    ai_db._db = db._DBTypeFactory(  # pyright: ignore[reportPrivateUsage]
      {
        'version': ai_db._db['version'],  # pyright: ignore[reportPrivateUsage]
        'db_version': ai_db._db['db_version'],  # pyright: ignore[reportPrivateUsage]
        'last_save': ai_db._db['last_save'],  # pyright: ignore[reportPrivateUsage]
        'models': ai_db._db['models'],  # pyright: ignore[reportPrivateUsage]
        'lora': ai_db._db['lora'],  # pyright: ignore[reportPrivateUsage]
      }
    )
  return 0


if __name__ == '__main__':
  raise SystemExit(Main())
