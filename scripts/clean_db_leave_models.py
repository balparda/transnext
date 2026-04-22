#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Cleans the database keeping only model and lora entries, removing all images and experiments.

This file is meant to be executed directly. It will prompt you to confirm before proceeding.

Usage
- ./scripts/clean_db_leave_models.py [args]
- Or: poetry run clean_db_leave_models [args]

"""

from __future__ import annotations

import pdb  # noqa: T100

from rich.console import Console
from transcrypto.utils import config as app_config
from transcrypto.utils import logging as cli_logging

from transnext.core import db


def Main() -> int:
  """Clean the DB: remove all images and experiments, keep model and lora inventory.

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
