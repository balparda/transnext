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

import re

from rich.console import Console
from transcrypto.utils import config as app_config
from transcrypto.utils import logging as cli_logging

from transnext.core import base, db, sdnapi

_MODEL_NOT_FOUND: re.Pattern[str] = re.compile(r'model\s#(?P<hash>.*)/(?P<name>.*)\snot found')
_LORA_NOT_FOUND: re.Pattern[str] = re.compile(r'lora\s#(?P<hash>.*)/(?P<name>.*)\snot found')


_PORT: int = 7860


def Main() -> int:  # noqa: C901, PLR0912
  """Show parse errors for all images in the DB.

  Returns:
    int: Exit code.

  """
  console: Console = cli_logging.InitLogging(2, color=True)[0]
  api = sdnapi.API(base.MakeURL(base.DEFAULT_HOST, _PORT), server_save_images=False)
  config: app_config.AppConfig = app_config.InitConfig('transnext', 'config.bin')
  with db.AIDatabase(config, api=api, read_only=True) as ai_db:  # noqa: PLR1702
    bad_models: dict[str, tuple[int, set[str]]] = {}
    bad_lora: dict[str, tuple[int, set[str]]] = {}
    hsh: str
    name: str
    for entry in ai_db._db['images'].values():  # pyright: ignore[reportPrivateUsage]
      for path, path_info in entry['paths'].items():
        filtered: list[str] = []
        for err in path_info['parse_errors'] or []:
          if err == 'upscaled' or 'size corrected' in err:
            continue
          if match := _MODEL_NOT_FOUND.match(err):
            hsh, name = match.groups()
            hsh, name = hsh.strip(), name.strip()
            hsh = hsh or name.lower()
            if hsh not in bad_models:
              bad_models[hsh] = (0, set())
            bad_models[hsh] = (bad_models[hsh][0] + 1, bad_models[hsh][1] | {name})
            continue
          if match := _LORA_NOT_FOUND.match(err):
            hsh, name = match.groups()
            hsh, name = hsh.strip(), name.strip()
            hsh = hsh or name.lower()
            if hsh not in bad_lora:
              bad_lora[hsh] = (0, set())
            bad_lora[hsh] = (bad_lora[hsh][0] + 1, bad_lora[hsh][1] | {name})
            continue
          filtered.append(err)
        if filtered:
          console.print(f'[red]{path}[/red]')
        print_info: bool = False
        for err in filtered:
          console.print(f'[yellow]{err}[/yellow]')
          if 'ambiguous' in err or 'lora/lyco possible' in err:
            print_info = True
        if print_info:
          console.print(f'{entry["info"]}')
        if filtered:
          console.print()
    console.print()
    console.print(
      f'[blue]Found {len(bad_models)} missing models and {len(bad_lora)} missing lora[/blue]'
    )
    console.print()
    for count, hsh, names in sorted(
      ((count, hsh, names) for hsh, (count, names) in bad_models.items()), reverse=True
    ):
      console.print(f'Missing model [green]{hsh!r}[/green] -> [yellow]{count}[/yellow] - {names}')
    console.print()
    for count, hsh, names in sorted(
      ((count, hsh, names) for hsh, (count, names) in bad_lora.items()), reverse=True
    ):
      console.print(f'Missing lora [cyan]{hsh!r}[/cyan] -> [yellow]{count}[/yellow] - {names}')
  return 0


if __name__ == '__main__':
  raise SystemExit(Main())
