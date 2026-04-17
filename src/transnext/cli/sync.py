# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""CLI: Make commands."""

from __future__ import annotations

import logging
import pathlib

import click
import typer
from transcrypto.cli import clibase

from transnext import gen
from transnext.core import base, db, sdnapi


@gen.app.command(
  'sync',
  help='Go over all known image dirs, check for new/deleted images, update DB accordingly.',
  epilog=('Example:\n\n\n\npoetry run gen sync\n\npoetry run gen sync ~/foo/bar/new/dir'),
)
@clibase.CLIErrorGuard
def Sync(  # documentation is help/epilog/args # noqa: D103
  *,
  ctx: click.Context,
  add_dir: pathlib.Path | None = typer.Argument(  # noqa: B008
    None,
    exists=True,
    file_okay=False,
    dir_okay=True,
    readable=True,
    help=(
      'Optional directory to add to the sync process; default: no new dir, just sync known ones.'
    ),
  ),
  force_api: bool = base.SD_FORCE_API,  # type: ignore[assignment]
) -> None:
  # open DB
  config: gen.GenConfig = ctx.obj
  api: sdnapi.API | None = None
  try:
    api = sdnapi.API(base.MakeURL(config.host, config.port), server_save_images=False)
  except sdnapi.APIConnectionError as err:
    if force_api:
      raise click.UsageError(
        f'Failed to connect to SDNext API server (to run with no API use `--no-force-api`): {err}'
      ) from err
    logging.warning(f'Failed to connect to SDNext API server, proceeding without API: {err}')
  with db.AIDatabase(config.appconfig, read_only=not config.db, api=api) as ai_db:
    ai_db.Sync(add_dir=add_dir)
  # DB is closed and saved
  config.console.print()
