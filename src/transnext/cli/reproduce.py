# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""CLI: Reproduce command.

Accepts either an image hash (hex string) or a file path, looks up the corresponding DB
entry, and calls Reproduce() to regenerate the image via the SDNext API.
"""

from __future__ import annotations

import logging

import click
import typer
from transcrypto.cli import clibase

from transnext import gen
from transnext.core import base, db, sdnapi


@gen.app.command(
  'reproduce',
  help='Reproduce an existing DB image by hash or file path.',
  epilog=(
    'Example:\n\n\n\n'
    'poetry run gen reproduce abc123def456\n\n'
    'poetry run gen reproduce ~/foo/bar/image.png'
  ),
)
@clibase.CLIErrorGuard
def Reproduce(  # documentation is help/epilog/args # noqa: D103
  *,
  ctx: click.Context,
  hash_or_path: str = typer.Argument(
    ...,
    help=(
      'Image hash (hex string) or file path to reproduce. '
      'If a path is given it will be resolved to a hash via the DB index.'
    ),
  ),
  backup: bool = base.SD_API_SERVER_SAVE,  # type: ignore[assignment]
) -> None:
  config: gen.GenConfig = ctx.obj
  if not config.db:
    raise click.UsageError('Cannot reproduce without DB access (do not use `--no-db`)')
  # open API and DB
  api = sdnapi.API(base.MakeURL(config.host, config.port), server_save_images=backup)
  with db.AIDatabase(config.appconfig, api=api) as ai_db:
    # set output, if specified
    if config.output is not None:
      ai_db.output = config.output
    # resolve path -> hash if needed
    img_hash: str = hash_or_path
    resolved: str | None = ai_db.Path(hash_or_path)
    if resolved is not None:
      logging.info(f'Resolved path {hash_or_path!r} -> hash {resolved!r}')
      img_hash = resolved
    else:
      logging.info(f'Treating {hash_or_path!r} as image hash (not found as a path in DB)')
    # reproduce
    new_entry, _img_bytes = ai_db.Reproduce(img_hash, api)
    logging.info(f'Reproduced: hash={new_entry["hash"]!r}, path={next(iter(new_entry["paths"]))!r}')
  # DB is closed and saved
  config.console.print()
