# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""CLI generate images."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import click
import typer
from rich import console as rich_console
from transcrypto.cli import clibase
from transcrypto.utils import config as app_config
from transcrypto.utils import logging as cli_logging

from transnext.core import base

from . import __version__


@dataclass(kw_only=True, slots=True, frozen=True)
class GenConfig(base.TransNextConfig):
  """Gen CLI global context, storing the configuration."""


# CLI app setup, this is an important object and can be imported elsewhere and called
app = typer.Typer(
  add_completion=True,
  no_args_is_help=True,
  help=(  # keep in sync with Main().help
    'TransNext: SDXL helper, searcher, maker, based on SDNext API.'
  ),
  epilog=(
    'Example:\n\n\n\n'
    '# --- Generating Images ---\n\n'
    'poetry run gen -vv --out ~/foo/bar make "dark knight" -n batman '
    '--cfg 7.5 -m SDXL_model_1234 -i 30 --sampler "Euler a"\n\n\n\n'
    '# --- Reproducing an Image ---\n\n'
    'poetry run gen reproduce abc123def456\n\n'
    'poetry run gen reproduce ~/foo/bar/image.png\n\n\n\n'
    '# --- Syncing the DB ---\n\n'
    'poetry run gen sync\n\n'
    'poetry run gen sync ~/foo/bar/new/dir\n\n\n\n'
    '# --- Emitting CLI Markdown Docs ---\n\n'
    'poetry run gen markdown > gen.md'
  ),
)


def Run() -> None:
  """Run the CLI."""
  app()


@app.callback(
  invoke_without_command=True,
  help=(  # keep in sync with app.help
    'TransNext: SDXL helper, searcher, maker, based on SDNext API.'
  ),
)  # have only one; this is the "constructor"
@clibase.CLIErrorGuard
def Main(  # documentation is help/epilog/args # noqa: D103
  *,
  ctx: click.Context,  # global context
  version: bool = typer.Option(False, '--version', help='Show version and exit.'),
  verbose: int = typer.Option(
    0,
    '-v',
    '--verbose',
    count=True,
    help='Verbosity (nothing=ERROR, -v=WARNING, -vv=INFO, -vvv=DEBUG).',
    min=0,
    max=3,
  ),
  color: bool | None = typer.Option(
    None,
    '--color/--no-color',
    help=(
      'Force enable/disable colored output (respects NO_COLOR env var if not provided). '
      'Defaults to having colors.'  # state default because None default means docs don't show it
    ),
  ),
  host: str = base.SD_HOST_OPTION,  # type: ignore[assignment]
  port: int = base.SD_PORT_OPTION,  # type: ignore[assignment]
  db: bool = base.SD_DB_USE_OPTION,  # type: ignore[assignment]
  output: pathlib.Path | None = base.SD_IMAGES_OUTPUT_OPTION,  # type: ignore[assignment]
) -> None:
  if version:
    typer.echo(__version__)
    raise typer.Exit(0)
  console: rich_console.Console
  console, verbose, color = cli_logging.InitLogging(
    verbose,
    color=color,
    include_process=False,  # decide if you want process names in logs
    soft_wrap=False,  # decide if you want soft wrapping of long lines
  )
  # create context with the arguments we received
  ctx.obj = GenConfig(
    console=console,
    verbose=verbose,
    color=color,
    appconfig=app_config.InitConfig('transnext', 'config.bin'),
    host=host,
    port=port,
    db=db,
    output=output,
  )
  # even though this is a convenient place to print(), beware that this runs even when
  # a subcommand is invoked; so prefer logging.debug/info/warning/error instead of print();
  # for example, if you run `markdown` subcommand, this will still print and spoil the output


@app.command(
  'markdown',
  help='Emit Markdown docs for the CLI (see README.md section "Creating a New Version").',
  epilog=('Example:\n\n\n\n$ poetry run gen markdown > gen.md\n\n<<saves CLI doc>>'),
)
@clibase.CLIErrorGuard
def Markdown(*, ctx: click.Context) -> None:  # documentation is help/epilog/args # noqa: D103
  config: GenConfig = ctx.obj
  config.console.print(clibase.GenerateTyperHelpMarkdown(app, prog_name='gen'))


# Import CLI modules to register their commands with the app
from transnext.cli import (  # noqa: E402
  make,  # pyright: ignore[reportUnusedImport] # noqa: F401
  reproduce,  # pyright: ignore[reportUnusedImport] # noqa: F401
  sync,  # pyright: ignore[reportUnusedImport] # noqa: F401
)
