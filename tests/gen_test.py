# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: mycli.py."""

from __future__ import annotations

from unittest import mock

import click
import pytest
import typer
from click import testing as click_testing
from transcrypto.utils import config as app_config
from transcrypto.utils import logging as cli_logging
from typer import testing

import transnext
from transnext import gen


@pytest.fixture(autouse=True)
def reset_cli() -> None:
  """Reset CLI singleton before each test."""
  cli_logging.ResetConsole()
  app_config.ResetConfig()


def CallCLI(args: list[str]) -> click_testing.Result:
  """Call the CLI with args.

  Args:
      args (list[str]): CLI arguments.

  Returns:
      click_testing.Result: CLI result.

  """
  return testing.CliRunner().invoke(gen.app, args)


def PrintedValue(console_mock: mock.Mock) -> object:
  """Return first argument passed to console.print(...).

  Args:
      console_mock (mock.Mock): console mock.

  Returns:
      object: first argument passed to console.print(...).

  """
  # console.print is a Mock; .call_args is (args, kwargs)
  args, _kwargs = console_mock.print.call_args
  return args[0] if args else None


def AssertRandomStrPrintedValue(printed: object, expected_prefix: str) -> None:
  """Assert RandomStr output matches CLI behavior.

  RandomStr prints the generated string plus a suffix that depends on whether color is enabled.
  We don't want tests to depend on NO_COLOR env var or rich console internals, so accept either.
  """
  assert isinstance(printed, str)
  assert printed.startswith(expected_prefix)
  suffix: str = printed[len(expected_prefix) :]
  assert suffix in {' - in color', ' - no colors'}


def test_version_flag() -> None:
  """Test."""
  result: click_testing.Result = CallCLI(['--version'])
  assert result.exit_code == 0
  assert result.stdout.strip() == transnext.__version__


def test_version_flag_raises_exit() -> None:
  """Test version flag raises typer.Exit with exit code 0."""
  ctx = mock.Mock(spec=click.Context)
  with pytest.raises(typer.Exit) as exc_info:
    gen.Main(ctx=ctx, version=True, verbose=0, color=None, foo=1000, bar='str default')
  assert exc_info.value.exit_code == 0


def test_run_function() -> None:
  """Test Run function calls app."""
  with mock.patch.object(gen, 'app') as app_mock:
    gen.Run()
    app_mock.assert_called_once()


def test_version_flag_ignores_extra_args() -> None:
  """Test."""
  result: click_testing.Result = CallCLI(['--version', 'markdown'])
  assert result.exit_code == 0
  assert result.stdout.strip() == transnext.__version__


def test_markdown_command_generates_docs() -> None:
  """Test markdown command generates documentation."""
  result: click_testing.Result = CallCLI(['markdown'])
  assert result.exit_code == 0, result.output
  # Verify it contains markdown-like content
  assert 'gen' in result.stdout
  assert '#' in result.stdout  # markdown headers
  assert '<!--' in result.stdout  # top comment
  # assert 'hello' in result.stdout and 'random' in result.stdout  # verify it includes subcommands
