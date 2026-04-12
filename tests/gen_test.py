# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: gen.py CLI entry point."""

from __future__ import annotations

from unittest import mock

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


def CallCLI(args: list[str]) -> click_testing.Result:
  """Call the CLI with args.

  Args:
      args (list[str]): CLI arguments.

  Returns:
      click_testing.Result: CLI result.

  """
  return testing.CliRunner().invoke(gen.app, args)


def testVersion() -> None:
  """--version prints version and exits."""
  result = CallCLI(['--version'])
  assert result.exit_code == 0
  assert '1.0.0' in result.output


def testNoArgsShowsHelp() -> None:
  """No arguments shows help text."""
  result = CallCLI([])
  assert result.exit_code in {0, 2}  # typer no_args_is_help exits with 2


def testMarkdown() -> None:
  """Markdown command generates CLI documentation."""
  result = CallCLI(['-vvv', 'markdown'])
  assert result.exit_code == 0
  assert 'gen' in result.output


def testRunCallsApp() -> None:
  """Run() calls app()."""
  with mock.patch.object(gen, 'app') as mock_app:
    gen.Run()
  mock_app.assert_called_once()
