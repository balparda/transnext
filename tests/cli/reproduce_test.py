# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: cli/reproduce.py."""

from __future__ import annotations

from unittest import mock

import pytest
from click import testing as click_testing
from transcrypto.utils import config as app_config
from transcrypto.utils import logging as cli_logging
from typer import testing

from transnext import gen
from transnext.core import db


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


def _MockDB() -> mock.MagicMock:
  """Create a mock AIDatabase instance configured for context manager use.

  Returns:
    mock.MagicMock: Configured mock DB.

  """
  mock_db_inst = mock.MagicMock()
  mock_db_inst.__enter__ = mock.Mock(return_value=mock_db_inst)
  mock_db_inst.__exit__ = mock.Mock(return_value=False)
  # Path() returns None by default (treat arg as a hash, not a path)
  mock_db_inst.Path.return_value = None
  # Reproduce() returns a tuple of (entry_dict, image_bytes)
  mock_db_inst.Reproduce.return_value = (
    {'hash': 'new-hash-123', 'paths': {'/output/2026-04-17/new-hash-123.png': {}}},
    b'image-bytes',
  )
  return mock_db_inst


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
def testReproduceByHash(
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
) -> None:
  """Reproduce command succeeds when given an image hash directly."""
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_class.return_value = mock_db_inst
  mock_api_class.return_value = mock.MagicMock()
  result: click_testing.Result = _CallCLI(['-vvv', 'reproduce', 'abc123def456'])
  assert not result.exit_code, result.output
  # Path() was called to check if it was a path in DB; it returned None, so hash is used
  mock_db_inst.Path.assert_called_once_with('abc123def456')
  # Reproduce() must have been called with the hash
  mock_db_inst.Reproduce.assert_called_once()
  call_args = mock_db_inst.Reproduce.call_args
  assert call_args[0][0] == 'abc123def456'


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
def testReproduceByPath(
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
) -> None:
  """Reproduce command resolves a file path to a hash via DB index."""
  mock_db_inst: mock.MagicMock = _MockDB()
  # Path() finds the entry and returns a hash
  mock_db_inst.Path.return_value = 'resolved-hash-999'
  mock_db_class.return_value = mock_db_inst
  mock_api_class.return_value = mock.MagicMock()
  result: click_testing.Result = _CallCLI(['-vvv', 'reproduce', '/some/path/image.png'])
  assert not result.exit_code, result.output
  mock_db_inst.Path.assert_called_once_with('/some/path/image.png')
  # Reproduce() is called with the resolved hash, not the path
  call_args = mock_db_inst.Reproduce.call_args
  assert call_args[0][0] == 'resolved-hash-999'


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
def testReproduceWithOutputDir(
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
  tmp_path: mock.Mock,
) -> None:
  """Reproduce command sets ai_db.output when --out is provided."""
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_class.return_value = mock_db_inst
  mock_api_class.return_value = mock.MagicMock()
  result: click_testing.Result = _CallCLI(
    ['-vvv', '--out', str(tmp_path), 'reproduce', 'abc123def456']
  )
  assert not result.exit_code, result.output
  # output property was set on the DB instance
  assert mock_db_inst.output == tmp_path
  mock_db_inst.Reproduce.assert_called_once()


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
def testReproduceWithBackup(
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
) -> None:
  """Reproduce command passes --backup flag to the API constructor."""
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_class.return_value = mock_db_inst
  mock_api_class.return_value = mock.MagicMock()
  result: click_testing.Result = _CallCLI(['-vvv', 'reproduce', 'abc123', '--backup'])
  assert not result.exit_code, result.output
  # API was constructed with server_save_images=True
  _, api_kwargs = mock_api_class.call_args
  assert api_kwargs.get('server_save_images') is True


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
def testReproduceNoBackup(
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
) -> None:
  """Reproduce command passes --no-backup flag (default) to the API constructor."""
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_class.return_value = mock_db_inst
  mock_api_class.return_value = mock.MagicMock()
  result: click_testing.Result = _CallCLI(['-vvv', 'reproduce', 'abc123', '--no-backup'])
  assert not result.exit_code, result.output
  _, api_kwargs = mock_api_class.call_args
  assert api_kwargs.get('server_save_images') is False


def testReproduceWithNoDbRaisesUsageError() -> None:
  """Reproduce command raises UsageError when --no-db is set."""
  result: click_testing.Result = _CallCLI(['-vvv', '--no-db', 'reproduce', 'abc123'])
  assert result.exit_code != 0


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
def testReproduceDBReproduceError(
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
) -> None:
  """CLIErrorGuard swallows db.Error from AIDatabase.Reproduce() and exits cleanly.

  CLIErrorGuard catches base.Error (parent of db.Error) and just prints it; it does NOT
  raise SystemExit(1), so exit_code is 0. This is the expected graceful error handling.
  """
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_inst.Reproduce.side_effect = db.Error('hash not found in DB')
  mock_db_class.return_value = mock_db_inst
  mock_api_class.return_value = mock.MagicMock()
  result: click_testing.Result = _CallCLI(['-vvv', 'reproduce', 'bad-hash'])
  # CLIErrorGuard catches db.Error and exits cleanly (exit_code 0)
  assert result.exit_code == 0
  # Reproduce() was called with the hash
  mock_db_inst.Reproduce.assert_called_once()
  call_args = mock_db_inst.Reproduce.call_args
  assert call_args[0][0] == 'bad-hash'
