# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: cli/sync.py."""

from __future__ import annotations

import pathlib
from unittest import mock

import pytest
from click import testing as click_testing
from transcrypto.utils import config as app_config
from transcrypto.utils import logging as cli_logging
from typer import testing

from transnext import gen
from transnext.core import sdnapi


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
  return mock_db_inst


# NOTE: We mock ServerVersion instead of the entire API class because
# typeguard instruments `api: sdnapi.API | None` and requires sdnapi.API
# to remain a valid type at runtime.


@mock.patch.object(sdnapi.API, 'ServerVersion', return_value=('abc123', '2026-01-01'))
@mock.patch('transnext.core.db.AIDatabase')
def testSyncSuccess(
  mock_db_class: mock.Mock,
  _mock_sv: mock.Mock,
) -> None:
  """Sync command succeeds with API available and no add_dir."""
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_class.return_value = mock_db_inst
  result: click_testing.Result = _CallCLI(['-vvv', 'sync'])
  assert not result.exit_code
  mock_db_inst.Sync.assert_called_once_with(add_dir=None)
  _, db_kwargs = mock_db_class.call_args
  assert isinstance(db_kwargs['api'], sdnapi.API)


@mock.patch.object(sdnapi.API, 'ServerVersion', return_value=('abc123', '2026-01-01'))
@mock.patch('transnext.core.db.AIDatabase')
def testSyncWithAddDir(
  mock_db_class: mock.Mock,
  _mock_sv: mock.Mock,
  tmp_path: mock.Mock,
) -> None:
  """Sync command passes add_dir argument through to DB.Sync."""
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_class.return_value = mock_db_inst
  result: click_testing.Result = _CallCLI(['-vvv', 'sync', str(tmp_path)])
  assert not result.exit_code
  call_kwargs: dict[str, object] = mock_db_inst.Sync.call_args[1]
  assert call_kwargs['add_dir'] == pathlib.Path(str(tmp_path))


@mock.patch.object(
  sdnapi.API,
  'ServerVersion',
  side_effect=sdnapi.APIConnectionError('refused'),
)
@mock.patch('transnext.core.db.AIDatabase')
def testSyncAPIConnectionFailsGracefully(
  mock_db_class: mock.Mock,
  _mock_sv: mock.Mock,
) -> None:
  """Sync proceeds without API when connection fails and --no-force-api."""
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_class.return_value = mock_db_inst
  result: click_testing.Result = _CallCLI(['-vvv', 'sync', '--no-force-api'])
  assert not result.exit_code
  mock_db_inst.Sync.assert_called_once_with(add_dir=None)
  _, db_kwargs = mock_db_class.call_args
  assert db_kwargs['api'] is None


@mock.patch.object(
  sdnapi.API,
  'ServerVersion',
  side_effect=sdnapi.APIConnectionError('refused'),
)
@mock.patch('transnext.core.db.AIDatabase')
def testSyncForceAPIFailsRaises(
  mock_db_class: mock.Mock,
  _mock_sv: mock.Mock,
) -> None:
  """Sync with --force-api raises error when API connection fails."""
  mock_db_class.return_value = _MockDB()
  result: click_testing.Result = _CallCLI(['-vvv', 'sync', '--force-api'])
  assert result.exit_code != 0


@mock.patch.object(sdnapi.API, 'ServerVersion', return_value=('abc123', '2026-01-01'))
@mock.patch('transnext.core.db.AIDatabase')
def testSyncWithNoDb(
  mock_db_class: mock.Mock,
  _mock_sv: mock.Mock,
) -> None:
  """Sync with --no-db passes read_only=True to AIDatabase."""
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_class.return_value = mock_db_inst
  result: click_testing.Result = _CallCLI(['-vvv', '--no-db', 'sync'])
  assert not result.exit_code
  _, db_kwargs = mock_db_class.call_args
  assert db_kwargs['read_only'] is True
