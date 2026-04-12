# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: cli/make.py."""

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


def _CallCLI(args: list[str]) -> click_testing.Result:
  """Call the CLI with args.

  Args:
      args (list[str]): CLI arguments.

  Returns:
      click_testing.Result: CLI result.

  """
  return testing.CliRunner().invoke(gen.app, args)


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
@mock.patch('transnext.core.db.AIMetaTypeFactory')
def testMakeSuccess(
  mock_meta_factory: mock.Mock,
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
  tmp_path: mock.Mock,
) -> None:
  """Make command succeeds with DB enabled."""
  mock_meta = mock.MagicMock()
  mock_meta_factory.return_value = mock_meta
  mock_db_inst = mock.MagicMock()
  mock_db_inst.__enter__ = mock.Mock(return_value=mock_db_inst)
  mock_db_inst.__exit__ = mock.Mock(return_value=False)
  mock_db_inst.GetModelHash.return_value = 'model-hash'
  mock_db_inst.Txt2Img.return_value = (
    mock.MagicMock(),
    b'img-data',
  )
  mock_db_class.return_value = mock_db_inst
  mock_api_inst = mock.MagicMock()
  mock_api_class.return_value = mock_api_inst
  result = _CallCLI(
    [
      '-vvv',
      '--no-db',
      '--out',
      str(tmp_path),
      'make',
      'a cute cat',
    ]
  )
  assert result.exit_code == 0
  mock_api_inst.GetModels.assert_called_once()
  mock_db_inst.Txt2Img.assert_called_once()


def testMakeNoDbNoOutRaises() -> None:
  """Make command fails with --no-db and no --out."""
  result = _CallCLI(['-vvv', '--no-db', 'make', 'test prompt'])
  assert result.exit_code != 0
