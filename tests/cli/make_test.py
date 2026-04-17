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
from transnext.cli import make
from transnext.core import base


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
  mock_db_inst.GetModelHash.return_value = 'model-hash'
  mock_db_inst.Txt2Img.return_value = (mock.MagicMock(), b'img-data')
  return mock_db_inst


def testNotForgotRecordingOn() -> None:
  """Make sure we don't submit  code with _DEBUG_RECORD accidentally left on."""
  assert not make._DEBUG_RECORD


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
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_class.return_value = mock_db_inst
  mock_api_inst = mock.MagicMock()
  mock_api_class.return_value = mock_api_inst
  result: click_testing.Result = _CallCLI(
    ['-vvv', '--out', str(tmp_path), 'make', 'a cute cat'],
  )
  assert not result.exit_code
  mock_db_inst.GetModelHash.assert_called_once_with('XLB_v10', api=mock_api_inst)
  mock_db_inst.Txt2Img.assert_called_once_with(mock_meta, mock_api_inst)


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
@mock.patch('transnext.core.db.AIMetaTypeFactory')
def testMakeWithAllOptions(
  mock_meta_factory: mock.Mock,
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
  tmp_path: mock.Mock,
) -> None:
  """Make command succeeds with all options specified."""
  mock_meta = mock.MagicMock()
  mock_meta_factory.return_value = mock_meta
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_class.return_value = mock_db_inst
  mock_api_class.return_value = mock.MagicMock()
  result: click_testing.Result = _CallCLI(
    [
      '-vvv',
      '--out',
      str(tmp_path),
      'make',
      'a cute cat',
      '-n',
      'ugly',
      '-i',
      '30',
      '-s',
      '123',
      '--vseed',
      '456',
      '--vstrength',
      '0.5',
      '-w',
      '512',
      '--height',
      '768',
      '--clip',
      '2',
      '--cfg',
      '7.5',
      '--cfg-end',
      '0.5',
      '--cfg-rescale',
      '0.3',
      '-m',
      'my_model',
      '--sampler',
      'Euler',
      '--parser',
      'native',
      '--freeu',
      '--b1',
      '1.3',
      '--b2',
      '1.4',
      '--s1',
      '0.9',
      '--s2',
      '0.2',
      '--backup',
    ]
  )
  assert not result.exit_code
  mock_db_inst.GetModelHash.assert_called_once()
  call_args: dict[str, object] = mock_meta_factory.call_args[0][0]
  assert call_args == {
    'cfg_end': 5,
    'cfg_rescale': 30,
    'cfg_scale': 75,
    'clip_skip': 20,
    'freeu': (130, 140, 90, 20),
    'height': 768,
    'model_hash': 'model-hash',
    'negative': 'ugly',
    'parser': 'native',
    'positive': 'a cute cat',
    'sampler': 'Euler',
    'sch_beta': None,
    'sch_sigma': None,
    'sch_spacing': None,
    'sch_type': None,
    'seed': 123,
    'steps': 30,
    'v_seed': (
      456,
      50,
    ),
    'width': 512,
  }


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
@mock.patch('transnext.core.db.AIMetaTypeFactory')
def testMakeBackupFlagPassedToAPI(
  mock_meta_factory: mock.Mock,
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
  tmp_path: mock.Mock,
) -> None:
  """Make --backup passes server_save_images=True to API constructor."""
  mock_meta_factory.return_value = mock.MagicMock()
  mock_db_class.return_value = _MockDB()
  mock_api_class.return_value = mock.MagicMock()
  result: click_testing.Result = _CallCLI(
    ['-vvv', '--out', str(tmp_path), 'make', 'prompt', '--backup'],
  )
  assert not result.exit_code
  mock_api_class.assert_called_once()
  _, kwargs = mock_api_class.call_args
  assert kwargs['server_save_images'] is True


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
@mock.patch('transnext.core.db.AIMetaTypeFactory')
def testMakeNoBackupDefault(
  mock_meta_factory: mock.Mock,
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
  tmp_path: mock.Mock,
) -> None:
  """Make without --backup passes server_save_images=False to API."""
  mock_meta_factory.return_value = mock.MagicMock()
  mock_db_class.return_value = _MockDB()
  mock_api_class.return_value = mock.MagicMock()
  result: click_testing.Result = _CallCLI(
    ['-vvv', '--out', str(tmp_path), 'make', 'prompt'],
  )
  assert not result.exit_code
  _, kwargs = mock_api_class.call_args
  assert kwargs['server_save_images'] is False


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
@mock.patch('transnext.core.db.AIMetaTypeFactory')
def testMakeWithOutputSetsDB(
  mock_meta_factory: mock.Mock,
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
  tmp_path: mock.Mock,
) -> None:
  """Make with --out sets the output directory on the DB."""
  mock_meta_factory.return_value = mock.MagicMock()
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_class.return_value = mock_db_inst
  mock_api_class.return_value = mock.MagicMock()
  result: click_testing.Result = _CallCLI(
    ['-vvv', '--out', str(tmp_path), 'make', 'prompt'],
  )
  assert not result.exit_code
  # output setter was called on the mock
  assert mock_db_inst.output == tmp_path


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
@mock.patch('transnext.core.db.AIMetaTypeFactory')
def testMakeVseedWithoutStrengthIgnored(
  mock_meta_factory: mock.Mock,
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
  tmp_path: mock.Mock,
) -> None:
  """Make with --vseed but default vstrength=0.0 produces v_seed=None."""
  mock_meta_factory.return_value = mock.MagicMock()
  mock_db_class.return_value = _MockDB()
  mock_api_class.return_value = mock.MagicMock()
  result: click_testing.Result = _CallCLI(
    ['-vvv', '--out', str(tmp_path), 'make', 'prompt', '--vseed', '42', '--vstrength', '0.0'],
  )
  assert not result.exit_code
  call_args: dict[str, object] = mock_meta_factory.call_args[0][0]
  assert call_args['v_seed'] is None


def testMakeNoDbNoOutRaises() -> None:
  """Make command fails with --no-db and no --out."""
  result: click_testing.Result = _CallCLI(
    ['-vvv', '--no-db', 'make', 'test prompt'],
  )
  assert result.exit_code != 0


def testMakeA1111SamplerRaises() -> None:
  """Make command fails with an A1111-only sampler."""
  sampler_name: str = next(iter(base.SamplerA1111)).value
  result: click_testing.Result = _CallCLI(
    [
      '-vvv',
      '--no-db',
      '--out',
      '/tmp',  # noqa: S108
      'make',
      'prompt',
      '--sampler',
      sampler_name,
    ],
  )
  assert result.exit_code != 0
