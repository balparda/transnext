# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: experiment.py CLI entry point and cli/cliexperiment.py sub-commands."""

from __future__ import annotations

from unittest import mock

import click
import pytest
from click import testing as click_testing
from transcrypto.utils import config as app_config
from transcrypto.utils import logging as cli_logging
from typer import testing

from transnext import experiment
from transnext.cli import cliexperiment
from transnext.core import base, newton


@pytest.fixture(autouse=True)
def reset_cli() -> None:
  """Reset CLI singleton before each test."""
  cli_logging.ResetConsole()
  app_config.ResetConfig()


def _CallCLI(args: list[str]) -> click_testing.Result:
  """Call the experiment CLI with args.

  Args:
    args (list[str]): CLI arguments.

  Returns:
    click_testing.Result: CLI result.

  """
  return testing.CliRunner().invoke(experiment.app, args)


def _MockDB() -> mock.MagicMock:
  """Create a mock AIDatabase for use as a context manager.

  Returns:
    mock.MagicMock: Configured mock DB.

  """
  mock_db_inst = mock.MagicMock()
  mock_db_inst.__enter__ = mock.Mock(return_value=mock_db_inst)
  mock_db_inst.__exit__ = mock.Mock(return_value=False)
  mock_db_inst.GetModelHash.return_value = 'model-hash'
  mock_db_inst.experiments = {}
  mock_db_inst.QueryNormalize = lambda x: x  # pyright: ignore[reportUnknownLambdaType]
  mock_db_inst.output = None
  return mock_db_inst


# ─── Guard against accidental debug recording ─────────────────────────────────


def testNotForgotRecordingOn() -> None:
  """Make sure _DEBUG_RECORD was not accidentally left on before committing."""
  assert not cliexperiment._DEBUG_RECORD


# ─── experiment.py — entry point ──────────────────────────────────────────────


def testVersion() -> None:
  """--version prints the version and exits cleanly."""
  result: click_testing.Result = _CallCLI(['--version'])
  assert result.exit_code == 0
  assert '1.1.0' in result.output


def testNoArgsShowsHelp() -> None:
  """No arguments shows help text."""
  result: click_testing.Result = _CallCLI([])
  assert result.exit_code in {0, 2}  # typer no_args_is_help exits with 2


def testMarkdown() -> None:
  """Markdown command emits Markdown-formatted docs."""
  result: click_testing.Result = _CallCLI(['--out', '/tmp', 'markdown'])  # noqa: S108
  assert result.exit_code == 0
  # transnext CLI docs markdown always have a '# ' heading
  assert '#' in result.output


# ─── cliexperiment._BuildAxes ─────────────────────────────────────────────────


def testBuildAxesValid() -> None:
  """_BuildAxes parses valid 'KEY:VAL1|VAL2' axis strings."""
  axs = cliexperiment._BuildAxes(['cfg_scale:6.0|7.0|8.0', 'sampler:Euler|DPM++ SDE'])
  assert len(axs) == 2
  assert axs[0] == newton.AxisType(key='cfg_scale', values=[60, 70, 80])
  assert axs[1]['key'] == 'sampler'
  assert axs[1]['values'] == ['Euler', 'DPM++ SDE']


def testBuildAxesEmptyListRaises() -> None:
  """_BuildAxes raises UsageError when no axes are given."""
  with pytest.raises(click.UsageError, match='axis'):
    cliexperiment._BuildAxes([])


def testBuildAxesDuplicateKeyRaises() -> None:
  """_BuildAxes raises UsageError when the same key appears twice."""
  with pytest.raises(click.UsageError, match='Duplicate'):
    cliexperiment._BuildAxes(['cfg_scale:7.0|8.0', 'cfg_scale:9.0|10.0'])


def testBuildAxesInvalidFormatRaises() -> None:
  """_BuildAxes raises base.Error for strings without a colon separator."""
  with pytest.raises(base.Error, match='format'):
    cliexperiment._BuildAxes(['no-colon-here'])


def testBuildAxesEmptyValuesRaises() -> None:
  """_BuildAxes raises base.Error when the values part is empty."""
  with pytest.raises(base.Error):
    cliexperiment._BuildAxes(['cfg_scale:'])


# ─── cliexperiment.New — full CLI integration ─────────────────────────────────


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
@mock.patch('transnext.core.newton.Experiments')
@mock.patch('transnext.core.db.AIMetaTypeFactory')
def testNewSuccess(
  mock_meta_factory: mock.Mock,
  mock_exps_class: mock.Mock,
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
  tmp_path: mock.Mock,
) -> None:
  """New command runs successfully with minimal valid args."""
  meta = mock.MagicMock()
  mock_meta_factory.return_value = meta
  mock_db_inst = _MockDB()
  mock_db_class.return_value = mock_db_inst
  mock_api_inst = mock.MagicMock()
  mock_api_class.return_value = mock_api_inst
  mock_exp = mock.MagicMock()
  mock_exp.Run.return_value = iter([])  # no images
  mock_exps_inst = mock.MagicMock()
  mock_exps_inst.Make.return_value = mock_exp
  mock_exps_class.return_value = mock_exps_inst
  result: click_testing.Result = _CallCLI(
    [
      '--out',
      str(tmp_path),
      'new',
      'a photo of a cat',
      '--seeds',
      '42',
      '--axis',
      'cfg_scale:7.0|8.0',
    ]
  )
  assert not result.exit_code, result.output
  mock_exps_inst.Make.assert_called_once()
  mock_exp.Run.assert_called_once()


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
@mock.patch('transnext.core.newton.Experiments')
@mock.patch('transnext.core.db.AIMetaTypeFactory')
def testNewNegativePrompt(
  mock_meta_factory: mock.Mock,
  mock_exps_class: mock.Mock,
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
  tmp_path: mock.Mock,
) -> None:
  """New command forwards negative prompt to AIMetaTypeFactory."""
  meta = mock.MagicMock()
  mock_meta_factory.return_value = meta
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_class.return_value = mock_db_inst
  mock_api_class.return_value = mock.MagicMock()
  mock_exp = mock.MagicMock()
  mock_exp.Run.return_value = iter([])
  mock_exps_inst = mock.MagicMock()
  mock_exps_inst.Make.return_value = mock_exp
  mock_exps_class.return_value = mock_exps_inst
  result = _CallCLI(
    [
      '--out',
      str(tmp_path),
      'new',
      'a photo of a cat',
      '-n',
      'ugly, blurry',
      '--seeds',
      '42',
      '--axis',
      'sampler:Euler|DPM++ SDE',
    ]
  )
  assert not result.exit_code, result.output
  call_args: dict[str, object] = mock_meta_factory.call_args[0][0]
  assert call_args['negative'] == 'ugly, blurry'


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
@mock.patch('transnext.core.newton.Experiments')
@mock.patch('transnext.core.db.AIMetaTypeFactory')
def testNewNoDbNoOutRaises(
  mock_meta_factory: mock.Mock,
  _mock_exps_class: mock.Mock,
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
) -> None:
  """New raises UsageError when both --no-db and no --out are given."""
  mock_meta_factory.return_value = mock.MagicMock()
  mock_db_class.return_value = _MockDB()
  mock_api_class.return_value = mock.MagicMock()
  result = _CallCLI(
    [
      '--no-db',
      'new',
      'a photo of a cat',
      '--seeds',
      '42',
      '--axis',
      'cfg_scale:7.0|8.0',
    ]
  )
  assert result.exit_code != 0
  assert 'out' in result.output.lower()


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
@mock.patch('transnext.core.newton.Experiments')
@mock.patch('transnext.core.db.AIMetaTypeFactory')
def testNewA1111SamplerRaises(
  mock_meta_factory: mock.Mock,
  _mock_exps_class: mock.Mock,
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
  tmp_path: mock.Mock,
) -> None:
  """New raises UsageError when a A1111-only sampler is used."""
  a1111_sampler: base.SamplerA1111 = next(iter(base.SamplerA1111))
  mock_meta_factory.return_value = mock.MagicMock()
  mock_db_class.return_value = _MockDB()
  mock_api_class.return_value = mock.MagicMock()
  result: click_testing.Result = _CallCLI(
    [
      '--out',
      str(tmp_path),
      'new',
      'a photo of a cat',
      '--seeds',
      '42',
      '--sampler',
      a1111_sampler.value,
      '--axis',
      'cfg_scale:7.0|8.0',
    ]
  )
  assert result.exit_code != 0


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
@mock.patch('transnext.core.newton.Experiments')
@mock.patch('transnext.core.db.AIMetaTypeFactory')
def testNewWithOutputSetsDBOutput(
  mock_meta_factory: mock.Mock,
  mock_exps_class: mock.Mock,
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
  tmp_path: mock.Mock,
) -> None:
  """New with --out sets the output directory on the DB instance."""
  mock_meta_factory.return_value = mock.MagicMock()
  mock_db_inst: mock.MagicMock = _MockDB()
  mock_db_class.return_value = mock_db_inst
  mock_api_class.return_value = mock.MagicMock()
  mock_exp = mock.MagicMock()
  mock_exp.Run.return_value = iter([])
  mock_exps_inst = mock.MagicMock()
  mock_exps_inst.Make.return_value = mock_exp
  mock_exps_class.return_value = mock_exps_inst
  result = _CallCLI(
    [
      '--out',
      str(tmp_path),
      'new',
      'a photo of a cat',
      '--seeds',
      '42',
      '--axis',
      'cfg_scale:7.0|8.0',
    ]
  )
  assert not result.exit_code, result.output
  assert mock_db_inst.output == tmp_path


@mock.patch('transnext.core.sdnapi.API')
@mock.patch('transnext.core.db.AIDatabase')
@mock.patch('transnext.core.newton.Experiments')
@mock.patch('transnext.core.db.AIMetaTypeFactory')
def testNewRandomSeedNegativeOne(
  mock_meta_factory: mock.Mock,
  mock_exps_class: mock.Mock,
  mock_db_class: mock.Mock,
  mock_api_class: mock.Mock,
  tmp_path: mock.Mock,
) -> None:
  """New command treats seed -1 as 'random seed' (replaces with a valid positive seed)."""
  mock_meta_factory.return_value = mock.MagicMock()
  mock_db_inst = _MockDB()
  mock_db_class.return_value = mock_db_inst
  mock_api_class.return_value = mock.MagicMock()
  mock_exp = mock.MagicMock()
  mock_exp.Run.return_value = iter([])
  mock_exps_inst = mock.MagicMock()
  mock_exps_inst.Make.return_value = mock_exp
  mock_exps_class.return_value = mock_exps_inst
  result: click_testing.Result = _CallCLI(
    [
      '--out',
      str(tmp_path),
      'new',
      'a photo of a cat',
      '--seeds',
      '-1',
      '--axis',
      'cfg_scale:7.0|8.0',
    ]
  )
  assert not result.exit_code, result.output
  # check that seeds passed to Make are valid (> 0)
  _, make_kwargs = mock_exps_inst.Make.call_args
  for seed in make_kwargs.get('seeds', mock_exps_inst.Make.call_args[0][2]):
    assert seed >= 1
