# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: transnext.core.newton module."""

from __future__ import annotations

import io
import pathlib
from collections import abc
from typing import cast
from unittest import mock

import pytest
from PIL import Image
from transcrypto.core import hashes

from transnext.core import base, db, newton

# ─── helpers ──────────────────────────────────────────────────────────────────


def _MakeMeta(overrides: dict[str, object] | None = None) -> db.AIMetaType:
  """Create an AIMetaType with a valid fixed seed for testing."""  # noqa: DOC201
  defaults: dict[str, object] = {'seed': 42, 'model_hash': 'abc123', 'cfg_scale': 70}
  if overrides:
    defaults.update(overrides)
  return db.AIMetaTypeFactory(defaults)


def _SmallPNG(w: int = 64, h: int = 64) -> bytes:
  """Create a minimal PNG image as bytes."""  # noqa: DOC201
  buf = io.BytesIO()
  Image.new('RGB', (w, h), color='red').save(buf, format='PNG')
  return buf.getvalue()


def _MakeDBImageWithPath(
  img_hash: str,
  path: pathlib.Path,
  w: int = 64,
  h: int = 64,
) -> db.DBImageType:
  """Create a DBImageType that points to a real file path."""  # noqa: DOC201
  return db.DBImageType(
    hash=img_hash,
    raw_hash='raw_' + img_hash[:8],
    size=100,
    width=w,
    height=h,
    format=base.ImageFormat.PNG.value,
    info=None,
    paths={
      str(path): db.DBImagePathType(
        main=False,
        created_at=1000000,
        origin=None,
        version=None,
        ai_meta=None,
        sd_info=None,
        sd_params=None,
        parse_errors=None,
      )
    },
  )


def _MockAIDB(experiments: dict[str, newton.ExperimentType] | None = None) -> mock.MagicMock:
  """Create a MagicMock AIDatabase with sensible defaults for experiment tests."""  # noqa: DOC201
  ai_db = mock.MagicMock(spec=db.AIDatabase)
  ai_db.experiments = experiments if experiments is not None else {}
  ai_db.QueryNormalize = lambda x: x  # pyright: ignore[reportUnknownLambdaType] # identity
  ai_db.output = None
  ai_db.Image.return_value = None  # no existing images by default
  return ai_db


def _SimpleCFGExperiment(
  ai_db: mock.MagicMock,
  seeds: list[int] | None = None,
  cfg_values: list[int] | None = None,
) -> newton.Experiment:
  """Create a simple Experiment with a cfg_scale axis."""  # noqa: DOC201
  meta: db.AIMetaType = _MakeMeta()
  axes: list[newton.AxisType] = [newton.AxisType(key='cfg_scale', values=cfg_values or [70, 80])]
  return newton.Experiment(
    ai_db, meta, axes, seeds or [42, 123], newton.ExperimentOptionsType(sidecar=None)
  )


# ─── Tunnel / ReplaceMe ────────────────────────────────────────────────────────


def testTunnelPassThrough() -> None:
  """Tunnel returns the input value unchanged."""
  assert newton.Tunnel(42) == 42
  assert newton.Tunnel('hello') == 'hello'


def testReplaceMeRaises() -> None:
  """ReplaceMe raises NotImplementedError when called."""
  with pytest.raises(NotImplementedError):
    newton.ReplaceMe(42)
  with pytest.raises(NotImplementedError):
    newton.ReplaceMe('x')


# ─── IntRangeTunnel ────────────────────────────────────────────────────────────


def testIntRangeTunnelValidValue() -> None:
  """IntRangeTunnel passes through values within range."""
  fn: abc.Callable[[int], int] = newton.IntRangeTunnel(1, 10)
  assert fn(1) == 1
  assert fn(5) == 5
  assert fn(10) == 10


def testIntRangeTunnelBelowMinRaises() -> None:
  """IntRangeTunnel raises Error for values below min."""
  fn: abc.Callable[[int], int] = newton.IntRangeTunnel(1, 10)
  with pytest.raises(newton.Error, match='out of range'):
    fn(0)


def testIntRangeTunnelAboveMaxRaises() -> None:
  """IntRangeTunnel raises Error for values above max."""
  fn: abc.Callable[[int], int] = newton.IntRangeTunnel(1, 10)
  with pytest.raises(newton.Error, match='out of range'):
    fn(11)


# ─── PromptReplaceTunnel ───────────────────────────────────────────────────────


def testPromptReplaceTunnelBasic() -> None:
  """PromptReplaceTunnel replaces % placeholder with the given value."""
  fn: abc.Callable[[str], str] = newton.PromptReplaceTunnel('a % cat')
  assert fn('black') == 'a black cat'
  assert fn('white') == 'a white cat'


def testPromptReplaceTunnelNoPlaceholderRaises() -> None:
  """PromptReplaceTunnel raises Error when prompt has no placeholder."""
  with pytest.raises(newton.Error, match='placeholder'):
    newton.PromptReplaceTunnel('no placeholder here')


def testPromptReplaceTunnelDoublePct() -> None:
  """PromptReplaceTunnel treats %% as a literal percent."""
  fn: abc.Callable[[str], str] = newton.PromptReplaceTunnel('a %% % cat')
  assert fn('black') == 'a % black cat'


def testPromptReplaceTunnelMultiplePlaceholders() -> None:
  """PromptReplaceTunnel replaces all % placeholders."""
  fn: abc.Callable[[str], str] = newton.PromptReplaceTunnel('% and %')
  assert fn('x') == 'x and x'


# ─── KeyHash ───────────────────────────────────────────────────────────────────


def testKeyHashDeterministic() -> None:
  """KeyHash returns the same hash for the same key."""
  k: newton.ExperimentKeyType = ['Euler', 70]
  assert newton.KeyHash(k) == newton.KeyHash(k)


def testKeyHashDifferentKeys() -> None:
  """Different keys produce different hashes."""
  assert newton.KeyHash(['a']) != newton.KeyHash(['b'])
  assert newton.KeyHash([1, 2]) != newton.KeyHash([2, 1])


def testKeyHashIsStr() -> None:
  """KeyHash returns a hex string."""
  h: str = newton.KeyHash([42, 'foo'])
  assert isinstance(h, str)
  assert len(h) == 64  # SHA256 hex


# ─── SeedValidate / SamplerValidate / CFGValidate ─────────────────────────────


def testSeedValidateValid() -> None:
  """SeedValidate passes valid seeds."""
  assert newton.SeedValidate(1) == 1
  assert newton.SeedValidate(base.SD_MAX_SEED) == base.SD_MAX_SEED


def testSeedValidateZeroRaises() -> None:
  """SeedValidate rejects seed=0."""
  with pytest.raises(newton.Error):
    newton.SeedValidate(0)


def testSamplerValidateValid() -> None:
  """SamplerValidate returns valid sampler strings."""
  sampler: base.Sampler = next(s for s in base.Sampler if s not in base.SamplerA1111)
  assert newton.SamplerValidate(sampler.value) == sampler.value


def testSamplerValidateInvalidRaises() -> None:
  """SamplerValidate raises ValueError for unknown sampler."""
  with pytest.raises(ValueError):  # noqa: PT011
    newton.SamplerValidate('not-a-sampler')


def testCFGValidateValid() -> None:
  """CFGValidate passes values within range."""
  assert newton.CFGValidate(base.SD_MIN_CFG_SCALE) == base.SD_MIN_CFG_SCALE
  assert newton.CFGValidate(base.SD_MAX_CFG_SCALE) == base.SD_MAX_CFG_SCALE


def testCFGValidateOutOfRangeRaises() -> None:
  """CFGValidate raises Error for out-of-range values."""
  with pytest.raises(newton.Error):
    newton.CFGValidate(base.SD_MAX_CFG_SCALE + 1)


# ─── Experiments class ────────────────────────────────────────────────────────


def testExperimentsMakeCreatesNew() -> None:
  """Experiments.Make creates a new experiment not previously seen."""
  ai_db: mock.MagicMock = _MockAIDB()
  meta: db.AIMetaType = _MakeMeta()
  axes: list[newton.AxisType] = [newton.AxisType(key='cfg_scale', values=[70, 80])]
  seeds: list[int] = [42, 123]
  exps = newton.Experiments(ai_db)
  exp: newton.Experiment = exps.Make(meta, axes, seeds, newton.ExperimentOptionsType(sidecar=None))
  assert isinstance(exp, newton.Experiment)
  assert exp.experiment_hash in ai_db.experiments


def testExperimentsMakeReturnsExistingIfSameHash() -> None:
  """Experiments.Make returns the existing Experiment if hash already in DB."""
  ai_db: mock.MagicMock = _MockAIDB()
  meta: db.AIMetaType = _MakeMeta()
  axes: list[newton.AxisType] = [newton.AxisType(key='cfg_scale', values=[70])]
  seeds: list[int] = [42]
  exps = newton.Experiments(ai_db)
  exp1: newton.Experiment = exps.Make(meta, axes, seeds, newton.ExperimentOptionsType(sidecar=None))
  exp2: newton.Experiment = exps.Make(meta, axes, seeds, newton.ExperimentOptionsType(sidecar=None))
  assert exp1 is exp2


def testExperimentsInitLoadsExistingExperiment() -> None:
  """Experiments.__init__ loads existing experiments from the DB dict."""
  ai_db: mock.MagicMock = _MockAIDB()
  meta: db.AIMetaType = _MakeMeta()
  axes: list[newton.AxisType] = [newton.AxisType(key='cfg_scale', values=[70])]
  seeds: list[int] = [42]
  # first, create an experiment to populate ai_db.experiments
  exps = newton.Experiments(ai_db)
  exp: newton.Experiment = exps.Make(meta, axes, seeds, newton.ExperimentOptionsType(sidecar=None))
  eh: str = exp.experiment_hash
  # now initialize a new Experiments from the same dict
  exps2 = newton.Experiments(ai_db)
  assert eh in exps2._objects


# ─── Experiment.__init__ ──────────────────────────────────────────────────────


def testExperimentInitBasic() -> None:
  """Experiment can be created with a cfg_scale axis."""
  ai_db: mock.MagicMock = _MockAIDB()
  exp: newton.Experiment = _SimpleCFGExperiment(ai_db)
  assert len(exp._keys) == 2  # 2 cfg values
  assert len(exp._seeds) == 2


def testExperimentInitInvalidSeedRaises() -> None:
  """Experiment raises Error when a seed is out of range."""
  ai_db: mock.MagicMock = _MockAIDB()
  meta: db.AIMetaType = _MakeMeta()
  axes: list[newton.AxisType] = [newton.AxisType(key='cfg_scale', values=[70])]
  with pytest.raises(newton.Error, match='out of range'):
    newton.Experiment(ai_db, meta, axes, [0], newton.ExperimentOptionsType(sidecar=None))


def testExperimentInitInvalidAxisKeyRaises() -> None:
  """Experiment raises Error for unknown axis key."""
  ai_db: mock.MagicMock = _MockAIDB()
  meta: db.AIMetaType = _MakeMeta()
  with pytest.raises(ValueError, match='not-a-key'):
    newton.Experiment(
      ai_db,
      meta,
      [newton.AxisType(key='not-a-key', values=['x'])],
      [42],
      newton.ExperimentOptionsType(sidecar=None),
    )


def testExperimentInitSamplerAxisValid() -> None:
  """Experiment with a sampler axis validates sampler names."""
  ai_db = _MockAIDB()
  meta = _MakeMeta()
  sampler: base.Sampler = next(s for s in base.Sampler if s not in base.SamplerA1111)
  exp = newton.Experiment(
    ai_db,
    meta,
    [newton.AxisType(key='sampler', values=[sampler.value])],
    [42],
    newton.ExperimentOptionsType(sidecar=None),
  )
  assert len(exp._keys) == 1


def testExperimentInitModelAxisValidates() -> None:
  """Experiment with model axis calls GetModel for validation."""
  ai_db = _MockAIDB()
  ai_db.GetModel.return_value = db.AIModelType(
    hash='h1',
    name='m1',
    alias='m1',
    autov3=None,
    path='',
    model_type='safetensors',
    function=db.ModelFunction.Model.value,
    metadata={},
    sidecar=None,
    description=None,
  )
  meta = _MakeMeta()
  exp = newton.Experiment(
    ai_db,
    meta,
    [newton.AxisType(key='model_hash', values=['h1'])],
    [42],
    newton.ExperimentOptionsType(sidecar=None),
  )
  assert len(exp._keys) == 1
  ai_db.GetModel.assert_called_with('h1')


def testExperimentInitPositiveAxisSetsTunnel() -> None:
  """Experiment sets up a PromptReplaceTunnel for positive axis."""
  ai_db = _MockAIDB()
  meta = _MakeMeta({'positive': 'a % photo'})
  exp = newton.Experiment(
    ai_db,
    meta,
    [newton.AxisType(key='positive', values=['cat', 'dog'])],
    [42],
    newton.ExperimentOptionsType(sidecar=None),
  )
  # keys are the applied values (prompt with placeholder replaced)
  assert ('a cat photo',) in exp._keys
  assert ('a dog photo',) in exp._keys


def testExperimentInitNegativeAxisSetsTunnel() -> None:
  """Experiment sets up a PromptReplaceTunnel for negative axis."""
  ai_db = _MockAIDB()
  meta = _MakeMeta({'negative': 'no % please'})
  exp = newton.Experiment(
    ai_db,
    meta,
    [newton.AxisType(key='negative', values=['blur', 'noise'])],
    [42],
    newton.ExperimentOptionsType(sidecar=None),
  )
  assert ('no blur please',) in exp._keys
  assert ('no noise please',) in exp._keys


def testExperimentInitDuplicateModelAxisRaises() -> None:
  """Experiment raises Error when model axis appears twice."""
  ai_db = _MockAIDB()
  ai_db.GetModel.return_value = db.AIModelType(
    hash='h1',
    name='m1',
    alias='m1',
    autov3=None,
    path='',
    model_type='safetensors',
    function=db.ModelFunction.Model.value,
    metadata={},
    sidecar=None,
    description=None,
  )
  meta = _MakeMeta()
  with pytest.raises(newton.Error, match='Multiple model axes'):
    newton.Experiment(
      ai_db,
      meta,
      [
        newton.AxisType(key='model_hash', values=['h1']),
        newton.AxisType(key='model_hash', values=['h1']),
      ],
      [42],
      newton.ExperimentOptionsType(sidecar=None),
    )


def testExperimentInitLoadedValid() -> None:
  """Experiment initializer re-uses loaded experiment data when valid."""
  ai_db = _MockAIDB()
  exps = newton.Experiments(ai_db)
  meta = _MakeMeta()
  axes: list[newton.AxisType] = [newton.AxisType(key='cfg_scale', values=[70])]
  seeds = [42]
  exp = exps.Make(meta, axes, seeds, newton.ExperimentOptionsType(sidecar=None))
  loaded = exp.experiment
  # re-load the same experiment
  exp2 = newton.Experiment(
    ai_db, meta, axes, seeds, newton.ExperimentOptionsType(sidecar=None), loaded=loaded
  )
  assert exp2.experiment_hash == exp.experiment_hash


def testExperimentInitLoadedConfigMismatchRaises() -> None:
  """Experiment raises Error when loaded config doesn't match."""
  ai_db = _MockAIDB()
  exps = newton.Experiments(ai_db)
  meta = _MakeMeta()
  axes: list[newton.AxisType] = [newton.AxisType(key='cfg_scale', values=[70])]
  exp = exps.Make(meta, axes, [42], newton.ExperimentOptionsType(sidecar=None))
  loaded = exp.experiment
  different_meta = _MakeMeta({'positive': 'changed'})
  with pytest.raises(newton.Error, match='config'):
    newton.Experiment(
      ai_db,
      different_meta,
      axes,
      [42],
      newton.ExperimentOptionsType(sidecar=None),
      loaded=loaded,
    )


def testExperimentInitLoadedAxesMismatchRaises() -> None:
  """Experiment raises Error when loaded axes don't match."""
  ai_db = _MockAIDB()
  exps = newton.Experiments(ai_db)
  meta = _MakeMeta()
  axes: list[newton.AxisType] = [newton.AxisType(key='cfg_scale', values=[70])]
  exp = exps.Make(meta, axes, [42], newton.ExperimentOptionsType(sidecar=None))
  loaded = exp.experiment
  different_axes: list[newton.AxisType] = [newton.AxisType(key='cfg_scale', values=[80])]
  with pytest.raises(newton.Error, match='axes'):
    newton.Experiment(
      ai_db,
      meta,
      different_axes,
      [42],
      newton.ExperimentOptionsType(sidecar=None),
      loaded=loaded,
    )


def testExperimentInitLoadedSeedsMismatchRaises() -> None:
  """Experiment raises Error when loaded seeds don't match."""
  ai_db = _MockAIDB()
  exps = newton.Experiments(ai_db)
  meta = _MakeMeta()
  axes: list[newton.AxisType] = [newton.AxisType(key='cfg_scale', values=[70])]
  exp = exps.Make(meta, axes, [42], newton.ExperimentOptionsType(sidecar=None))
  loaded = exp.experiment
  with pytest.raises(newton.Error, match='seeds'):
    newton.Experiment(
      ai_db, meta, axes, [999], newton.ExperimentOptionsType(sidecar=None), loaded=loaded
    )


def testExperimentInitLoadedHashMismatchRaises() -> None:
  """Experiment raises Error when loaded hash doesn't match recomputed hash."""
  ai_db = _MockAIDB()
  exps = newton.Experiments(ai_db)
  meta = _MakeMeta()
  axes: list[newton.AxisType] = [newton.AxisType(key='cfg_scale', values=[70])]
  exp = exps.Make(meta, axes, [42], newton.ExperimentOptionsType(sidecar=None))
  loaded = exp.experiment
  tampered = dict(loaded)
  tampered['hash'] = 'bad-hash'
  with pytest.raises(newton.Error, match='hash'):
    newton.Experiment(
      ai_db,
      meta,
      axes,
      [42],
      newton.ExperimentOptionsType(sidecar=None),
      loaded=cast('newton.ExperimentType', tampered),
    )


# ─── Experiment.experiment_hash ───────────────────────────────────────────────


def testExperimentHashDeterministic() -> None:
  """Experiment hash is deterministic for the same config/axes/seeds."""
  ai_db = _MockAIDB()
  meta = _MakeMeta()
  axes: list[newton.AxisType] = [newton.AxisType(key='cfg_scale', values=[70])]
  exp1 = newton.Experiment(ai_db, meta, axes, [42], newton.ExperimentOptionsType(sidecar=None))
  exp2 = newton.Experiment(ai_db, meta, axes, [42], newton.ExperimentOptionsType(sidecar=None))
  assert exp1.experiment_hash == exp2.experiment_hash


def testExperimentHashDiffersForDifferentConfig() -> None:
  """Different configs produce different hashes."""
  ai_db = _MockAIDB()
  axes: list[newton.AxisType] = [newton.AxisType(key='cfg_scale', values=[70])]
  exp1 = newton.Experiment(
    ai_db, _MakeMeta({'positive': 'cat'}), axes, [42], newton.ExperimentOptionsType(sidecar=None)
  )
  exp2 = newton.Experiment(
    ai_db, _MakeMeta({'positive': 'dog'}), axes, [42], newton.ExperimentOptionsType(sidecar=None)
  )
  assert exp1.experiment_hash != exp2.experiment_hash


# ─── Experiment._ActualCellSize ────────────────────────────────────────────────


def testActualCellSizeNoResultsRaises() -> None:
  """_ActualCellSize raises Error when no results are in the DB."""
  ai_db = _MockAIDB()
  ai_db.Image.return_value = None
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70])
  with pytest.raises(newton.Error, match='No result images'):
    exp._ActualCellSize()


def testActualCellSizeReturnsFromDB() -> None:
  """_ActualCellSize returns (w+padding, h+padding) from a known DB image."""
  ai_db = _MockAIDB()
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70])
  # inject a result into the results dict
  kh = next(iter(exp._k_dict))
  exp._results['42'][kh] = 'fake-hash'
  ai_db.Image.return_value = db.DBImageType(
    hash='fake-hash',
    raw_hash='r',
    size=100,
    width=64,
    height=32,
    format='PNG',
    info=None,
    paths={},
  )
  w, h = exp._ActualCellSize()
  assert w == 64 + newton._GRID_PADDING
  assert h == 32 + newton._GRID_PADDING


# ─── Experiment._AxisValueLabel ────────────────────────────────────────────────


def testAxisValueLabelSeed() -> None:
  """_AxisValueLabel returns 'Seed N' for seed field."""
  ai_db = _MockAIDB()
  exp = _SimpleCFGExperiment(ai_db)
  assert exp._AxisValueLabel(newton.AxisField.Seed, 42) == 'Seed 42'


def testAxisValueLabelCFG() -> None:
  """_AxisValueLabel returns 'CFG N.N' for CFG field."""
  ai_db = _MockAIDB()
  exp = _SimpleCFGExperiment(ai_db)
  assert exp._AxisValueLabel(newton.AxisField.CFG, 70) == 'CFG 7.0'


def testAxisValueLabelSampler() -> None:
  """_AxisValueLabel returns the sampler value as-is."""
  ai_db = _MockAIDB()
  sampler = next(s for s in base.Sampler if s not in base.SamplerA1111)
  exp = newton.Experiment(
    ai_db,
    _MakeMeta(),
    [newton.AxisType(key='sampler', values=[sampler.value])],
    [42],
    newton.ExperimentOptionsType(sidecar=None),
  )
  result = exp._AxisValueLabel(newton.AxisField.Sampler, sampler.value)
  assert result == sampler.value


def testAxisValueLabelModelFound() -> None:
  """_AxisValueLabel returns model alias when model is found in DB."""
  ai_db = _MockAIDB()
  ai_db.GetModel.return_value = db.AIModelType(
    hash='h1',
    name='full-name',
    alias='alias',
    autov3=None,
    path='',
    model_type='safetensors',
    function=db.ModelFunction.Model.value,
    metadata={},
    sidecar=None,
    description=None,
  )
  exp = _SimpleCFGExperiment(ai_db)
  result = exp._AxisValueLabel(newton.AxisField.Model, 'h1')
  assert result == 'alias'


def testAxisValueLabelModelNotFound() -> None:
  """_AxisValueLabel falls back to hash prefix when model not found."""
  ai_db = _MockAIDB()
  ai_db.GetModel.side_effect = db.Error('not found')
  exp = _SimpleCFGExperiment(ai_db)
  result = exp._AxisValueLabel(newton.AxisField.Model, 'abcdef123456789')
  assert result == 'abcdef123456'  # first 12 chars


def testAxisValueLabelPositiveTruncatesLongText() -> None:
  """_AxisValueLabel truncates very long positive/negative prompt labels."""
  ai_db = _MockAIDB()
  meta = _MakeMeta({'positive': 'a % photo'})
  exp = newton.Experiment(
    ai_db,
    meta,
    [newton.AxisType(key='positive', values=['cat'])],
    [42],
    newton.ExperimentOptionsType(sidecar=None),
  )
  long_text = 'x' * (newton._LABEL_TRUNCATE + 10)
  result = exp._AxisValueLabel(newton.AxisField.Positive, long_text)
  assert result.endswith('...')
  assert len(result) == newton._LABEL_TRUNCATE + 3  # truncated + '...'


# ─── Experiment._DimValues ─────────────────────────────────────────────────────


def testDimValuesSeedReturnsSeedList() -> None:
  """_DimValues returns the seed list for the Seed field."""
  ai_db = _MockAIDB()
  exp = _SimpleCFGExperiment(ai_db, seeds=[42, 99])
  assert list(exp._DimValues(newton.AxisField.Seed)) == [42, 99]


def testDimValuesCFGReturnsAxisValues() -> None:
  """_DimValues returns the axis values for the CFG field."""
  ai_db = _MockAIDB()
  exp = _SimpleCFGExperiment(ai_db, cfg_values=[70, 80, 90])
  vals = list(exp._DimValues(newton.AxisField.CFG))
  assert vals == [70, 80, 90]


def testDimValuesUnknownFieldRaises() -> None:
  """_DimValues raises Error for a field not in the experiment axes."""
  ai_db = _MockAIDB()
  exp = _SimpleCFGExperiment(ai_db)
  # Sampler axis is not in the experiment, so asking for it should fail
  with pytest.raises(newton.Error, match='not in experiment axes'):
    exp._DimValues(newton.AxisField.Sampler)


# ─── Experiment._LookupResultHash ─────────────────────────────────────────────


def testLookupResultHashFound() -> None:
  """_LookupResultHash returns the stored hash when result exists."""
  ai_db = _MockAIDB()
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70])
  kh = next(iter(exp._k_dict))
  exp._results['42'][kh] = 'target-hash'
  dim_indices = {newton.AxisField.Seed: 0, newton.AxisField.CFG: 0}
  assert exp._LookupResultHash(dim_indices) == 'target-hash'


def testLookupResultHashMissingRaises() -> None:
  """_LookupResultHash raises Error when result is None (not yet run)."""
  ai_db = _MockAIDB()
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70])
  # results are None by default
  dim_indices = {newton.AxisField.Seed: 0, newton.AxisField.CFG: 0}
  with pytest.raises(newton.Error, match='Missing result'):
    exp._LookupResultHash(dim_indices)


# ─── Experiment.Run ────────────────────────────────────────────────────────────


def testExperimentRunYieldsAllImages(tmp_path: pathlib.Path) -> None:
  """Experiment.Run yields one image per (key x seed) combination plus grid images."""
  ai_db = _MockAIDB()
  png_bytes = _SmallPNG(64, 32)
  png_hash = hashes.Hash256(png_bytes).hex()
  img_file = tmp_path / 'img.png'
  img_file.write_bytes(png_bytes)
  img_obj = _MakeDBImageWithPath(png_hash, img_file, w=64, h=32)
  ai_db.Txt2Img.return_value = (img_obj, png_bytes)
  ai_db.Image.return_value = img_obj
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70, 80])
  api = mock.MagicMock(spec=db.APIProtocol)
  results = list(exp.Run(api))
  # 2 generated images + 1 grid (2 cols x 1 row = 1 grid)
  assert ai_db.Txt2Img.call_count == 2
  assert len(results) >= 3  # at minimum 2 generated + 1 grid


def testExperimentRunSkipsExistingResult(tmp_path: pathlib.Path) -> None:
  """Experiment.Run skips combinations that already have a result on disk."""
  ai_db = _MockAIDB()
  png_bytes = _SmallPNG()
  img_file = tmp_path / 'img.png'
  img_file.write_bytes(png_bytes)
  png_hash = hashes.Hash256(png_bytes).hex()
  img_obj = _MakeDBImageWithPath(png_hash, img_file)
  # pre-populate result  for one key
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70, 80])
  first_kh = next(iter(exp._k_dict))
  exp._results['42'][first_kh] = png_hash
  ai_db.Image.side_effect = lambda h, **_: img_obj if h == png_hash else None  # pyright: ignore[reportUnknownLambdaType]
  ai_db.Txt2Img.return_value = (img_obj, png_bytes)
  results = list(exp.Run(mock.MagicMock(spec=db.APIProtocol)))
  # only 1 combination needs generation (the other is pre-existing)
  assert ai_db.Txt2Img.call_count == 1
  assert len(results) >= 1


def testExperimentRunRedoRegeneratesExisting(tmp_path: pathlib.Path) -> None:
  """Experiment.Run regenerates existing images when redo=True."""
  ai_db = _MockAIDB()
  png_bytes = _SmallPNG()
  img_file = tmp_path / 'img.png'
  img_file.write_bytes(png_bytes)
  png_hash = hashes.Hash256(png_bytes).hex()
  img_obj = _MakeDBImageWithPath(png_hash, img_file)
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70])
  kh = next(iter(exp._k_dict))
  exp._results['42'][kh] = png_hash
  ai_db.Image.return_value = img_obj
  ai_db.Txt2Img.return_value = (img_obj, png_bytes)
  list(exp.Run(mock.MagicMock(spec=db.APIProtocol), redo=True))
  # should regenerate even though result existed
  assert ai_db.Txt2Img.call_count == 1


def testExperimentRunSavesEveryN(tmp_path: pathlib.Path) -> None:
  """Experiment.Run calls Save() every _SAVE_EVERY generated images."""
  ai_db = _MockAIDB()
  png_bytes = _SmallPNG()
  img_file = tmp_path / 'img.png'
  img_file.write_bytes(png_bytes)
  png_hash = hashes.Hash256(png_bytes).hex()
  img_obj = _MakeDBImageWithPath(png_hash, img_file)
  ai_db.Image.return_value = img_obj
  ai_db.Txt2Img.return_value = (img_obj, png_bytes)
  # create experiment with exactly _SAVE_EVERY combinations to trigger a save
  cfg_vals = list(range(70, 70 + newton._SAVE_EVERY))
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=cfg_vals)
  list(exp.Run(mock.MagicMock(spec=db.APIProtocol)))
  ai_db.Save.assert_called()


# ─── Experiment.Grid ──────────────────────────────────────────────────────────


def testExperimentGridNoOutputDir(tmp_path: pathlib.Path) -> None:
  """Grid returns DBImageType and bytes without saving to disk."""
  ai_db = _MockAIDB()
  png_bytes = _SmallPNG(64, 32)
  img_file = tmp_path / 'img.png'
  img_file.write_bytes(png_bytes)
  png_hash = hashes.Hash256(png_bytes).hex()
  img_obj = _MakeDBImageWithPath(png_hash, img_file, w=64, h=32)
  ai_db.Image.return_value = img_obj
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70, 80])
  # manually populate results so Grid doesn't need Run
  for kh in exp._k_dict:
    exp._results['42'][kh] = png_hash
  grids = list(exp.Grid())
  assert len(grids) == 1
  db_img, grid_bytes = grids[0]
  assert len(grid_bytes) > 0
  assert db_img['width'] > 0
  assert next(iter(db_img['paths'])) == ''  # no path (no output dir)  # noqa: PLC1901


def testExperimentGridWithOutputDir(tmp_path: pathlib.Path) -> None:
  """Grid saves PNG files when output_dir is specified."""
  ai_db = _MockAIDB()
  png_bytes = _SmallPNG(64, 32)
  img_file = tmp_path / 'img.png'
  img_file.write_bytes(png_bytes)
  png_hash = hashes.Hash256(png_bytes).hex()
  img_obj = _MakeDBImageWithPath(png_hash, img_file, w=64, h=32)
  ai_db.Image.return_value = img_obj
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70, 80])
  for kh in exp._k_dict:
    exp._results['42'][kh] = png_hash
  out_dir = tmp_path / 'grids'
  out_dir.mkdir()  # must exist before Grid() creates the date subdirectory
  grids = list(exp.Grid(output_dir=out_dir, tm=1000000))
  assert len(grids) == 1
  db_img, _ = grids[0]
  saved_path = next(iter(db_img['paths']))
  assert saved_path  # path is non-empty since we gave output_dir
  assert pathlib.Path(saved_path).exists()


def testExperimentGridOverrideAxes(tmp_path: pathlib.Path) -> None:
  """Grid accepts grid_axes_override to control axis assignment."""
  ai_db = _MockAIDB()
  png_bytes = _SmallPNG(64, 32)
  img_file = tmp_path / 'img.png'
  img_file.write_bytes(png_bytes)
  png_hash = hashes.Hash256(png_bytes).hex()
  img_obj = _MakeDBImageWithPath(png_hash, img_file, w=64, h=32)
  ai_db.Image.return_value = img_obj
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70, 80])
  for kh in exp._k_dict:
    exp._results['42'][kh] = png_hash
  # swap x and y
  override = [newton.AxisField.Seed, newton.AxisField.CFG]
  grids = list(exp.Grid(grid_axes_override=override))
  assert len(grids) == 1


def testExperimentGridOverrideDuplicateRaises(tmp_path: pathlib.Path) -> None:
  """Grid raises Error when grid_axes_override has duplicates."""
  ai_db = _MockAIDB()
  png_bytes = _SmallPNG(64, 32)
  img_file = tmp_path / 'img.png'
  img_file.write_bytes(png_bytes)
  png_hash = hashes.Hash256(png_bytes).hex()
  img_obj = _MakeDBImageWithPath(png_hash, img_file, w=64, h=32)
  ai_db.Image.return_value = img_obj
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70])
  for kh in exp._k_dict:
    exp._results['42'][kh] = png_hash
  with pytest.raises(newton.Error, match='Duplicate dimensions'):
    list(exp.Grid(grid_axes_override=[newton.AxisField.CFG, newton.AxisField.CFG]))


def testExperimentGridOverrideMismatchRaises(tmp_path: pathlib.Path) -> None:
  """Grid raises Error when grid_axes_override dims don't match experiment dims."""
  ai_db = _MockAIDB()
  png_bytes = _SmallPNG(64, 32)
  img_file = tmp_path / 'img.png'
  img_file.write_bytes(png_bytes)
  png_hash = hashes.Hash256(png_bytes).hex()
  img_obj = _MakeDBImageWithPath(png_hash, img_file, w=64, h=32)
  ai_db.Image.return_value = img_obj
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70])
  for kh in exp._k_dict:
    exp._results['42'][kh] = png_hash
  with pytest.raises(newton.Error, match='does not match experiment dims'):
    list(exp.Grid(grid_axes_override=[newton.AxisField.Sampler]))


def testExperimentGridHigherDimProducesMultipleGrids(tmp_path: pathlib.Path) -> None:
  """Grid produces one image per higher-dim combination."""
  ai_db = _MockAIDB()
  png_bytes = _SmallPNG(64, 32)
  img_file = tmp_path / 'img.png'
  img_file.write_bytes(png_bytes)
  png_hash = hashes.Hash256(png_bytes).hex()
  img_obj = _MakeDBImageWithPath(png_hash, img_file, w=64, h=32)
  ai_db.Image.return_value = img_obj
  # 2 cfg values x 2 seeds → 2D grid; use seed as higher dim for 2 separate grids
  exp = _SimpleCFGExperiment(ai_db, seeds=[42, 99], cfg_values=[70, 80])
  for kh in exp._k_dict:
    exp._results['42'][kh] = png_hash
    exp._results['99'][kh] = png_hash
  # make seed the higher dimension → 2 grids (one per seed)
  grids = list(exp.Grid(grid_axes_override=[newton.AxisField.CFG, newton.AxisField.Seed]))
  assert len(grids) == 1  # no higher dims; Seed is Y


def testExperimentGridBuildOneGridImageNotFoundRaises(tmp_path: pathlib.Path) -> None:
  """_BuildOneGrid raises Error when an image hash is not in the DB."""
  ai_db = _MockAIDB()
  png_bytes = _SmallPNG(64, 32)
  img_file = tmp_path / 'img.png'
  img_file.write_bytes(png_bytes)
  png_hash = hashes.Hash256(png_bytes).hex()
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70])
  kh = next(iter(exp._k_dict))
  exp._results['42'][kh] = png_hash
  # ActualCellSize needs at least one result image
  ai_db.Image.side_effect = [
    db.DBImageType(
      hash=png_hash,
      raw_hash='r',
      size=100,
      width=64,
      height=32,
      format='PNG',
      info=None,
      paths={
        str(img_file): db.DBImagePathType(
          main=False,
          created_at=0,
          origin=None,
          version=None,
          ai_meta=None,
          sd_info=None,
          sd_params=None,
          parse_errors=None,
        )
      },
    ),
    None,  # second call (in _BuildOneGrid) returns None → image not found
  ]
  with pytest.raises(newton.Error, match='not found in DB'):
    list(exp.Grid())


def testExperimentGridBuildOneGridNoFileRaises(tmp_path: pathlib.Path) -> None:
  """_BuildOneGrid raises Error when image file does not exist on disk."""
  ai_db = _MockAIDB()
  nonexistent = tmp_path / 'nonexistent.png'
  png_bytes = _SmallPNG(64, 32)
  png_hash = hashes.Hash256(png_bytes).hex()
  # make a DBImageType pointing to a nonexistent path
  img_obj = _MakeDBImageWithPath(png_hash, nonexistent, w=64, h=32)
  ai_db.Image.return_value = img_obj
  exp = _SimpleCFGExperiment(ai_db, seeds=[42], cfg_values=[70])
  kh = next(iter(exp._k_dict))
  exp._results['42'][kh] = png_hash
  with pytest.raises(newton.Error, match='No file found on disk'):
    list(exp.Grid())


# ─── _MakeTagImage ─────────────────────────────────────────────────────────────


def testMakeTagImageShortText() -> None:
  """_MakeTagImage returns an RGB image for short text."""
  img = newton._MakeTagImage('CFG 7.0', 200, 100)
  assert img.mode == 'RGB'
  assert img.size == (200, 100)


def testMakeTagImageLongText() -> None:
  """_MakeTagImage handles text too long for one line (falls back to multiline)."""
  long_text = 'this is a very long text string that will not fit on a single line at all here'
  img = newton._MakeTagImage(long_text, 200, 100)
  assert img.mode == 'RGB'
  assert img.size == (200, 100)


def testMakeTagImageCustomColors() -> None:
  """_MakeTagImage respects background_color and text_color arguments."""
  img = newton._MakeTagImage('X', 100, 100, background_color='#222222', text_color='yellow')
  assert img.mode == 'RGB'


def testMakeTagImageEmptyText() -> None:
  """_MakeTagImage renders '<empty>' for empty/whitespace-only text."""
  img = newton._MakeTagImage('   ', 100, 100)
  assert img.mode == 'RGB'
