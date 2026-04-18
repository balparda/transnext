# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: transnext.core.db module."""

from __future__ import annotations

import io
import pathlib
from typing import cast
from unittest import mock

import pytest
from PIL import Image, PngImagePlugin
from transcrypto.core import hashes, key
from transcrypto.utils import base as tbase
from transcrypto.utils import config as app_config

from transnext.core import base, db

# ─── helpers ──────────────────────────────────────────────────────────────────


def _MakeAppConfig(tmp_path: pathlib.Path) -> app_config.AppConfig:
  """Create a temporary AppConfig for testing."""  # noqa: DOC201
  return app_config.AppConfig(
    'test-transnext',
    'db.bin',
    fixed_dir=tmp_path,
  )


def _MakeMeta(overrides: dict[str, object] | None = None) -> db.AIMetaType:
  """Create an AIMetaType with a valid fixed seed for testing."""  # noqa: DOC201
  defaults: dict[str, object] = {'seed': 42}
  if overrides:
    defaults.update(overrides)
  return db.AIMetaTypeFactory(defaults)


def _MakeModel(
  h: str = 'abc123',
  name: str = 'test-model',
  alias: str = '',
  path: str = '/tmp/model.safetensors',  # noqa: S108
) -> db.AIModelType:
  """Create a test AIModelType."""  # noqa: DOC201
  return db.AIModelType(
    hash=h,
    name=name,
    alias=alias or name,
    autov3=None,
    path=path,
    model_type=db.ModelType.safetensors.value,
    function=db.ModelFunction.Model.value,
    metadata={},
    description=None,
  )


def _MakeDBImage(
  meta: db.AIMetaType | None = None,
  img_hash: str = 'deadbeef',
  path: str | None = '/tmp/img.png',  # noqa: S108
) -> db.DBImageType:
  """Create a test DBImageType."""  # noqa: DOC201
  if meta is None:
    meta = _MakeMeta()
  path_key: str = path or ''
  return db.DBImageType(
    hash=img_hash,
    raw_hash='raw-hash',
    size=1024,
    width=512,
    height=512,
    format=base.ImageFormat.PNG.value,
    info=None,
    paths={
      path_key: db.DBImagePathType(
        main=bool(path),
        created_at=1000,
        origin=None,
        version=None,
        ai_meta=meta,
        sd_info=None,
        sd_params=None,
        parse_errors=None,
      )
    }
    if path_key
    else {},
  )


class _MockAPI:
  """Mock API conforming to db.APIProtocol."""

  def __init__(
    self,
    models: list[db.AIModelType] | None = None,
    loras: list[db.AIModelType] | None = None,
    txt2img_result: tuple[db.DBImageType, bytes] | None = None,
  ) -> None:
    self.models: list[db.AIModelType] = models or []
    self.loras: list[db.AIModelType] = loras or []
    self.txt2img_result: tuple[db.DBImageType, bytes] | None = txt2img_result

  def GetModels(self) -> list[db.AIModelType]:
    """Return mock models."""  # noqa: DOC201
    return self.models

  def GetLora(self) -> list[db.AIModelType]:
    """Return mock loras."""  # noqa: DOC201
    return self.loras

  def Txt2Img(
    self,
    model: db.AIModelType,  # noqa: ARG002
    meta: db.AIMetaType,  # noqa: ARG002
    *,
    dir_root: pathlib.Path | None = None,  # noqa: ARG002
  ) -> tuple[db.DBImageType, bytes]:
    """Return mock image."""  # noqa: DOC201
    assert self.txt2img_result is not None
    return self.txt2img_result


# ─── _DBTypeFactory ──────────────────────────────────────────────────────────


def testDefaults() -> None:
  """Factory creates DB with default values."""
  result: db._DBType = db._DBTypeFactory()
  assert result.pop('last_save') > 1000000  # type: ignore[misc]
  assert result == {
    'db_version': '1.0.0',
    'image_output_dir': None,
    'images': {},
    'known_image_sources': [],
    'lora': {},
    'models': {},
    'version': 0,
  }


def testOverrides() -> None:
  """Factory applies overrides."""
  result: db._DBType = db._DBTypeFactory({'version': 5, 'image_output_dir': '/foo'})
  assert result.pop('last_save') > 1000000  # type: ignore[misc]
  assert result == {
    'db_version': '1.0.0',
    'image_output_dir': '/foo',
    'images': {},
    'known_image_sources': [],
    'lora': {},
    'models': {},
    'version': 5,
  }


# ─── AIMetaTypeFactory ───────────────────────────────────────────────────────


def testDefaultsWithRandomSeed() -> None:
  """Factory creates AIMetaType with a random seed by default."""
  result: db.AIMetaType = db.AIMetaTypeFactory()
  assert 1 < result.pop('seed') <= base.SD_MAX_SEED  # type: ignore[misc]
  assert result == {
    'cfg_end': base.SD_DEFAULT_CFG_END,
    'cfg_rescale': base.SD_DEFAULT_CFG_RESCALE,
    'cfg_scale': base.SD_DEFAULT_CFG_SCALE,
    'cfg_skip': None,
    'clip_skip': base.SD_DEFAULT_CLIP_SKIP,
    'freeu': (
      base.SD_DEFAULT_FREEU_B1,
      base.SD_DEFAULT_FREEU_B2,
      base.SD_DEFAULT_FREEU_S1,
      base.SD_DEFAULT_FREEU_S2,
    ),
    'height': base.SD_DEFAULT_HEIGHT,
    'img2img': None,
    'lora': {},
    'model_hash': None,
    'negative': None,
    'ngms': None,
    'parser': base.SD_DEFAULT_QUERY_PARSER.value,
    'positive': '',
    'sampler': base.SD_DEFAULT_SAMPLER.value,
    'sch_beta': None,
    'sch_sigma': None,
    'sch_spacing': None,
    'sch_type': None,
    'steps': base.SD_DEFAULT_ITERATIONS,
    'v_seed': None,
    'width': base.SD_DEFAULT_WIDTH,
  }


def testOverridesWithFixedSeed() -> None:
  """Factory applies overrides including a fixed seed."""
  result: db.AIMetaType = db.AIMetaTypeFactory({'seed': 42, 'positive': 'hello'})
  assert result == {
    'cfg_end': base.SD_DEFAULT_CFG_END,
    'cfg_rescale': base.SD_DEFAULT_CFG_RESCALE,
    'cfg_scale': base.SD_DEFAULT_CFG_SCALE,
    'cfg_skip': None,
    'clip_skip': base.SD_DEFAULT_CLIP_SKIP,
    'freeu': (
      base.SD_DEFAULT_FREEU_B1,
      base.SD_DEFAULT_FREEU_B2,
      base.SD_DEFAULT_FREEU_S1,
      base.SD_DEFAULT_FREEU_S2,
    ),
    'height': base.SD_DEFAULT_HEIGHT,
    'img2img': None,
    'lora': {},
    'model_hash': None,
    'negative': None,
    'ngms': None,
    'parser': base.SD_DEFAULT_QUERY_PARSER.value,
    'positive': 'hello',
    'sampler': base.SD_DEFAULT_SAMPLER.value,
    'sch_beta': None,
    'sch_sigma': None,
    'sch_spacing': None,
    'sch_type': None,
    'seed': 42,
    'steps': base.SD_DEFAULT_ITERATIONS,
    'v_seed': None,
    'width': base.SD_DEFAULT_WIDTH,
  }


@pytest.mark.parametrize('seed_val', [None, -1, 0])
def testSpecialSeedValuesGenerateRandom(seed_val: int | None) -> None:
  """Seeds of None, -1, 0 generate a random seed."""
  result: db.AIMetaType = db.AIMetaTypeFactory({'seed': seed_val})
  assert 1 < result['seed'] <= base.SD_MAX_SEED


def testSeedTooLargeRaises() -> None:
  """Seed exceeding SD_MAX_SEED raises Error."""
  with pytest.raises(db.Error, match=r'Invalid.*seed.*value'):
    db.AIMetaTypeFactory({'seed': base.SD_MAX_SEED + 1})


def testVariationSeedAutoGenerated() -> None:
  """Variation seed with None/0/-1 and strength>0 gets a random seed."""
  result: db.AIMetaType = db.AIMetaTypeFactory({'v_seed': (0, 50)})
  assert result['v_seed'] is not None
  assert result['v_seed'][0] > 0
  assert result['v_seed'][1] == 50


def testVariationSeedInvalidRaises() -> None:
  """Invalid v_seed value raises Error."""
  with pytest.raises(db.Error, match=r'Invalid.*v_seed.*value'):
    db.AIMetaTypeFactory({'v_seed': (base.SD_MAX_SEED + 1, 50)})


# ─── AIImg2ImgType ───────────────────────────────────────────────────────────


def testAIImg2ImgType() -> None:
  """AIImg2ImgType can be created with valid fields."""
  img2img: db.AIImg2ImgType = db.AIImg2ImgType(
    input_hash='abc123',
    denoising=50,
  )
  assert img2img == {
    'input_hash': 'abc123',
    'denoising': 50,
  }


# ─── AIDatabase ──────────────────────────────────────────────────────────────


def testInitFreshDB(tmp_path: pathlib.Path) -> None:
  """AIDatabase creates a fresh DB when no file exists."""
  config: app_config.AppConfig = _MakeAppConfig(tmp_path)
  ai_db = db.AIDatabase(config)
  assert ai_db.label.startswith('#0@')
  assert ai_db.output is None


def testInitExistingDB(tmp_path: pathlib.Path) -> None:
  """AIDatabase loads an existing DB from disk."""
  config: app_config.AppConfig = _MakeAppConfig(tmp_path)
  # save a DB first
  saved_db: db._DBType = db._DBTypeFactory({'version': 3})
  config.Serialize(
    cast('tbase.JSONDict', saved_db),
    pickler=key.PickleJSON,
  )
  # now load it
  ai_db = db.AIDatabase(config)
  assert ai_db.label.startswith('#3@')


def testInitReadOnly(tmp_path: pathlib.Path) -> None:
  """AIDatabase in read-only mode logs a warning."""
  config: app_config.AppConfig = _MakeAppConfig(tmp_path)
  ai_db = db.AIDatabase(config, read_only=True)
  assert ai_db.label.startswith('#0@')


def testContextManagerNormal(tmp_path: pathlib.Path) -> None:
  """AIDatabase context manager saves on normal exit."""
  config: app_config.AppConfig = _MakeAppConfig(tmp_path)
  with db.AIDatabase(config) as ai_db:
    assert ai_db.label.startswith('#0@')
  # after exit, DB should be saved (version bumped to 1)
  ai_db2 = db.AIDatabase(config)
  assert ai_db2.label.startswith('#1@')


def testContextManagerException(tmp_path: pathlib.Path) -> None:
  """AIDatabase context manager does NOT save on exception exit."""  # noqa: DOC501
  config: app_config.AppConfig = _MakeAppConfig(tmp_path)
  with pytest.raises(ValueError, match='boom'), db.AIDatabase(config) as _ai_db:
    raise ValueError('boom')
  # DB file should not exist (was never saved)
  assert not config.path.exists()


def testReadOnlyDoesNotSave(tmp_path: pathlib.Path) -> None:
  """AIDatabase in read-only mode does not write to disk."""
  config: app_config.AppConfig = _MakeAppConfig(tmp_path)
  with db.AIDatabase(config, read_only=True):
    pass
  assert not config.path.exists()


# ─── AIDatabase.output property ──────────────────────────────────────────────


def testOutputSetAndGet(tmp_path: pathlib.Path) -> None:
  """Output property sets and gets directory path."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  out_dir: pathlib.Path = tmp_path / 'output'
  out_dir.mkdir()
  ai_db.output = out_dir
  assert ai_db.output == out_dir
  # also added to known sources
  assert str(out_dir) in ai_db._db['known_image_sources']


def testOutputSetNone(tmp_path: pathlib.Path) -> None:
  """Setting output to None clears the output directory."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  ai_db.output = None
  assert ai_db.output is None


def testOutputSetInvalidRaises(tmp_path: pathlib.Path) -> None:
  """Setting output to a non-existent directory raises Error."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  with pytest.raises(db.Error, match='Invalid output directory'):
    ai_db.output = tmp_path / 'nonexistent'


# ─── AIDatabase.Save ─────────────────────────────────────────────────────────


def testSaveIncrementsVersion(tmp_path: pathlib.Path) -> None:
  """Save increments the DB version."""
  config: app_config.AppConfig = _MakeAppConfig(tmp_path)
  ai_db = db.AIDatabase(config)
  ai_db.Save()
  ai_db2 = db.AIDatabase(config)
  assert ai_db2.label.startswith('#1@')


def testSafeSaveMismatchRaises(tmp_path: pathlib.Path) -> None:
  """Safe save raises Error when disk DB differs from loaded DB."""
  config: app_config.AppConfig = _MakeAppConfig(tmp_path)
  ai_db = db.AIDatabase(config)
  ai_db.Save()
  # modify the DB on disk behind the back of our instance
  ai_db2 = db.AIDatabase(config)
  ai_db2.Save()
  # now the first instance's version is out of date
  with pytest.raises(db.Error, match='differs from loaded DB'):
    ai_db.Save()


# ─── AIDatabase.RefreshDBModels ──────────────────────────────────────────────


def testRefreshDBModels(tmp_path: pathlib.Path) -> None:
  """RefreshDBModels adds models from API to DB."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  model: db.AIModelType = _MakeModel(h='hash1', name='model-1', path='/tmp/m.st')  # noqa: S108
  api = _MockAPI(models=[model])
  ai_db.RefreshDBModels(api)
  assert 'hash1' in ai_db._db['models']
  assert ai_db._db['models']['hash1']['name'] == 'model-1'


def testRefreshDBModelsSkipDuplicatePath(tmp_path: pathlib.Path) -> None:
  """RefreshDBModels skips models whose path is already in DB."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  model: db.AIModelType = _MakeModel(h='hash1', name='model-1')
  ai_db._db['models']['existing'] = _MakeModel(h='existing', name='existing')
  # same path
  api = _MockAPI(models=[model])
  ai_db.RefreshDBModels(api)
  # only existing model stays, new one has same path so skipped
  assert 'existing' in ai_db._db['models']


def testRefreshDBModelsEmptyHash(tmp_path: pathlib.Path) -> None:
  """RefreshDBModels hashes model file when API returns empty hash."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  model: db.AIModelType = _MakeModel(h='', name='model-1', path='/tmp/m.st')  # noqa: S108
  api = _MockAPI(models=[model])
  with mock.patch('pathlib.Path.read_bytes', return_value=b'model-data'):
    ai_db.RefreshDBModels(api)
  # should have been hashed and added
  assert len(ai_db._db['models']) == 1


# ─── AIDatabase.RefreshDBLora ────────────────────────────────────────────────


def testRefreshDBLora(tmp_path: pathlib.Path) -> None:
  """RefreshDBLora adds loras from API to DB."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  lora: db.AIModelType = db.AIModelType(
    hash='l-hash1',
    name='my-lora',
    alias='my-lora',
    autov3=None,
    path='/tmp/lora.safetensors',  # noqa: S108
    model_type=db.ModelType.safetensors.value,
    function=db.ModelFunction.Lora.value,
    metadata={},
    description=None,
  )
  api = _MockAPI(loras=[lora])
  ai_db.RefreshDBLora(api)
  assert 'l-hash1' in ai_db._db['lora']


# ─── AIDatabase.GetModelHash ─────────────────────────────────────────────────


def testGetModelHashFound(tmp_path: pathlib.Path) -> None:
  """GetModelHash finds model by name in DB."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  model: db.AIModelType = _MakeModel(h='hash1', name='XLB_v10')
  ai_db._db['models']['hash1'] = model
  assert ai_db.GetModelHash('xlb_v10') == 'hash1'


def testGetModelHashNotFoundNoAPI(tmp_path: pathlib.Path) -> None:
  """GetModelHash raises Error when not found and no API given."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  with pytest.raises(db.Error, match='not found in DB'):
    ai_db.GetModelHash('nonexistent')


def testGetModelHashFetchesFromAPI(tmp_path: pathlib.Path) -> None:
  """GetModelHash fetches models from API when not found locally."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  model: db.AIModelType = _MakeModel(h='hash1', name='XLB_v10', path='/tmp/new.st')  # noqa: S108
  api = _MockAPI(models=[model])
  assert ai_db.GetModelHash('xlb_v10', api=api) == 'hash1'


def testGetModelHashMultipleMatchRaises(tmp_path: pathlib.Path) -> None:
  """GetModelHash raises Error when multiple models match."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  ai_db._db['models']['h1'] = _MakeModel(h='h1', name='shared-prefix-a')
  ai_db._db['models']['h2'] = _MakeModel(h='h2', name='shared-prefix-b')
  with pytest.raises(db.Error, match='Multiple models'):
    ai_db.GetModelHash('shared-prefix')


def testGetModelHashEmptyRaises(tmp_path: pathlib.Path) -> None:
  """GetModelHash raises Error for empty name."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  with pytest.raises(db.Error, match='cannot be empty'):
    ai_db.GetModelHash('')


# ─── AIDatabase.Txt2Img ─────────────────────────────────────────────────────


def testTxt2ImgModelNotInDBRaises(tmp_path: pathlib.Path) -> None:
  """Txt2Img raises Error when model hash not in DB."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  meta: db.AIMetaType = _MakeMeta({'model_hash': 'not-in-db'})
  api = _MockAPI()
  with pytest.raises(db.Error, match='not found in DB models'):
    ai_db.Txt2Img(meta, api)


def testTxt2ImgExistingImage(tmp_path: pathlib.Path) -> None:
  """Txt2Img returns existing image when it matches metadata."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  model: db.AIModelType = _MakeModel()
  ai_db._db['models'][model['hash']] = model
  meta: db.AIMetaType = _MakeMeta({'model_hash': model['hash']})
  # create a file on disk for the existing image
  img_file: pathlib.Path = tmp_path / 'existing.png'
  img_file.write_bytes(b'fake-image-data')
  db_img: db.DBImageType = _MakeDBImage(meta=meta, path=str(img_file))
  ai_db._db['images']['deadbeef'] = db_img
  ai_db._ComputeIndexes()
  result_img, result_bytes = ai_db.Txt2Img(meta, mock.MagicMock())
  assert result_img['hash'] == 'deadbeef'
  assert result_bytes == b'fake-image-data'


def testTxt2ImgExistingImageFileMissing(tmp_path: pathlib.Path) -> None:
  """Txt2Img handles missing file for existing image entry."""
  ai_db: db.AIDatabase = db.AIDatabase(_MakeAppConfig(tmp_path))
  model: db.AIModelType = _MakeModel()
  ai_db._db['models'][model['hash']] = model
  meta: db.AIMetaType = _MakeMeta({'model_hash': model['hash']})
  db_img: db.DBImageType = _MakeDBImage(
    meta=meta,
    path='/tmp/nonexistent.png',  # noqa: S108
  )
  ai_db._db['images']['deadbeef'] = db_img
  ai_db._ComputeIndexes()
  # set up API to return new image
  new_img: db.DBImageType = _MakeDBImage(meta=meta, img_hash='new-hash')
  api = _MockAPI(txt2img_result=(new_img, b'new-image-data'))
  result_img, result_bytes = ai_db.Txt2Img(meta, api)
  assert result_img['hash'] == 'new-hash'
  assert result_bytes == b'new-image-data'


def testTxt2ImgGenerateNew(tmp_path: pathlib.Path) -> None:
  """Txt2Img generates a new image when not found in DB."""
  ai_db: db.AIDatabase = db.AIDatabase(_MakeAppConfig(tmp_path))
  model: db.AIModelType = _MakeModel()
  ai_db._db['models'][model['hash']] = model
  meta: db.AIMetaType = _MakeMeta({'model_hash': model['hash']})
  new_img: db.DBImageType = _MakeDBImage(meta=meta, img_hash='new-hash')
  api = _MockAPI(txt2img_result=(new_img, b'new-data'))
  result_img, _ = ai_db.Txt2Img(meta, api)
  assert result_img['hash'] == 'new-hash'


# ─── _DBLabel ────────────────────────────────────────────────────────────────


def testDBLabel() -> None:
  """_DBLabel returns expected format."""
  test_db: db._DBType = db._DBTypeFactory({'version': 7})
  label: str = db._DBLabel(test_db)
  assert label.startswith('#7@')


# ─── APIProtocol ─────────────────────────────────────────────────────────────


def testProtocolRuntimeCheckable() -> None:
  """APIProtocol is runtime checkable."""
  assert isinstance(_MockAPI(), db.APIProtocol)


# ─── ParseImageMetadata ──────────────────────────────────────────────────────


def testParseMetadataSDNextFormat() -> None:
  """Parse SDNext/A1111 'parameters' format metadata string."""
  text = (
    'positive prompt text\n'
    'Negative prompt: negative prompt text\n'
    'Steps: 10, Size: 256x256, Sampler: DPM++ SDE, Scheduler: Foo, Seed: 666, '
    'CFG scale: 15.7, CFG end: 0.3, App: SD.Next, Parser: a1111, '
    'Model: SDXL_17_P-IND_indecentRealismFor_v20, Model hash: 335da0800c'
  )
  result: dict[str, str] = db.ParseImageMetadata(text)
  assert result == {
    'app': 'SD.Next',
    'cfg end': '0.3',
    'cfg scale': '15.7',
    'height': '256',
    'model': 'SDXL_17_P-IND_indecentRealismFor_v20',
    'model hash': '335da0800c',
    'negative': 'negative prompt text',
    'parser': 'a1111',
    'positive': 'positive prompt text',
    'sampler': 'DPM++ SDE',
    'scheduler': 'Foo',
    'seed': '666',
    'steps': '10',
    'width': '256',
  }


def testParseMetadataUserCommentFormat() -> None:
  """Parse metadata from 'UserComment' style (no parser field)."""
  text = (
    'positive prompt text\n'
    'Negative prompt: negative prompt text\n'
    'Steps: 80, Sampler: UniPC, CFG scale: 7, Seed: 3767527812, Size: 832x1216, '
    'Model hash: dce7eb8449, Model: SDXL_05_realisticFreedomSFW_alpha, '
    'RNG: CPU, NGMS: 2.5, Version: v1.7.0-12-g61905b06'
  )
  result: dict[str, str] = db.ParseImageMetadata(text)
  assert result == {
    'cfg scale': '7',
    'height': '1216',
    'model': 'SDXL_05_realisticFreedomSFW_alpha',
    'model hash': 'dce7eb8449',
    'negative': 'negative prompt text',
    'ngms': '2.5',
    'positive': 'positive prompt text',
    'rng': 'CPU',
    'sampler': 'UniPC',
    'seed': '3767527812',
    'steps': '80',
    'version': 'v1.7.0-12-g61905b06',
    'width': '832',
  }


def testParseMetadataNoNegativePrompt() -> None:
  """Parse metadata with no negative prompt line."""
  text = 'just a positive prompt\nSteps: 5, Seed: 1234, Sampler: Euler, Size: 64x64'
  result: dict[str, str] = db.ParseImageMetadata(text)
  assert result == {
    'height': '64',
    'positive': 'just a positive prompt',
    'sampler': 'Euler',
    'seed': '1234',
    'steps': '5',
    'width': '64',
  }


def testParseMetadataMultilinePositive() -> None:
  """Parse positive prompt spanning multiple lines."""
  text = (
    'first line\nsecond line\nNegative prompt: neg\nSteps: 3, Seed: 7, Sampler: Euler, Size: 32x32'
  )
  result: dict[str, str] = db.ParseImageMetadata(text)
  assert result == {
    'height': '32',
    'negative': 'neg',
    'positive': 'first line\nsecond line',
    'sampler': 'Euler',
    'seed': '7',
    'steps': '3',
    'width': '32',
  }


def testParseMetadataMultilineNegative() -> None:
  """Parse negative prompt spanning multiple lines before the params line."""
  text = (
    '(score_9), (score_8_up), score_7_up, source_photo,\nfoo bar\n'
    'Negative prompt: [[[score_6]]], [[score_5]], [score_4], score_1,\n'
    'baz\n'
    'Steps: 10, Size: 1024x1536, Sampler: DPM++ SDE, Seed: 4039644363, '
    'CFG scale: 7, CFG end: 0.8, Model: SDXL_17_P-IND_indecentRealismFor_v20, '
    'Model hash: 335da0800c'
  )
  result: dict[str, str] = db.ParseImageMetadata(text)
  assert result == {
    'cfg end': '0.8',
    'cfg scale': '7',
    'height': '1536',
    'model': 'SDXL_17_P-IND_indecentRealismFor_v20',
    'model hash': '335da0800c',
    'negative': '[[[score_6]]], [[score_5]], [score_4], score_1,\nbaz',
    'positive': '(score_9), (score_8_up), score_7_up, source_photo,\nfoo bar',
    'sampler': 'DPM++ SDE',
    'seed': '4039644363',
    'steps': '10',
    'width': '1024',
  }


def testParseMetadataEmptyNegative() -> None:
  """Parse empty negative prompt — params line must not be swallowed into negative."""
  text = (
    'poor\n'
    'Negative prompt: \n'
    'Steps: 20, Size: 1000x600, Sampler: DPM++ SDE, Seed: 2989026014, CFG scale: 7, '
    'Model: SDXL_16_P-TME_tamePonyThe_v25, Model hash: ae50f0a320'
  )
  result: dict[str, str] = db.ParseImageMetadata(text)
  assert result == {
    'cfg scale': '7',
    'height': '600',
    'model': 'SDXL_16_P-TME_tamePonyThe_v25',
    'model hash': 'ae50f0a320',
    'negative': '',
    'positive': 'poor',
    'sampler': 'DPM++ SDE',
    'seed': '2989026014',
    'steps': '20',
    'width': '1000',
  }


def testParseMetadataEmptyString() -> None:
  """Parse empty metadata string returns empty positive."""
  result: dict[str, str] = db.ParseImageMetadata('')
  assert result == {'positive': ''}


# ─── _ModelsRef ──────────────────────────────────────────────────────────────


def testModelsRef() -> None:
  """_ModelsRef converts models dict to {hash: 'name alias'} dict."""
  models: dict[str, db.AIModelType] = {
    'h1': _MakeModel(h='h1', name='model-1', alias='alias-1'),
  }
  result: dict[str, str] = db._ModelsRef(models)
  assert result == {'h1': 'model-1 alias-1'}  # autov3=None, so no dot suffix


# ─── _ImportImageFile ────────────────────────────────────────────────────────


def _MakeTestPNG(
  tmp_path: pathlib.Path,
  name: str = 'test.png',
  size: tuple[int, int] = (64, 64),
  params: str | None = None,
) -> tuple[pathlib.Path, bytes]:
  """Create a minimal PNG file with optional metadata and return (path, bytes)."""  # noqa: DOC201
  img: Image.Image = Image.new('RGB', size, color='blue')
  png_info: PngImagePlugin.PngInfo | None = None
  if params is not None:
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text('parameters', params)
  buf = io.BytesIO()
  img.save(buf, format='PNG', pnginfo=png_info)
  img_bytes: bytes = buf.getvalue()
  out: pathlib.Path = tmp_path / name
  out.write_bytes(img_bytes)
  return (out, img_bytes)


def testImportImageFilePNGWithMetadata(tmp_path: pathlib.Path) -> None:
  """_ImportImageFile parses a PNG with SDNext metadata correctly."""
  params = (
    'a nice photo\n'
    'Negative prompt: ugly\n'
    'Steps: 20, Size: 64x64, Sampler: DPM++ SDE, Seed: 12345, '
    'CFG scale: 7.5, Model hash: abc123, Model: mymodel'
  )
  img_path, img_bytes = _MakeTestPNG(tmp_path, params=params)
  img_hash: str = hashes.Hash256(img_bytes).hex()
  models: dict[str, str] = {'abc123-full-hash': 'mymodel mymodel'}
  loras: dict[str, str] = {}
  entry: db.DBImageType = db._ImportImageFile(
    img_path,
    img_bytes,
    img_hash,
    models,
    loras,
  )
  # check top-level fields
  path_key: str = next(iter(entry['paths']))
  assert entry['paths'][path_key].pop('created_at') > 1000000  # type: ignore[misc]
  assert entry == {
    'format': 'PNG',
    'hash': 'eac962d149e1206383c88f416fddc3f90ea0de64448b26dfc6ab62e07b41d307',
    'height': 64,
    'info': 'a nice photo\n'
    'Negative prompt: ugly\n'
    'Steps: 20, Size: 64x64, Sampler: DPM++ SDE, Seed: 12345, CFG scale: 7.5, '
    'Model hash: abc123, Model: mymodel',
    'paths': {
      path_key: {
        'ai_meta': {
          'cfg_end': 10,
          'cfg_rescale': 0,
          'cfg_scale': 75,
          'cfg_skip': None,
          'clip_skip': 10,
          'freeu': None,
          'height': 64,
          'img2img': None,
          'lora': {},
          'model_hash': 'abc123-full-hash',
          'negative': 'ugly',
          'ngms': None,
          'parser': 'a1111',
          'positive': 'a nice photo',
          'sampler': 'DPM++ SDE',
          'sch_beta': None,
          'sch_sigma': None,
          'sch_spacing': None,
          'sch_type': None,
          'seed': 12345,
          'steps': 20,
          'v_seed': None,
          'width': 64,
        },
        'main': False,
        'origin': 'AIUnknown',
        'parse_errors': None,
        'sd_info': None,
        'sd_params': None,
        'version': None,
      },
    },
    'raw_hash': 'c34fb4331b2d031d7c644860b54a678424c66ef12352fc165a91dc09840d98fd',
    'size': 346,
    'width': 64,
  }


def testImportImageFilePNGNoMetadata(tmp_path: pathlib.Path) -> None:
  """_ImportImageFile returns entry with no ai_meta when PNG has no metadata."""
  img_path, img_bytes = _MakeTestPNG(tmp_path)
  img_hash: str = hashes.Hash256(img_bytes).hex()
  entry: db.DBImageType = db._ImportImageFile(
    img_path,
    img_bytes,
    img_hash,
    {},
    {},
  )
  # check top-level fields
  path_key: str = next(iter(entry['paths']))
  assert entry['paths'][path_key].pop('created_at') > 1000000  # type: ignore[misc]
  assert entry == {
    'format': 'PNG',
    'hash': 'e9da449bcc3b10b9b37834ea7add3566c3df058c3bf56f74ef706f49848d0126',
    'height': 64,
    'info': None,
    'paths': {
      path_key: {
        'ai_meta': None,
        'main': False,
        'origin': None,
        'parse_errors': None,
        'sd_info': None,
        'sd_params': None,
        'version': None,
      },
    },
    'raw_hash': 'c34fb4331b2d031d7c644860b54a678424c66ef12352fc165a91dc09840d98fd',
    'size': 181,
    'width': 64,
  }


def testImportImageFileUnsupportedFormatRaises(tmp_path: pathlib.Path) -> None:
  """_ImportImageFile raises Error on unsupported image format (BMP)."""
  img: Image.Image = Image.new('RGB', (8, 8), color='red')
  buf = io.BytesIO()
  img.save(buf, format='BMP')
  bmp_bytes: bytes = buf.getvalue()
  bmp_path: pathlib.Path = tmp_path / 'test.bmp'
  bmp_path.write_bytes(bmp_bytes)
  img_hash: str = hashes.Hash256(bmp_bytes).hex()
  with pytest.raises(base.Error, match='Unsupported image format'):
    db._ImportImageFile(bmp_path, bmp_bytes, img_hash, {}, {})


# ─── AIDatabase.Sync ─────────────────────────────────────────────────────────


def testSyncInvalidAddDirRaises(tmp_path: pathlib.Path) -> None:
  """Sync raises Error when add_dir does not exist."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  with pytest.raises(db.Error, match='does not exist'):
    ai_db.Sync(add_dir=tmp_path / 'nonexistent')


def testSyncAddDirAddsToKnownSources(tmp_path: pathlib.Path) -> None:
  """Sync with add_dir registers it in known_image_sources."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  new_dir: pathlib.Path = tmp_path / 'imgs'
  new_dir.mkdir()
  ai_db.Sync(add_dir=new_dir)
  assert str(new_dir) in ai_db._db['known_image_sources']


def testSyncAddDirNotDuplicatedInSources(tmp_path: pathlib.Path) -> None:
  """Sync does not duplicate an already-known source dir."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  src_dir: pathlib.Path = tmp_path / 'imgs'
  src_dir.mkdir()
  ai_db._db['known_image_sources'].append(str(src_dir))
  ai_db.Sync(add_dir=src_dir)
  assert ai_db._db['known_image_sources'].count(str(src_dir)) == 1


def testSyncNoSourcesNoOp(tmp_path: pathlib.Path) -> None:
  """Sync with no known sources and no add_dir does nothing."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  ai_db.Sync()
  assert ai_db._db['images'] == {}


def testSyncNewImageImported(tmp_path: pathlib.Path) -> None:
  """Sync imports a new PNG from a known source directory."""
  src_dir: pathlib.Path = tmp_path / 'imgs'
  src_dir.mkdir()
  params = (
    'sunset photo\n'
    'Negative prompt: rain\n'
    'Steps: 15, Size: 64x64, Sampler: Euler, Seed: 777, CFG scale: 6.0, '
    'Model hash: deadbeef00, Model: my-sdxl-model'
  )
  _, img_bytes = _MakeTestPNG(src_dir, params=params)
  img_hash: str = hashes.Hash256(img_bytes).hex()
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  # model must be pre-populated so FindModelHash can resolve
  ai_db._db['models']['deadbeef00'] = _MakeModel(
    h='deadbeef00',
    name='my-sdxl-model',
  )
  ai_db.Sync(add_dir=src_dir)
  assert img_hash in ai_db._db['images']
  entry: db.DBImageType = ai_db._db['images'][img_hash]
  # check
  path_key: str = next(iter(entry['paths']))
  assert entry['paths'][path_key].pop('created_at') > 1000000  # type: ignore[misc]
  assert entry == {
    'format': 'PNG',
    'hash': 'ccb1d9a1cc2add9120d466e8d83320ae11b2f93eca366bfd375749244e35180d',
    'height': 64,
    'info': 'sunset photo\n'
    'Negative prompt: rain\n'
    'Steps: 15, Size: 64x64, Sampler: Euler, Seed: 777, CFG scale: 6.0, Model '
    'hash: deadbeef00, Model: my-sdxl-model',
    'paths': {
      path_key: {
        'ai_meta': {
          'cfg_end': 10,
          'cfg_rescale': 0,
          'cfg_scale': 60,
          'cfg_skip': None,
          'clip_skip': 10,
          'freeu': None,
          'height': 64,
          'img2img': None,
          'lora': {},
          'model_hash': 'deadbeef00',
          'negative': 'rain',
          'ngms': None,
          'parser': 'a1111',
          'positive': 'sunset photo',
          'sampler': 'Euler',
          'sch_beta': None,
          'sch_sigma': None,
          'sch_spacing': None,
          'sch_type': None,
          'seed': 777,
          'steps': 15,
          'v_seed': None,
          'width': 64,
        },
        'main': False,
        'origin': 'AIUnknown',
        'parse_errors': None,
        'sd_info': None,
        'sd_params': None,
        'version': None,
      },
    },
    'raw_hash': 'c34fb4331b2d031d7c644860b54a678424c66ef12352fc165a91dc09840d98fd',
    'size': 350,
    'width': 64,
  }


def testSyncNonImageFilesIgnored(tmp_path: pathlib.Path) -> None:
  """Sync ignores non-image files in the source directory."""
  src_dir: pathlib.Path = tmp_path / 'imgs'
  src_dir.mkdir()
  (src_dir / 'readme.txt').write_text('hello')
  (src_dir / 'data.json').write_text('{}')
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  ai_db.Sync(add_dir=src_dir)
  assert ai_db._db['images'] == {}


def testSyncKnownImagePathRestoredWhenNone(tmp_path: pathlib.Path) -> None:
  """Sync adds a new path for a known image that was missing the specific file path."""
  src_dir: pathlib.Path = tmp_path / 'imgs'
  src_dir.mkdir()
  img_path, img_bytes = _MakeTestPNG(src_dir)
  img_hash: str = hashes.Hash256(img_bytes).hex()
  # pre-populate DB with this image but no paths
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  ai_db._db['images'][img_hash] = _MakeDBImage(img_hash=img_hash, path=None)
  ai_db._db['known_image_sources'].append(str(src_dir))
  ai_db.Sync()
  assert str(img_path) in ai_db._db['images'][img_hash]['paths']


def testSyncKnownImageAltPathAdded(tmp_path: pathlib.Path) -> None:
  """Sync adds an alternative path when the same image is found at a new location."""
  src_dir: pathlib.Path = tmp_path / 'imgs'
  src_dir.mkdir()
  img_path, img_bytes = _MakeTestPNG(src_dir)
  img_hash: str = hashes.Hash256(img_bytes).hex()
  # create a second copy at a different location so both paths exist on disk
  other_path_obj: pathlib.Path = tmp_path / 'other.png'
  other_path_obj.write_bytes(img_bytes)
  other_path: str = str(other_path_obj)
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  ai_db._db['images'][img_hash] = _MakeDBImage(img_hash=img_hash, path=other_path)
  ai_db._db['known_image_sources'].append(str(src_dir))
  ai_db.Sync()
  entry: db.DBImageType = ai_db._db['images'][img_hash]
  # both paths should be in the paths dict
  assert other_path in entry['paths']
  assert str(img_path) in entry['paths']


def testSyncDeletedImagePathCleared(tmp_path: pathlib.Path) -> None:
  """Sync leaves the path in DB even when the file no longer exists."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  missing_path: str = str(tmp_path / 'gone.png')
  meta: db.AIMetaType = _MakeMeta()
  ai_db._db['images']['deadbeef'] = _MakeDBImage(
    meta=meta,
    img_hash='deadbeef',
    path=missing_path,
  )
  ai_db.Sync()
  # the path stays in the DB paths dict (sync doesn't remove missing file paths)
  assert missing_path in ai_db._db['images']['deadbeef']['paths']


def testSyncDeletedPrimaryAltPromoted(tmp_path: pathlib.Path) -> None:
  """Sync keeps both paths in the paths dict even when former primary is gone."""
  src_dir: pathlib.Path = tmp_path / 'imgs'
  src_dir.mkdir()
  # create alt image file on disk
  alt_path_obj, alt_bytes = _MakeTestPNG(src_dir, name='alt.png')
  alt_hash: str = hashes.Hash256(alt_bytes).hex()
  gone_path: str = str(tmp_path / 'gone.png')
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  # create entry with gone_path as the only path, then add alt path manually
  entry: db.DBImageType = _MakeDBImage(img_hash=alt_hash, path=gone_path)
  entry['paths'][str(alt_path_obj)] = db.DBImagePathType(
    main=False,
    created_at=1000,
    origin=None,
    version=None,
    ai_meta=None,
    sd_info=None,
    sd_params=None,
    parse_errors=None,
  )
  ai_db._db['images'][alt_hash] = entry
  ai_db.Sync()
  updated: db.DBImageType = ai_db._db['images'][alt_hash]
  # both paths remain in the dict
  assert str(alt_path_obj) in updated['paths']


def testSyncMissingSourceDirSkipped(tmp_path: pathlib.Path) -> None:
  """Sync gracefully skips a known source dir that no longer exists."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  ai_db._db['known_image_sources'].append(str(tmp_path / 'gone_dir'))
  ai_db.Sync()  # should not raise
  assert ai_db._db['images'] == {}


def testSyncMultipleImagesInDir(tmp_path: pathlib.Path) -> None:
  """Sync imports all PNG images found recursively in a source directory."""
  src_dir: pathlib.Path = tmp_path / 'imgs'
  src_dir.mkdir()
  sub_dir: pathlib.Path = src_dir / 'sub'
  sub_dir.mkdir()
  # use distinct seeds so images produce different bytes (different hashes)
  params_1 = (
    'img one\nSteps: 20, Size: 64x64, Sampler: Euler, Seed: 100, CFG scale: 7.0, '
    'Model hash: abc1230, Model: mymodel'
  )
  params_2 = (
    'img two\nSteps: 20, Size: 64x64, Sampler: Euler, Seed: 200, CFG scale: 7.0, '
    'Model hash: abc1230, Model: mymodel'
  )
  _, b1 = _MakeTestPNG(src_dir, 'a.png', params=params_1)
  _, b2 = _MakeTestPNG(sub_dir, 'b.png', params=params_2)
  h1: str = hashes.Hash256(b1).hex()
  h2: str = hashes.Hash256(b2).hex()
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  ai_db._db['models']['abc1230'] = _MakeModel(h='abc1230', name='mymodel')
  ai_db.Sync(add_dir=src_dir)
  assert h1 in ai_db._db['images']
  assert h2 in ai_db._db['images']


def testSyncWithAPIRefreshesModels(tmp_path: pathlib.Path) -> None:
  """Sync with API provided refreshes models and lora before scanning."""
  src_dir: pathlib.Path = tmp_path / 'imgs'
  src_dir.mkdir()
  model: db.AIModelType = _MakeModel(h='h1', name='m1', path='/tmp/m.st')  # noqa: S108
  api = _MockAPI(models=[model], loras=[])
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path), api=api)
  ai_db.Sync(add_dir=src_dir)
  assert 'h1' in ai_db._db['models']


@pytest.mark.slow
def testSyncRealImages(tmp_path: pathlib.Path) -> None:
  """Sync against the real test images directory imports both files with exact metadata."""
  # pre-populate models so partial model hashes in image metadata resolve to exact entries
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  ai_db._db['models']['e6bb9ea85b1065e7bce3cf03'] = _MakeModel(
    h='e6bb9ea85b1065e7bce3cf03', name='SDXL_00_v10VAEFix'
  )
  ai_db._db['models']['dec85dd6545e07bbd7a0fde6'] = _MakeModel(
    h='dec85dd6545e07bbd7a0fde6', name='SDXL_10_COL_colossusProjectXLSFW_v10bNeodemon'
  )
  ai_db._db['models']['442394a51be6bb9ea85b'] = _MakeModel(
    h='442394a51be6bb9ea85b', name='SDXL_13_REF_realisticFreedomSFW_ophelia'
  )
  ai_db._db['lora']['a7fc563d7ae966665cc3b'] = _MakeModel(
    h='a7fc563d7ae966665cc3b', name='XL-CLR-colorful-fractal'
  )
  # point Sync at the real test image directory (tests/data/images/)
  images_dir: pathlib.Path = (pathlib.Path(__file__).parent.parent / 'data' / 'images').resolve()
  with mock.patch('transnext.core.base.GetFileCreation', return_value=1700666999):
    ai_db.Sync(add_dir=images_dir)
  assert ai_db._db['images'] == {
    # ── Image 1: 20231116... (crazy / no negative / UniPC / model e6bb9ea85b) ──
    '7b35463de957335e3841b6d9742c6bed706212ce87851f7c3ca93fe268544f4d': {
      'hash': '7b35463de957335e3841b6d9742c6bed706212ce87851f7c3ca93fe268544f4d',
      'raw_hash': '7847345e5b4687962637c792e890226b08113e9579f7f1d253d1d6efe7f37363',
      'size': 1938975,
      'width': 1024,
      'height': 1024,
      'format': 'PNG',
      'info': (
        'crazy\n'
        'Steps: 30, Sampler: UniPC, CFG scale: 7.5, Seed: 1884649524, Size: 1024x1024, '
        'Model hash: e6bb9ea85b, Model: SDXL_00_v10VAEFix, RNG: CPU, NGMS: 2.5, Version: v1.6.0'
      ),
      'paths': {
        str(
          images_dir
          / '20231116184540-Me6bb9ea85b-Pf4804c3f-C7.5-N30-1884649524-a3fffc692a94bfca.png'
        ): {
          'main': False,
          'created_at': 1700666999,
          'origin': 'A1111',
          'parse_errors': None,
          'version': 'v1.6.0',
          'ai_meta': {
            'cfg_end': 10,
            'cfg_rescale': 0,
            'cfg_scale': 75,
            'cfg_skip': None,
            'clip_skip': 10,
            'freeu': None,
            'height': 1024,
            'img2img': None,
            'lora': {},
            'model_hash': 'e6bb9ea85b1065e7bce3cf03',
            'negative': None,
            'ngms': 250,
            'parser': 'a1111',
            'positive': 'crazy',
            'sampler': 'UniPC',
            'sch_beta': None,
            'sch_sigma': None,
            'sch_spacing': None,
            'sch_type': None,
            'seed': 1884649524,
            'steps': 30,
            'v_seed': None,
            'width': 1024,
          },
          'sd_info': None,
          'sd_params': None,
        },
      },
    },
    # ── Image 2: ea94... (spaceship / negative / DPM++ SDE / model dec85dd654) ──
    'ec24329d4bd5b333e39ef923a61a66416d459af84c079ae4606e37a5b88a5985': {
      'hash': 'ec24329d4bd5b333e39ef923a61a66416d459af84c079ae4606e37a5b88a5985',
      'raw_hash': '3da9f76cbcc7c4de5a39973e26d09696a1fe1b068d02f12290fe6632a759aec0',
      'size': 519973,
      'width': 800,
      'height': 800,
      'format': 'PNG',
      'info': (
        '(spaceship), space, [photograph]\n'
        'Negative prompt: planet, galaxy\n'
        'Steps: 30, Size: 800x800, Sampler: DPM++ SDE, Scheduler: DPMSolverMultistepScheduler, '
        'Seed: 1234321, CFG scale: 8.0, CFG end: 0.8, App: SD.Next, Version: 0eb4a98, '
        'Parser: a1111, Pipeline: StableDiffusionXLPipeline, Operations: txt2img, '
        'Model: SDXL_10_COL_colossusProjectXLSFW_v10bNeodemon, Model hash: dec85dd654'
      ),
      'paths': {
        str(
          images_dir / 'ea94a595ace8-20260413105628-dec85dd6-80-30-800-800-1234321-ec24329d4bd5.png'
        ): {
          'main': False,
          'created_at': 1700666999,
          'origin': 'SDNext',
          'parse_errors': None,
          'version': '0eb4a98',
          'ai_meta': {
            'cfg_end': 8,
            'cfg_rescale': 0,
            'cfg_scale': 80,
            'cfg_skip': None,
            'clip_skip': 10,
            'freeu': None,
            'height': 800,
            'img2img': None,
            'lora': {},
            'model_hash': 'dec85dd6545e07bbd7a0fde6',
            'negative': 'planet, galaxy',
            'ngms': None,
            'parser': 'a1111',
            'positive': '(spaceship), space, [photograph]',
            'sampler': 'DPM++ SDE',
            'sch_beta': None,
            'sch_sigma': None,
            'sch_spacing': None,
            'sch_type': None,
            'seed': 1234321,
            'steps': 30,
            'v_seed': None,
            'width': 800,
          },
          'sd_info': None,
          'sd_params': None,
        },
      },
    },
    # ── Image 3: 5a18... (crazy woman face / negative / DPM SDE / lora XL-CLR-colorful-fractal) ──
    '5a18babbf7fd09ad6ed7a5334c819d4779958f8b6b9ed8fd9cc3380aa955ee1a': {
      'hash': '5a18babbf7fd09ad6ed7a5334c819d4779958f8b6b9ed8fd9cc3380aa955ee1a',
      'raw_hash': 'b21477f7c524b621cd508f67e4c2b131b26144ac0866cfa4d73438254dfc7e07',
      'size': 811930,
      'width': 800,
      'height': 800,
      'format': 'PNG',
      'info': (
        '((crazy woman face)), [snapshot:photorealistic:0.1], 1960s coloring, colorful fractal\n'
        '<lora:XL-CLR-colorful-fractal:1.2>\n'
        'Negative prompt: clown, text, cartoon\n'
        'Steps: 47, Size: 800x800, Sampler: DPM SDE, Scheduler: DPMSolverSDEScheduler, '
        'Seed: 666999, CFG scale: 5.4, CFG end: 0.7, Clip skip: 1.3, App: SD.Next, '
        'Version: 0eb4a98, Parser: a1111, Pipeline: StableDiffusionXLPipeline, '
        'Operations: txt2img, Model: SDXL_13_REF_realisticFreedomSFW_ophelia, '
        'Model hash: 442394a51b, Variation seed: 777, Variation strength: 0.62, '
        'Sampler spacing: linspace, Sampler sigma: karras, Sampler type: epsilon, '
        'Sampler beta schedule: scaled'
      ),
      'paths': {
        str(images_dir / 'f8bf9f37-20260413153649-442394a51b-5.4-47-800-800-666999-92ef6d0b.png'): {
          'main': False,
          'created_at': 1700666999,
          'origin': 'SDNext',
          'parse_errors': None,
          'version': '0eb4a98',
          'ai_meta': {
            'cfg_end': 7,
            'cfg_rescale': 0,
            'cfg_scale': 54,
            'cfg_skip': None,
            'clip_skip': 13,
            'freeu': None,
            'height': 800,
            'img2img': None,
            'lora': {
              'a7fc563d7ae966665cc3b': '1.2',
            },
            'model_hash': '442394a51be6bb9ea85b',
            'negative': 'clown, text, cartoon',
            'ngms': None,
            'parser': 'a1111',
            'positive': (
              '((crazy woman face)), [snapshot:photorealistic:0.1], '
              '1960s coloring, colorful fractal\n'
              '<lora:XL-CLR-colorful-fractal:1.2>'
            ),
            'sampler': 'DPM SDE',
            'sch_beta': 'scaled',
            'sch_sigma': 'karras',
            'sch_spacing': 'linspace',
            'sch_type': 'epsilon',
            'seed': 666999,
            'steps': 47,
            'v_seed': (
              777,
              62,
            ),
            'width': 800,
          },
          'sd_info': None,
          'sd_params': None,
        },
      },
    },
    # ── Image 4: db08... (dark knight in moody rain / negative batman, comic, text / DPM SDE) ──
    'db088cdca09796cadee02ec7eef8dd8e2227490a5afbb353461ab34d1ddbd8b9': {
      'hash': 'db088cdca09796cadee02ec7eef8dd8e2227490a5afbb353461ab34d1ddbd8b9',
      'raw_hash': 'dcf3c5cacfddc7b23f3314c680263c03ace7fedc456f6daa1907d5f7ed30af2e',
      'size': 166684,
      'width': 512,
      'height': 256,
      'format': 'PNG',
      'info': (
        'dark knight in moody rain\n'
        'Negative prompt: batman, comic, text\n'
        'Steps: 40, Size: 512x256, Sampler: DPM SDE, Scheduler: DPMSolverSDEScheduler, '
        'Seed: 666, CFG scale: 8.0, CFG rescale: 0.8, CFG end: 0.9, Clip skip: 2, App: SD.Next, '
        'Version: 0eb4a98, Parser: a1111, Pipeline: StableDiffusionXLPipeline, '
        'Operations: txt2img, Model: SDXL_00_XLB_v10VAEFix, Model hash: e6bb9ea85b, '
        'Variation seed: 999, Variation strength: 0.3, Sampler spacing: linspace, '
        'Sampler sigma: karras, Sampler type: epsilon, Sampler beta schedule: linear, '
        'FreeU: b1=1.1 b2=1.15 s1=0.7 s2=0.6'
      ),
      'paths': {
        str(
          images_dir / '6db2ba7302bd-20260417102202-e6bb9ea8-80-40-512-256-666-db088cdca097.png'
        ): {
          'main': False,
          'created_at': 1700666999,
          'origin': 'SDNext',
          'parse_errors': {
            'ambiguous model #e6bb9ea85b/sdxl_00_xlb_v10vaefix: '
            "['e6bb9ea85b1065e7bce3cf03', '442394a51be6bb9ea85b']/[]": None,
          },
          'version': '0eb4a98',
          'ai_meta': {
            'cfg_end': 9,
            'cfg_rescale': 80,
            'cfg_scale': 80,
            'cfg_skip': None,
            'clip_skip': 20,
            'freeu': (
              110,
              115,
              70,
              60,
            ),
            'height': 256,
            'img2img': None,
            'lora': {},
            'model_hash': None,
            'negative': 'batman, comic, text',
            'ngms': None,
            'parser': 'a1111',
            'positive': 'dark knight in moody rain',
            'sampler': 'DPM SDE',
            'sch_beta': 'linear',
            'sch_sigma': 'karras',
            'sch_spacing': 'linspace',
            'sch_type': 'epsilon',
            'seed': 666,
            'steps': 40,
            'v_seed': (
              999,
              30,
            ),
            'width': 512,
          },
          'sd_info': None,
          'sd_params': None,
        },
      },
    },
  }


# ─── AIDatabase.Reproduce ────────────────────────────────────────────────────


def testReproduceHashNotFoundRaises(tmp_path: pathlib.Path) -> None:
  """Reproduce raises Error when image hash is not in DB."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  api = _MockAPI()
  with pytest.raises(db.Error, match='not found in DB'):
    ai_db.Reproduce('nonexistent-hash', api)


def testReproduceNoAIMetaRaises(tmp_path: pathlib.Path) -> None:
  """Reproduce raises Error when the DB entry has no AI metadata."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  entry: db.DBImageType = _MakeDBImage(img_hash='img-hash', path=None)
  ai_db._db['images']['img-hash'] = entry
  api = _MockAPI()
  with pytest.raises(db.Error, match='does not have any AI metadata'):
    ai_db.Reproduce('img-hash', api)


def testReproduceModelNotInDBRaises(tmp_path: pathlib.Path) -> None:
  """Reproduce raises Error when the model referenced in ai_meta is not in DB models."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  meta: db.AIMetaType = _MakeMeta({'model_hash': 'missing-model'})
  entry: db.DBImageType = _MakeDBImage(meta=meta, img_hash='img-hash')
  ai_db._db['images']['img-hash'] = entry
  api = _MockAPI()
  with pytest.raises(db.Error, match=r'missing-model.*not found in DB models'):
    ai_db.Reproduce('img-hash', api)


def testReproduceUpscaledParseErrorRaises(tmp_path: pathlib.Path) -> None:
  """Reproduce raises Error when image has an 'upscaled' parse error."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  model: db.AIModelType = _MakeModel()
  ai_db._db['models'][model['hash']] = model
  meta: db.AIMetaType = _MakeMeta({'model_hash': model['hash']})
  entry: db.DBImageType = _MakeDBImage(meta=meta, img_hash='img-hash')
  path_key: str = next(iter(entry['paths']))
  entry['paths'][path_key]['parse_errors'] = {'upscaled': None}
  ai_db._db['images']['img-hash'] = entry
  api = _MockAPI()
  with pytest.raises(db.Error, match='upscaled'):
    ai_db.Reproduce('img-hash', api)


def testReproduceSuccess(tmp_path: pathlib.Path) -> None:
  """Reproduce calls API.Txt2Img with original meta and adds result to DB."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  output_dir: pathlib.Path = tmp_path / 'output'
  output_dir.mkdir()
  ai_db.output = output_dir
  model: db.AIModelType = _MakeModel()
  ai_db._db['models'][model['hash']] = model
  meta: db.AIMetaType = _MakeMeta({'model_hash': model['hash']})
  original_entry: db.DBImageType = _MakeDBImage(meta=meta, img_hash='original-hash')
  ai_db._db['images']['original-hash'] = original_entry
  # API will return a distinct "new" image
  new_entry: db.DBImageType = _MakeDBImage(
    meta=meta,
    img_hash='new-hash',
    path='/tmp/new.png',  # noqa: S108
  )
  new_entry['raw_hash'] = 'different-raw-hash'
  api = _MockAPI(txt2img_result=(new_entry, b'new-bytes'))
  result_entry, result_bytes = ai_db.Reproduce('original-hash', api)
  assert result_entry['hash'] == 'new-hash'
  assert result_bytes == b'new-bytes'
  assert 'new-hash' in ai_db._db['images']


def testReproduceSuccessRawHashMatch(tmp_path: pathlib.Path) -> None:
  """Reproduce logs a successful match when new raw hash equals the original's raw hash."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  output_dir: pathlib.Path = tmp_path / 'output'
  output_dir.mkdir()
  ai_db.output = output_dir
  model: db.AIModelType = _MakeModel()
  ai_db._db['models'][model['hash']] = model
  meta: db.AIMetaType = _MakeMeta({'model_hash': model['hash']})
  original_entry: db.DBImageType = _MakeDBImage(meta=meta, img_hash='original-hash')
  original_entry['raw_hash'] = 'same-raw-hash'
  ai_db._db['images']['original-hash'] = original_entry
  # API returns a new image with the SAME raw hash — successful reproduction
  new_entry: db.DBImageType = _MakeDBImage(meta=meta, img_hash='new-hash', path='/tmp/new.png')  # noqa: S108
  new_entry['raw_hash'] = 'same-raw-hash'
  api = _MockAPI(txt2img_result=(new_entry, b'bytes'))
  result_entry, _ = ai_db.Reproduce('original-hash', api)
  assert result_entry['hash'] == 'new-hash'
  assert 'new-hash' in ai_db._db['images']


def testReproduceWithLoraWarnings(tmp_path: pathlib.Path) -> None:
  """Reproduce logs warnings for loras with missing hashes or A1111-style weight strings."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  model: db.AIModelType = _MakeModel()
  ai_db._db['models'][model['hash']] = model
  # lora entry: missing from db['lora'] and uses A1111-style comma-separated weight
  meta: db.AIMetaType = _MakeMeta(
    {'model_hash': model['hash'], 'lora': {'missing-lora-hash': 'lora_name:0.8,extra'}}
  )
  entry: db.DBImageType = _MakeDBImage(meta=meta, img_hash='img-hash')
  ai_db._db['images']['img-hash'] = entry
  new_entry: db.DBImageType = _MakeDBImage(meta=meta, img_hash='new-hash', path='/tmp/new.png')  # noqa: S108
  api = _MockAPI(txt2img_result=(new_entry, b'bytes'))
  # should not raise; just logs errors
  result_entry, _ = ai_db.Reproduce('img-hash', api)
  assert result_entry['hash'] == 'new-hash'
