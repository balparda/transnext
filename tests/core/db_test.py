# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: transnext.core.db module."""

from __future__ import annotations

import pathlib
from typing import cast
from unittest import mock

import pytest
from transai.core import ai
from transcrypto.core import key
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
  path: str = '/tmp/model.safetensors',  # noqa: S108
) -> db.AIModelType:
  """Create a test AIModelType."""  # noqa: DOC201
  return db.AIModelType(
    hash=h,
    name=name,
    path=path,
    type='safetensors',
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
  return db.DBImageType(
    hash=img_hash,
    raw_hash='raw-hash',
    path=path,
    alt_path=[],
    size=1024,
    width=512,
    height=512,
    format=db.ImageFormat.PNG.value,
    created_at=1000,
    ai_meta=meta,
    sd_info={},
    sd_params={},
  )


class _MockAPI:
  """Mock API conforming to db.APIProtocol."""

  def __init__(
    self,
    models: list[db.AIModelType] | None = None,
    txt2img_result: tuple[db.DBImageType, bytes] | None = None,
  ) -> None:
    self.models: list[db.AIModelType] = models or []
    self.txt2img_result: tuple[db.DBImageType, bytes] | None = txt2img_result
    self.load_model_calls: list[str] = []

  def GetModels(self) -> list[db.AIModelType]:
    """Return mock models."""  # noqa: DOC201
    return self.models

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
  assert result['version'] == 0
  assert result['images'] == {}
  assert result['models'] == {}
  assert result['known_image_sources'] == []
  assert result['image_output_dir'] is None


def testOverrides() -> None:
  """Factory applies overrides."""
  result: db._DBType = db._DBTypeFactory({'version': 5, 'image_output_dir': '/foo'})
  assert result['version'] == 5
  assert result['image_output_dir'] == '/foo'


# ─── ImageFormat ─────────────────────────────────────────────────────────────


def testImageFormatValues() -> None:
  """ImageFormat enum has correct values."""
  assert db.ImageFormat.JPEG.value == 'JPEG'
  assert db.ImageFormat.PNG.value == 'PNG'
  assert db.ImageFormat.GIF.value == 'GIF'


# ─── AIMetaTypeFactory ───────────────────────────────────────────────────────


def testDefaultsWithRandomSeed() -> None:
  """Factory creates AIMetaType with a random seed by default."""
  result: db.AIMetaType = db.AIMetaTypeFactory()
  assert not result['model_hash']
  assert not result['positive']
  assert result['negative'] is None
  assert 1 < result['seed'] <= ai.AI_MAX_SEED
  assert result['width'] == base.SD_DEFAULT_WIDTH
  assert result['height'] == base.SD_DEFAULT_HEIGHT
  assert result['steps'] == base.SD_DEFAULT_ITERATIONS
  assert result['sampler'] == base.SD_DEFAULT_SAMPLER.value
  assert result['parser'] == base.SD_DEFAULT_QUERY_PARSER.value
  assert result['cfg_scale'] == base.SD_DEFAULT_CFG_SCALE
  assert result['cfg_end'] == base.SD_DEFAULT_CFG_END
  assert result['clip_skip'] == base.SD_DEFAULT_CLIP_SKIP


def testOverridesWithFixedSeed() -> None:
  """Factory applies overrides including a fixed seed."""
  result: db.AIMetaType = db.AIMetaTypeFactory({'seed': 42, 'positive': 'hello'})
  assert result['seed'] == 42
  assert result['positive'] == 'hello'


@pytest.mark.parametrize('seed_val', [None, -1, 0])
def testSpecialSeedValuesGenerateRandom(seed_val: int | None) -> None:
  """Seeds of None, -1, 0 generate a random seed."""
  result: db.AIMetaType = db.AIMetaTypeFactory({'seed': seed_val})
  assert 1 < result['seed'] <= ai.AI_MAX_SEED


def testInvalidSeedRaises() -> None:
  """Invalid seed value raises Error."""
  with pytest.raises(db.Error, match='Invalid seed value'):
    db.AIMetaTypeFactory({'seed': 1})  # must be > 1


def testSeedTooLargeRaises() -> None:
  """Seed exceeding AI_MAX_SEED raises Error."""
  with pytest.raises(db.Error, match='Invalid seed value'):
    db.AIMetaTypeFactory({'seed': ai.AI_MAX_SEED + 1})


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


def testTxt2ImgModelNotInDBRaises(tmp_path: pathlib.Path) -> None:
  """Txt2Img raises Error when model hash not in DB."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  meta: db.AIMetaType = _MakeMeta({'model_hash': 'not-in-db'})
  api = _MockAPI()
  with pytest.raises(db.Error, match='not found in DB models'):
    ai_db.Txt2Img(api, meta)


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
  result_img, result_bytes = ai_db.Txt2Img(mock.MagicMock(), meta)
  assert result_img['hash'] == 'deadbeef'
  assert result_bytes == b'fake-image-data'


def testTxt2ImgExistingImageFileMissing(tmp_path: pathlib.Path) -> None:
  """Txt2Img handles missing file for existing image entry."""
  ai_db: db.AIDatabase = db.AIDatabase(_MakeAppConfig(tmp_path))
  model: db.AIModelType = _MakeModel()
  ai_db._db['models'][model['hash']] = model
  meta: db.AIMetaType = _MakeMeta({'model_hash': model['hash']})
  db_img: db.DBImageType = _MakeDBImage(meta=meta, path='/tmp/nonexistent.png')  # noqa: S108
  ai_db._db['images']['deadbeef'] = db_img
  # set up API to return new image
  new_img: db.DBImageType = _MakeDBImage(meta=meta, img_hash='new-hash')
  api = _MockAPI(txt2img_result=(new_img, b'new-image-data'))
  result_img, result_bytes = ai_db.Txt2Img(api, meta)
  assert result_img['hash'] == 'new-hash'
  assert result_bytes == b'new-image-data'
  # the old entry should have cleared path
  assert ai_db._db['images']['deadbeef']['path'] is None


def testTxt2ImgGenerateNew(tmp_path: pathlib.Path) -> None:
  """Txt2Img generates a new image when not found in DB."""
  ai_db: db.AIDatabase = db.AIDatabase(_MakeAppConfig(tmp_path))
  model: db.AIModelType = _MakeModel()
  ai_db._db['models'][model['hash']] = model
  meta: db.AIMetaType = _MakeMeta({'model_hash': model['hash']})
  new_img: db.DBImageType = _MakeDBImage(meta=meta, img_hash='new-hash')
  api = _MockAPI(txt2img_result=(new_img, b'new-data'))
  result_img, _ = ai_db.Txt2Img(api, meta)
  assert result_img['hash'] == 'new-hash'
  assert 'new-hash' in ai_db._db['images']


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
