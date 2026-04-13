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
from transai.core import ai
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


# ─── _ParseImageMetadata ─────────────────────────────────────────────────────


def testParseMetadataSDNextFormat() -> None:
  """Parse SDNext/A1111 'parameters' format metadata string."""
  text = (
    'positive prompt text\n'
    'Negative prompt: negative prompt text\n'
    'Steps: 10, Size: 256x256, Sampler: DPM++ SDE, Scheduler: Foo, Seed: 666, '
    'CFG scale: 15.7, CFG end: 0.3, App: SD.Next, Parser: a1111, '
    'Model: SDXL_17_P-IND_indecentRealismFor_v20, Model hash: 335da0800c'
  )
  result: dict[str, str] = db._ParseImageMetadata(text)
  assert result['positive'] == 'positive prompt text'
  assert result['negative'] == 'negative prompt text'
  assert result['steps'] == '10'
  assert result['width'] == '256'
  assert result['height'] == '256'
  assert result['sampler'] == 'DPM++ SDE'
  assert result['seed'] == '666'
  assert result['cfg scale'] == '15.7'
  assert result['cfg end'] == '0.3'
  assert result['parser'] == 'a1111'
  assert result['model hash'] == '335da0800c'
  assert result['model'] == 'SDXL_17_P-IND_indecentRealismFor_v20'


def testParseMetadataUserCommentFormat() -> None:
  """Parse metadata from 'UserComment' style (no parser field)."""
  text = (
    'positive prompt text\n'
    'Negative prompt: negative prompt text\n'
    'Steps: 80, Sampler: UniPC, CFG scale: 7, Seed: 3767527812, Size: 832x1216, '
    'Model hash: dce7eb8449, Model: SDXL_05_realisticFreedomSFW_alpha, '
    'RNG: CPU, NGMS: 2.5, Version: v1.7.0-12-g61905b06'
  )
  result: dict[str, str] = db._ParseImageMetadata(text)
  assert result['positive'] == 'positive prompt text'
  assert result['negative'] == 'negative prompt text'
  assert result['steps'] == '80'
  assert result['sampler'] == 'UniPC'
  assert result['cfg scale'] == '7'
  assert result['seed'] == '3767527812'
  assert result['width'] == '832'
  assert result['height'] == '1216'
  assert result['model hash'] == 'dce7eb8449'
  assert result['model'] == 'SDXL_05_realisticFreedomSFW_alpha'


def testParseMetadataNoNegativePrompt() -> None:
  """Parse metadata with no negative prompt line."""
  text = 'just a positive prompt\nSteps: 5, Seed: 1234, Sampler: Euler, Size: 64x64'
  result: dict[str, str] = db._ParseImageMetadata(text)
  assert result['positive'] == 'just a positive prompt'
  assert 'negative' not in result
  assert result['steps'] == '5'
  assert result['seed'] == '1234'
  assert result['width'] == '64'
  assert result['height'] == '64'


def testParseMetadataMultilinePositive() -> None:
  """Parse positive prompt spanning multiple lines."""
  text = (
    'first line\nsecond line\nNegative prompt: neg\nSteps: 3, Seed: 7, Sampler: Euler, Size: 32x32'
  )
  result: dict[str, str] = db._ParseImageMetadata(text)
  assert result['positive'] == 'first line\nsecond line'
  assert result['negative'] == 'neg'


def testParseMetadataEmptyString() -> None:
  """Parse empty metadata string returns empty positive."""
  result: dict[str, str] = db._ParseImageMetadata('')
  assert not result['positive']
  assert 'negative' not in result


# ─── _FindModelHash ──────────────────────────────────────────────────────────


def testFindModelHashExactMatch() -> None:
  """Exact hash match in models returns the hash unchanged."""
  models: set[str] = {'abc123-full-hash'}
  assert db._FindModelHash('abc123-full-hash', models) == 'abc123-full-hash'


def testFindModelHashPrefixMatch() -> None:
  """Prefix match returns the full model hash."""
  models: set[str] = {'abc123-full-hash'}
  assert db._FindModelHash('abc123', models) == 'abc123-full-hash'


def testFindModelHashNoMatch() -> None:
  """No match returns the partial hash as-is."""
  models: set[str] = {'abc123-full-hash'}
  assert db._FindModelHash('deadbeef', models) == 'deadbeef'


def testFindModelHashAmbiguous() -> None:
  """Ambiguous prefix returns the partial hash as-is."""
  models: set[str] = {'abc123aaa', 'abc123bbb'}
  assert db._FindModelHash('abc123', models) == 'abc123'


def testFindModelHashEmpty() -> None:
  """Empty hash returns empty string."""
  assert not db._FindModelHash('', set())


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
  models: set[str] = {'abc123-full-hash'}
  entry: db.DBImageType = db._ImportImageFile(img_path, img_bytes, img_hash, models)
  assert entry['hash'] == img_hash
  assert entry['path'] == str(img_path)
  assert entry['width'] == 64
  assert entry['height'] == 64
  assert entry['format'] == db.ImageFormat.PNG.value
  assert entry['size'] == len(img_bytes)
  assert entry['ai_meta']['positive'] == 'a nice photo'
  assert entry['ai_meta']['negative'] == 'ugly'
  assert entry['ai_meta']['steps'] == 20
  assert entry['ai_meta']['seed'] == 12345
  assert entry['ai_meta']['cfg_scale'] == 75  # 7.5 x 10
  assert entry['ai_meta']['model_hash'] == 'abc123-full-hash'  # prefix resolved


def testImportImageFilePNGNoMetadata(tmp_path: pathlib.Path) -> None:
  """_ImportImageFile handles PNG with no embedded metadata (uses defaults)."""
  img_path, img_bytes = _MakeTestPNG(tmp_path)
  img_hash: str = hashes.Hash256(img_bytes).hex()
  entry: db.DBImageType = db._ImportImageFile(img_path, img_bytes, img_hash, set())
  assert entry['hash'] == img_hash
  assert not entry['ai_meta']['positive']
  assert entry['ai_meta']['negative'] is None


def testImportImageFileUnsupportedFormatRaises(tmp_path: pathlib.Path) -> None:
  """_ImportImageFile raises Error on unsupported image format (GIF not supported)."""
  # create a BMP image (not in our supported output formats map)
  img: Image.Image = Image.new('RGB', (8, 8), color='red')
  buf = io.BytesIO()
  img.save(buf, format='BMP')
  bmp_bytes: bytes = buf.getvalue()
  bmp_path: pathlib.Path = tmp_path / 'test.bmp'
  bmp_path.write_bytes(bmp_bytes)
  img_hash: str = hashes.Hash256(bmp_bytes).hex()
  with pytest.raises(db.Error, match='Unsupported image format'):
    db._ImportImageFile(bmp_path, bmp_bytes, img_hash, set())


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
  img_path, img_bytes = _MakeTestPNG(src_dir, params=params)
  img_hash: str = hashes.Hash256(img_bytes).hex()
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  ai_db.Sync(add_dir=src_dir)
  assert img_hash in ai_db._db['images']
  entry: db.DBImageType = ai_db._db['images'][img_hash]
  assert entry['path'] == str(img_path)
  assert entry['ai_meta']['positive'] == 'sunset photo'
  assert entry['ai_meta']['negative'] == 'rain'
  assert entry['ai_meta']['seed'] == 777
  assert entry['ai_meta']['steps'] == 15


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
  """Sync restores path for a known image whose path was None."""
  src_dir: pathlib.Path = tmp_path / 'imgs'
  src_dir.mkdir()
  img_path, img_bytes = _MakeTestPNG(src_dir)
  img_hash: str = hashes.Hash256(img_bytes).hex()
  # pre-populate DB with this image but path=None
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  ai_db._db['images'][img_hash] = _MakeDBImage(img_hash=img_hash, path=None)
  ai_db._db['known_image_sources'].append(str(src_dir))
  ai_db.Sync()
  assert ai_db._db['images'][img_hash]['path'] == str(img_path)


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
  assert entry['path'] == other_path
  assert str(img_path) in entry['alt_path']


def testSyncDeletedImagePathCleared(tmp_path: pathlib.Path) -> None:
  """Sync clears the path of a DB image whose file no longer exists."""
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  missing_path: str = str(tmp_path / 'gone.png')
  meta: db.AIMetaType = _MakeMeta()
  ai_db._db['images']['deadbeef'] = _MakeDBImage(meta=meta, img_hash='deadbeef', path=missing_path)
  ai_db.Sync()
  assert ai_db._db['images']['deadbeef']['path'] is None


def testSyncDeletedPrimaryAltPromoted(tmp_path: pathlib.Path) -> None:
  """Sync promotes an alt path when former primary path is gone but alt still exists."""
  src_dir: pathlib.Path = tmp_path / 'imgs'
  src_dir.mkdir()
  # create alt image file on disk
  alt_path_obj, alt_bytes = _MakeTestPNG(src_dir, name='alt.png')
  alt_hash: str = hashes.Hash256(alt_bytes).hex()
  gone_path: str = str(tmp_path / 'gone.png')
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  entry: db.DBImageType = _MakeDBImage(img_hash=alt_hash, path=gone_path)
  entry['alt_path'] = [str(alt_path_obj)]
  ai_db._db['images'][alt_hash] = entry
  ai_db.Sync()
  updated: db.DBImageType = ai_db._db['images'][alt_hash]
  assert updated['path'] == str(alt_path_obj)
  assert gone_path not in updated['alt_path']


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
  _, b1 = _MakeTestPNG(src_dir, 'a.png')
  _, b2 = _MakeTestPNG(sub_dir, 'b.png')
  h1: str = hashes.Hash256(b1).hex()
  h2: str = hashes.Hash256(b2).hex()
  ai_db = db.AIDatabase(_MakeAppConfig(tmp_path))
  ai_db.Sync(add_dir=src_dir)
  assert h1 in ai_db._db['images']
  assert h2 in ai_db._db['images']
