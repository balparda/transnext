# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""TransNext core database module."""

from __future__ import annotations

import abc as abstract
import enum
import logging
import pathlib
import threading
from typing import Protocol, Self, TypedDict, cast, runtime_checkable

from transai.core import ai
from transcrypto.core import aes, hashes, key
from transcrypto.utils import base as tbase
from transcrypto.utils import config as app_config
from transcrypto.utils import saferandom, timer

from transnext import __version__
from transnext.core import base

_DB_COMPRESS_LEVEL = 5  # default compression level for DB saving
_DB_DISK_LOCK: threading.Lock = threading.Lock()  # lock for thread-safe DB operations


class Error(base.Error):
  """TransNext DB exception."""


class _DBType(TypedDict):
  """DB object type.

  Should be suitable for JSON and pickle serialization, so no complex types or custom classes.
  Don't use sets.
  """

  version: int  # DB version; increment on save
  db_version: str  # package version (transnext.__version__) at time of last save
  last_save: int  # timestamp of last save
  images: dict[str, DBImageType]  # {image_hash: image_metadata}
  models: dict[str, AIModelType]  # {model_hash: model}
  known_image_sources: list[str]  # places to track images from
  image_output_dir: str | None  # current directory to save generated images to


def _DBTypeFactory(overrides: dict[str, object] | None = None) -> _DBType:
  """Create new _DBType object with default values.

  Returns:
    A new _DBType object with default values.

  """
  obj: _DBType = {
    'version': 0,
    'db_version': __version__,  # set to current package version on creation
    'last_save': timer.Now(),
    'images': {},
    'models': {},
    'known_image_sources': [],
    'image_output_dir': None,
  }
  obj.update(overrides or {})  # type: ignore[typeddict-item]
  return obj


class ImageFormat(enum.Enum):
  """Image format enum."""

  JPEG = 'JPEG'
  PNG = 'PNG'
  GIF = 'GIF'


class DBImageType(TypedDict):
  """DB image metadata type."""

  hash: str  # image hash: the hash of the file on disk, usually a PNG file
  raw_hash: str  # raw image hash: the hash of the raw image data computed internally to PIL
  path: str | None  # image file path
  alt_path: list[str]  # alternative paths where the image is found (e.g., duplicates)
  size: int  # image file size
  width: int  # image width
  height: int  # image height
  format: str  # ImageFormat string (e.g., JPEG, PNG, GIF)
  created_at: int  # timestamp of when the image was added to the DB
  ai_meta: AIMetaType  # AI generation metadata
  sd_info: tbase.JSONDict  # additional info from SDNext API response
  sd_params: tbase.JSONDict  # additional info from SDNext API response


class AIModelType(TypedDict):
  """AI model metadata type."""

  hash: str  # hash of the model file
  name: str  # AI model identifier
  path: str  # path to the model file
  type: str  # model file type (e.g., 'safetensors', 'ckpt')
  description: str | None  # brief description of the model's capabilities


class AIMetaType(TypedDict):
  """AI image metadata.

  This is supposed to hold all the options this software handles that affects image generation,
  so that we can later on use it for filtering and searching images by these parameters.
  It should be complete enough to allow us to reproduce the generation of the image.

  Keep it shallow and simple, only primitive types, no complex objects or custom classes,
  to ensure it is comparable and serializable without issues.

  No dict, no list. If we add that we'll have to implement custom comparison.
  """

  model_hash: str  # AI model hash
  positive: str  # positive prompt used for AI processing
  negative: str | None  # negative prompt used for AI processing
  seed: int  # seed used in AI processing
  width: int  # image width
  height: int  # image height
  steps: int  # number of steps used in AI processing
  cfg_scale: int  # CFG scale used in AI processing (guidance scale, times 10 for int storage)
  cfg_end: int  # CFG end step used in AI processing (times 10 for int storage)
  sampler: str  # base.Sampler string used in AI processing
  parser: str  # base.QueryParser string used for the prompts
  clip_skip: int  # CLIP skip used in AI processing


def AIMetaTypeFactory(overrides: dict[str, object] | None = None) -> AIMetaType:
  """Create new AIMetaType object with default values.

  SPECIAL CASE: seed in {None, -1, 0} will NOT overwrite, will generate a random seed!

  Returns:
    A new AIMetaType object with default values.

  Raises:
    Error: if the resulting seed value is invalid (not in the range 1 < seed ≤ ai.AI_MAX_SEED)

  """
  obj: AIMetaType = {
    'model_hash': '',
    'positive': '',
    'negative': None,
    'seed': -1,  # start with random flag
    'width': base.SD_DEFAULT_WIDTH,
    'height': base.SD_DEFAULT_HEIGHT,
    'steps': base.SD_DEFAULT_ITERATIONS,
    'sampler': base.SD_DEFAULT_SAMPLER.value,
    'parser': base.SD_DEFAULT_QUERY_PARSER.value,
    'cfg_scale': base.SD_DEFAULT_CFG_SCALE,
    'cfg_end': base.SD_DEFAULT_CFG_END,
    'clip_skip': base.SD_DEFAULT_CLIP_SKIP,
  }
  obj.update(overrides or {})  # type: ignore[typeddict-item]
  # make sure seed is actually set now
  obj['seed'] = saferandom.RandBits(31) if obj['seed'] in {None, -1, 0} else obj['seed']
  if not 1 < obj['seed'] <= ai.AI_MAX_SEED:
    raise Error(f'Invalid seed value in AIMetaType: {obj}')
  return obj


class AIDatabase:
  """AI database."""

  def __init__(
    self,
    config: app_config.AppConfig,
    *,
    read_only: bool = False,
    aes_key: aes.AESKey | None = None,
    safe_save: bool = True,
    compress_save: bool = False,
  ) -> None:
    """Initialize the AI Database.

    Args:
      config: The application configuration object.
      read_only: (default False) Whether to open the database in read-only mode.
      aes_key: (default None) Optional AES key for encrypting/decrypting the database file
      safe_save: (default True) Whether to use a safe save method that reads the existing DB file
          before writing, to prevent data loss from clobbering; if False, it will overwrite
          the file directly
      compress_save: (default False) Whether to compress the DB file when saving; if True, it will
          save as a compressed file

    """
    self._config: app_config.AppConfig = config
    self._read_only: bool = read_only
    self._key: aes.AESKey | None = aes_key
    self._safe_save: bool = safe_save
    self._compress_save: bool = compress_save
    self._db: _DBType
    self._open = timer.Timer('AIDatabase', emit_log=False)
    with _DB_DISK_LOCK:  # ensure thread-safe load operations
      if self._config.path.exists():
        self._db = cast(
          '_DBType', config.DeSerialize(decryption_key=self._key, unpickler=key.UnpickleJSON)
        )
        logging.info(f'Loaded DB from {self._config.path}: {self.label}')
      else:
        self._db = _DBTypeFactory()
        logging.warning(f'DB file not found, will start fresh, {self.label}')
    if self._read_only:
      logging.warning('ATTENTION: AIDatabase opened in read-only mode, changes will not be saved!')

  def __enter__(self) -> Self:
    """Context manager entry, returns self.

    Returns:
      self, for use within the context

    """
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    _exc_value: BaseException | None,
    _traceback: object,
  ) -> None:
    """Context manager exit. If not exception, saves the database to file.

    Args:
      exc_type: exception type, if any
      _exc_value: exception value, if any
      _traceback: traceback object, if any

    """
    logging.info(f'AIDatabase was open for {self._open}')
    if exc_type is not None:
      logging.error('Exception occurred in AIDatabase context: *NOT* saving DB due to exception')
      return  # do not save if there was an exception
    self.Save()

  @property
  def label(self) -> str:
    """Get a human-readable label for the database, for logging and display purposes.

    Returns:
      string '#<N>@<tm>'

    """
    return _DBLabel(self._db)

  @property
  def output(self) -> pathlib.Path | None:
    """Get current output directory for generated images.

    Returns:
      pathlib.Path or None if not set

    """
    return pathlib.Path(self._db['image_output_dir']) if self._db['image_output_dir'] else None

  @output.setter
  def output(self, value: pathlib.Path | str | None) -> None:
    """Set current output directory for generated images.

    Args:
      value: pathlib.Path, string path, or None to clear the output directory

    Raises:
      Error: if the provided path is invalid (does not exist or is not a directory)

    """
    path: pathlib.Path | None = (
      pathlib.Path(value).expanduser().resolve() if value is not None else None
    )
    if path is None:
      self._db['image_output_dir'] = None
      logging.info('Cleared DB image output directory')
      return
    if not path.exists() or not path.is_dir():
      raise Error(f'Invalid output directory: {path} (must exist and be a directory)')
    str_path: str = str(path)
    self._db['image_output_dir'] = str_path
    if str_path not in self._db['known_image_sources']:
      self._db['known_image_sources'].append(str_path)  # track this path as a known image source
    logging.info(f'Set DB image output directory to: {path} (also added to known image sources)')

  def Save(self) -> None:
    """Save the database to file.

    Raises:
      Error: if safe_save is enabled and the existing DB on disk differs from the loaded DB

    """
    if self._read_only:
      logging.warning('AIDatabase in read-only mode: will *NOT* save! (would have saved here)')
      return
    with _DB_DISK_LOCK:  # ensure thread-safe save operations
      # check on previous save
      if self._safe_save and self._config.path.exists():
        logging.debug('Safe save enabled, reading existing DB before saving to prevent data loss')
        existing_db: _DBType = cast(
          '_DBType',
          self._config.DeSerialize(
            decryption_key=self._key, unpickler=key.UnpickleJSON, silent=True
          ),
        )
        if (
          existing_db['version'] != self._db['version']
          or existing_db['last_save'] != self._db['last_save']
        ):
          raise Error(
            f'DB on disk {_DBLabel(existing_db)} differs from loaded DB {self.label}, '
            'aborting save to prevent data loss'
          )
      # update DB metadata before saving
      prev_label: str = self.label
      self._db.update(
        {
          'version': self._db['version'] + 1,  # increment DB version on each save
          'db_version': __version__,  # set to current package version on creation
          'last_save': timer.Now(),
        }
      )
      # save the DB to disk with optional encryption and compression
      self._config.Serialize(
        cast('tbase.JSONDict', self._db),
        encryption_key=self._key,
        pickler=key.PickleJSON,
        compress=_DB_COMPRESS_LEVEL if self._compress_save else None,
      )
      logging.info(f'DB saved to {self._config.path}: {prev_label} -> {self.label}')

  def GetModelHash(self, model_name: str, *, api: APIProtocol) -> str:
    """Get the model hash for a given model key. If API is given will also try to fetch.

    Args:
      model_name: The model name to look up
      api: APIProtocol instance to use for fetching available models if not already in DB

    Returns:
      The model hash corresponding to the given model name

    Raises:
      Error: on error or if the model is not found

    """
    # check the name
    model_name = model_name.lower().strip()
    if not model_name:
      raise Error('Model name cannot be empty')
    # search in DB models
    possible_models: list[AIModelType] = [
      model for model in self._db['models'].values() if model_name in model['name'].lower()
    ]
    if len(possible_models) > 1:
      raise Error(f'Multiple models found matching name "{model_name}": {possible_models}')
    if len(possible_models) == 1:
      return possible_models[0]['hash']  # found unique
    # model not found, we will fetch *all* models
    logging.warning(f'Model with name "{model_name}" not found in DB, fetching all models from API')
    models: list[AIModelType] = api.GetModels()
    # add missing hashes
    all_paths: set[str] = {model['path'] for model in self._db['models'].values() if model['path']}
    for model in models:
      if model['path'] in all_paths:
        continue  # skip duplicates by path (we know we have this one), we only want unique models
      if not model['hash'].strip():
        logging.warning(f'Model with empty hash received from API, hashing {model["path"]}')
        model['hash'] = hashes.Hash256(pathlib.Path(model['path']).read_bytes()).hex()
      logging.info(f'Adding model to DB: {model["name"]} -> {model["hash"]}')
      self._db['models'][model['hash']] = model  # add to DB models
    # we have refreshed the DB models, try searching again
    possible_models = [
      model for model in self._db['models'].values() if model_name in model['name'].lower()
    ]
    if len(possible_models) > 1:
      raise Error(f'Multiple models found matching name "{model_name}": {possible_models}')
    if len(possible_models) == 1:
      return possible_models[0]['hash']  # found unique
    raise Error(
      f'Model with name "{model_name}" not found, try available names: '
      f'{[model["name"] for model in self._db["models"].values()]}'
    )

  def Txt2Img(self, api: APIProtocol, meta: AIMetaType) -> tuple[DBImageType, bytes]:
    """Generate image from text prompt, store in DB.

    Args:
      api: APIProtocol instance to use for making the API call
      meta: AIMetaType object containing the generation metadata (e.g., prompt, steps,
          seed, width, height, sampler_id, model_key)

    Returns:
      A tuple containing the DBImageType object and the raw image data.

    Raises:
      Error: on error

    """
    # we have the model?
    if meta['model_hash'] not in self._db['models']:
      raise Error(f'Model with hash {meta["model_hash"]} not found in DB models')
    # try to find it here already
    db_entry: DBImageType
    for h, db_entry in self._db['images'].items():
      if db_entry['path'] and db_entry['ai_meta'] == meta:
        # we have a file that matches, or should have...
        path = pathlib.Path(db_entry['path'])
        if not path.exists():
          logging.warning(f'Image {meta} should exist as {h}: {path} but FILE NOT FOUND')
          db_entry['path'] = None  # clear the path since it's not valid
          continue  # give up on this, probably means this entry will be replaced...
        logging.info(f'Image {meta} exists with hash {h}: {path}')
        return (db_entry, path.read_bytes())
    # not found, generate new
    img_bytes: bytes
    db_entry, img_bytes = api.Txt2Img(
      self._db['models'][meta['model_hash']].copy(), meta, dir_root=self.output
    )
    self._db['images'][db_entry['hash']] = db_entry.copy()  # add to DB images
    return (db_entry, img_bytes)


def _DBLabel(db: _DBType) -> str:
  """Get a human-readable label for the database, for logging and display purposes.

  Returns:
    string '#<N>@<tm>'

  """
  return f'#{db["version"]}@{timer.TimeStr(db["last_save"])}'


@runtime_checkable
class APIProtocol(Protocol):
  """SDNext API protocol contract needed by the DB."""

  @abstract.abstractmethod
  def GetModels(self) -> list[AIModelType]:
    """Get list of available models from SDNext API.

    Returns:
      A list of AIModelType objects representing the available models. BEWARE: hash may be ''!

    Raises:
      Error: If there is an error with the API call or if the response is invalid.

    """

  @abstract.abstractmethod
  def Txt2Img(
    self, model: AIModelType, meta: AIMetaType, *, dir_root: pathlib.Path | None = None
  ) -> tuple[DBImageType, bytes]:
    """Generate image from text prompt using SDNext API.

    See: <https://github.com/vladmandic/sdnext/blob/master/cli/api-txt2img.py>

    Args:
      model: AIModelType object representing the model to use for generation
      meta: AIMetaType object containing the generation metadata (e.g., prompt, steps,
          seed, width, height, sampler_id, model_key)
      dir_root: (default: None) Directory root to save the generated image, if None don't save

    Returns:
      A tuple containing the DBImageType object and the raw image data.

    Raises:
      Error: If there is an error with the API call, if the response is invalid,
          or if the image data is invalid.

    """
