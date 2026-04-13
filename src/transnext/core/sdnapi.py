# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""TransNext core SDNext API module."""

from __future__ import annotations

import base64
import enum
import io
import json
import logging
import pathlib
import time
from collections import abc
from typing import cast

import requests
import urllib3
from PIL import Image
from transcrypto.core import hashes
from transcrypto.utils import base as tbase
from transcrypto.utils import human, timer

from transnext.core import base, db

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_SUCCESS_STATUSES: set[int] = {200}

_APP_NAME: str = 'sdnext.git'
_API_PREFIX: dict[int, str] = {
  1: '/sdapi/v1/',
  2: '/sdapi/v2/',
}


class Endpoints(enum.Enum):
  """SDNext API endpoints."""

  SYSTEM_STATUS = 'system-info/status'
  MODELS = 'sd-models'
  OPTIONS = 'options'
  RELOAD_CHECKPOINT = 'reload-checkpoint'
  TXT2IMG = 'txt2img'


_API_TYPES: dict[Endpoints, abc.Callable[..., requests.Response]] = {
  Endpoints.SYSTEM_STATUS: requests.get,
  Endpoints.MODELS: requests.get,
  Endpoints.OPTIONS: requests.post,
  Endpoints.RELOAD_CHECKPOINT: requests.post,
  Endpoints.TXT2IMG: requests.post,
}


class Error(base.Error):
  """TransNext SDNext API exception."""


class APIConnectionError(Error, ConnectionError):
  """TransNext SDNext API connection exception."""


class API(db.APIProtocol):
  """SDNext API client."""

  def __init__(self, api_url: str, *, version: int = 1, server_save_images: bool = False) -> None:
    """Initialize SDNext API client.

    Args:
      api_url: Base URL of the SDNext API, e.g. "http://127.0.0.1:5000"
      version: API version to use (default is 1)
      server_save_images: Whether if the server will save a copy of the images too (default False)

    Raises:
      APIConnectionError: If there is a connection error to the SDNext API.

    """  # noqa: DOC502
    self._api_url: str = api_url
    self._version: int = version
    self._server_save_images: bool = server_save_images
    server: tuple[str, str] = self.ServerVersion()
    logging.info(
      f'API(v#{self._version})/{server[0]}/{server[1]} '
      f'@ {self._api_url}{" + SAVE" if self._server_save_images else ""}'
    )

  def Call(self, endpoint: Endpoints, payload: tbase.JSONDict | None = None) -> tbase.JSONValue:
    """Call SDNext API endpoint with given payload.

    Args:
      endpoint: API endpoint to call
      payload: JSON-serializable payload to send in the request body (default is None)

    Returns:
      The JSON response from the API as a dictionary.

    Raises:
      APIConnectionError: If there is a connection error to the SDNext API.

    """  # noqa: DOC502
    return _Call(
      _API_TYPES[endpoint], self._api_url, _API_PREFIX[self._version] + endpoint.value, payload
    )

  def ServerVersion(self) -> tuple[str, str]:
    """Get SDNext API server version info.

    Returns:
      A tuple containing the server version hash and updated timestamp, example:
      ('0eb4a98e0', '2026-04-04')

    Raises:
      Error: If there is an error with the API call or if the response is invalid.

    """
    info: tbase.JSONValue = self.Call(Endpoints.SYSTEM_STATUS, {'full': True, 'refresh': True})
    if not isinstance(info, dict) or 'version' not in info:
      raise Error(f'Invalid system status response from SDNext API: {info}')
    version: dict[str, str] = cast('dict[str, str]', info['version'])
    if (
      not isinstance(version, dict)  # pyright: ignore[reportUnnecessaryIsInstance]
      or 'app' not in version
      or 'updated' not in version
      or 'hash' not in version
    ):
      raise Error(f'Invalid version info in system status response from SDNext API: {info}')
    if version['app'] != _APP_NAME:
      raise Error(f'Unexpected app in version info from SDNext API: {version}')
    return (version['hash'], version['updated'])

  def GetModels(self) -> list[db.AIModelType]:
    """Get list of available models from SDNext API.

    Returns:
      A list of AIModelType objects representing the available models. BEWARE: hash may be ''!

    Raises:
      Error: If there is an error with the API call or if the response is invalid.

    """
    models: tbase.JSONValue = self.Call(Endpoints.MODELS)
    if not isinstance(models, list) or not models:
      raise Error(f'Invalid models response from SDNext API: {models}')
    # we got a valid response, parse it
    parsed: list[db.AIModelType] = []
    for model in models:
      if not isinstance(model, dict):
        raise Error(f'Invalid model entry in SDNext API response: {model}')
      model_path = pathlib.Path(cast('str', model.get('filename', '')).strip())
      new_model: db.AIModelType = db.AIModelType(
        hash=cast('str', model.get('sha256', '') or '').strip(),
        name=cast('str', model.get('model_name', '') or '').strip(),
        path=str(model_path),
        type=cast('str', model.get('type', '') or '').strip(),
        description=None,
      )
      if not model_path.exists():
        raise Error(f'Model file not found for model from SDNext API: {new_model}')
      if not new_model['name'] or not new_model['type']:
        raise Error(f'Missing model name or type for model from SDNext API: {new_model}')
      parsed.append(new_model)
    # done, return
    return parsed

  def LoadModel(self, model: str) -> None:
    """Load model in SDNext API.

    Args:
      model: Model name to load (e.g., "model1")

    """
    self.Call(Endpoints.OPTIONS, {'sd_model_checkpoint': model})
    self.Call(Endpoints.RELOAD_CHECKPOINT)  # needed if running in api-only to trigger new load

  def Txt2Img(
    self, model: db.AIModelType, meta: db.AIMetaType, *, dir_root: pathlib.Path | None = None
  ) -> tuple[db.DBImageType, bytes]:
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
    # look at model loading
    if model['hash'] != meta['model_hash']:
      raise Error(f'Model hash mismatch: expected {meta["model_hash"]}, got {model["hash"]}')
    self.LoadModel(model['name'])
    # set the options
    if (
      meta['width'] <= 0
      or meta['height'] <= 0
      or meta['width'] % 16 != 0
      or meta['height'] % 16 != 0
    ):
      raise Error(f'Invalid image dimensions: {meta}')
    options: tbase.JSONDict = {
      'save_images': self._server_save_images,
      'send_images': True,
      'prompt': meta['positive'],
      'negative_prompt': meta['negative'],
      'steps': meta['steps'],
      'seed': meta['seed'],
      'width': meta['width'],
      'height': meta['height'],
      'sampler_name': meta['sampler'],
      'prompt_attention': meta['parser'],
      'cfg_scale': meta['cfg_scale'] / 10,  # remember to divide by 10
      'cfg_end': meta['cfg_end'] / 10,  # remember to divide by 10
      'clip_skip': meta['clip_skip'] // 10,  # TODO: in future, when accepts float do regular /
      'clip_skip_enabled': True,
    }
    # make the call to the APIs
    data: tbase.JSONValue = self.Call(Endpoints.TXT2IMG, options)
    tm_created: int = timer.Now()  # use exact time we got is back
    if not isinstance(data, dict) or 'info' not in data or 'parameters' not in data:
      raise Error(f'Invalid image metadata received from SDNext API: {data}')
    if (
      meta['width'] != data['parameters']['width'] or meta['height'] != data['parameters']['height']  # type: ignore[index,call-overload]
    ):
      raise Error(
        f'Expected image of size {data["parameters"]["width"]}x{data["parameters"]["height"]} from '  # type: ignore[index,call-overload]
        f'SDNext API, got {meta["width"]}x{meta["height"]}: {data}'
      )
    # extract the data
    img_data: bytes
    raw_hash: str
    img_data, raw_hash = _ExtractImageData(data)
    img_hash: str = hashes.Hash256(img_data).hex()
    logging.info(f'Got generated image, {human.HumanizedBytes(len(img_data))}: {img_hash}')
    # if we are going to save the image, figure out the full path
    full_path: pathlib.Path | None = None
    if dir_root is not None:
      date_str: str = time.strftime('%Y-%m-%d', time.gmtime(tm_created))
      tm_str: str = time.strftime('%Y%m%d%H%M%S', time.gmtime(tm_created))
      out_dir: pathlib.Path = dir_root / date_str
      out_dir.mkdir(exist_ok=True)  # make sure the date dir exists, create it if not
      filename: str = (
        f'{base.PromptHash(meta["positive"], meta["negative"])}-'
        f'{tm_str}-'
        f'{model["hash"][:8]}-'
        f'{meta["cfg_scale"]}-{meta["steps"]}-'
        f'{meta["width"]}-{meta["height"]}-'
        f'{meta["seed"]}-'
        f'{img_hash[:12]}.png'
      )
      full_path = out_dir / filename
      full_path.write_bytes(img_data)
      logging.info(f'SDNext API image saved: {full_path}, {human.HumanizedBytes(len(img_data))}')
    # create the metadata
    db_image: db.DBImageType = {
      'hash': img_hash,
      'raw_hash': raw_hash,
      'path': str(full_path) if full_path else None,
      'alt_path': [],
      'size': len(img_data),
      'width': meta['width'],
      'height': meta['height'],
      'format': db.ImageFormat.PNG.value,
      'created_at': tm_created,
      'ai_meta': meta.copy(),
      'sd_info': json.loads(cast('str', data['info'])),
      'sd_params': cast('tbase.JSONDict', data['parameters']),
    }
    if (
      meta['width'] != db_image['sd_info']['width']
      or meta['height'] != db_image['sd_info']['height']
    ):
      raise Error(
        f'Expected image of size {db_image["sd_info"]["width"]}x{db_image["sd_info"]["height"]} '
        f'from SDNext API, got {meta["width"]}x{meta["height"]}: {data}'
      )
    # all is good, save the image to disk and return the DB entry and raw image data
    logging.debug(f'SDNext API image metadata: {db_image}')
    return (db_image, img_data)


def _ExtractImageData(data: tbase.JSONDict) -> tuple[bytes, str]:
  """Extract and validate image data from SDNext API response.

  Args:
    data: JSON response from SDNext API containing the image data and metadata

  Returns:
    A tuple containing the PNG image data as bytes and the computed hash of the raw image data.

  Raises:
    Error: If the image data is missing, invalid, or does not match the expected format

  """
  # check image is here, exactly as expected, decode it
  if 'images' not in data:
    raise Error(f'No images received from SDNext API: {data}')
  if 'parameters' not in data:
    raise Error(f'Image metadata not received from SDNext API: {data}')
  if not isinstance((data_images := data['images']), list) or len(data_images) != 1:
    raise Error(f'Expected exactly 1 image from SDNext API: {data}')
  b64: str = data['images'][0]  # type: ignore[index,assignment]
  if ',' in b64:
    # the code in the API callers of SDNext repo have "b64.split(',', 1)[0]" here, but
    # that seems to be unnecessary in this API, so we'll raise until we see it happening
    raise Error(f'Unexpected comma in base64 image data from SDNext API: {data}')
  img_data: bytes = base64.b64decode(b64)
  if not img_data:
    raise Error(f'Image data empty from SDNext API: {data}')
  # open the image so we can do some basic validation
  with Image.open(io.BytesIO(img_data)) as image:
    # do some basic validation on the image and metadata, prepare the DB entry for it
    if image.format != db.ImageFormat.PNG.value:
      raise Error(f'Expected PNG image from SDNext API, got {image.format}: {data}')
    if image.width != data['parameters']['width'] or image.height != data['parameters']['height']:  # type: ignore[index,call-overload]
      raise Error(
        f'Expected image of size {data["parameters"]["width"]}x{data["parameters"]["height"]} from '  # type: ignore[index,call-overload]
        f'SDNext API, got {image.width}x{image.height}: {data}'
      )
    # compute the hash of the raw image data for the DB entry, this is more format-agnostic
    raw_hash: str = hashes.Hash256(image.convert('RGBA').tobytes()).hex()
  # passed all checks
  return (img_data, raw_hash)


def _Call(
  method: abc.Callable[..., requests.Response],
  sd_url: str,
  endpoint: str,
  payload: tbase.JSONDict | None = None,
) -> tbase.JSONValue:
  """Call SDNext API endpoint with given payload.

  Args:
    method: HTTP method function from requests library (e.g., requests.post)
    sd_url: Base URL of the SDNext API, e.g. "http://127.0.0.1:5000"
    endpoint: API endpoint to call
    payload: JSON-serializable payload to send in the request body (default is None)

  Returns:
    The JSON response from the API as a dictionary.

  Raises:
    APIConnectionError: If there is a connection error to the SDNext API.
    Error: If the response status code is not successful or if there is an error parsing response.

  """
  full_url: str = f'{sd_url}{endpoint}'
  logging.debug(f'Calling SDNext API {method.__name__.upper()} {full_url} with payload: {payload}')
  try:
    req: requests.Response = method(full_url, json=payload, timeout=300, verify=False)
    if req.status_code not in _SUCCESS_STATUSES:
      raise Error(f'Status {req.status_code} {req.reason} - {req.text}')  # noqa: TRY301
    return cast('tbase.JSONValue', req.json())
  except requests.exceptions.ConnectionError as err:
    raise APIConnectionError(f'Failed to connect to SDNext API {method} {full_url}') from err
  except Exception as err:
    raise Error(f'Error calling SDNext API {method} {full_url}: {err}') from err
