# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: transnext.core.sdnapi module."""

from __future__ import annotations

import base64
import io
import json
from unittest import mock

import pytest
import requests
from PIL import Image
from transcrypto.core import hashes
from transcrypto.utils import base as tbase

from transnext.core import db, sdnapi

# ─── helpers ──────────────────────────────────────────────────────────────────


def _MakePNG(width: int = 64, height: int = 64) -> bytes:
  """Create a minimal valid PNG in memory."""  # noqa: DOC201
  img: Image.Image = Image.new('RGBA', (width, height), color=(255, 0, 0, 255))
  buf = io.BytesIO()
  img.save(buf, format='PNG')
  return buf.getvalue()


def _B64PNG(width: int = 64, height: int = 64) -> str:
  """Create base64-encoded PNG string."""  # noqa: DOC201
  return base64.b64encode(_MakePNG(width, height)).decode('ascii')


def _MakeMeta(overrides: dict[str, object] | None = None) -> db.AIMetaType:
  """Create an AIMetaType with a valid fixed seed for testing."""  # noqa: DOC201
  defaults: dict[str, object] = {'seed': 42, 'model_hash': 'abc123'}
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


def _MockCallResponse(json_data: object, status: int = 200) -> mock.Mock:
  """Create a mock requests.Response."""  # noqa: DOC201
  resp = mock.Mock(spec=requests.Response)
  resp.status_code = status
  resp.reason = 'OK' if status == 200 else 'Error'
  resp.text = json.dumps(json_data) if isinstance(json_data, dict | list) else str(json_data)
  resp.json.return_value = json_data
  return resp


def _Txt2ImgAPIData(
  width: int = 64,
  height: int = 64,
) -> dict[str, object]:
  """Build a full valid Txt2Img API response dict."""  # noqa: DOC201
  return {
    'images': [_B64PNG(width, height)],
    'info': json.dumps({'width': width, 'height': height}),
    'parameters': {'width': width, 'height': height},
  }


# ─── Endpoints ────────────────────────────────────────────────────────────────


def testEndpointValues() -> None:
  """Endpoints have correct string values."""
  assert sdnapi.Endpoints.MODELS.value == 'sd-models'
  assert sdnapi.Endpoints.OPTIONS.value == 'options'
  assert sdnapi.Endpoints.RELOAD_CHECKPOINT.value == 'reload-checkpoint'
  assert sdnapi.Endpoints.TXT2IMG.value == 'txt2img'


# ─── Error ────────────────────────────────────────────────────────────────────


def testErrorIsSubclass() -> None:
  """Error inherits from base.Error."""  # noqa: DOC501
  with pytest.raises(sdnapi.Error):
    raise sdnapi.Error('test')
  assert issubclass(sdnapi.Error, Exception)


# ─── API ──────────────────────────────────────────────────────────────────────


def testAPIInit() -> None:
  """API initializes with correct attributes."""
  api = sdnapi.API('http://localhost:7860')
  assert api._api_url == 'http://localhost:7860'
  assert api._version == 1
  assert api._server_save_images is False


def testAPIInitWithOptions() -> None:
  """API initializes with custom version and save options."""
  api = sdnapi.API('http://host:8080', version=2, server_save_images=True)
  assert api._version == 2
  assert api._server_save_images is True


def testCallModels() -> None:
  """Call routes to correct HTTP method and endpoint."""
  api = sdnapi.API('http://localhost:7860')
  with mock.patch.object(sdnapi, '_Call', return_value=[{'model': 'data'}]) as mock_call:
    result: tbase.JSONValue = api.Call(sdnapi.Endpoints.MODELS)
  assert isinstance(result, list)
  mock_call.assert_called_once()
  # check the endpoint path is correct
  args = mock_call.call_args
  assert args[0][2] == '/sdapi/v1/sd-models'


def testCallTxt2ImgWithPayload() -> None:
  """Call passes payload through to _Call."""
  api = sdnapi.API('http://localhost:7860')
  with mock.patch.object(sdnapi, '_Call', return_value={'result': 'ok'}) as mock_call:
    result: tbase.JSONValue = api.Call(sdnapi.Endpoints.TXT2IMG, {'prompt': 'hello'})
  assert result == {'result': 'ok'}
  args = mock_call.call_args
  assert args[0][3] == {'prompt': 'hello'}


def testCallV2Prefix() -> None:
  """Call uses v2 prefix when version is 2."""
  api = sdnapi.API('http://localhost:7860', version=2)
  with mock.patch.object(sdnapi, '_Call', return_value={}) as mock_call:
    api.Call(sdnapi.Endpoints.MODELS)
  args = mock_call.call_args
  assert args[0][2] == '/sdapi/v2/sd-models'


def testGetModelsValid() -> None:
  """GetModels parses valid response."""
  api = sdnapi.API('http://localhost:7860')
  api_response: tbase.JSONValue = [
    {
      'sha256': 'hash1',
      'model_name': 'my-model',
      'filename': '/tmp/model.safetensors',  # noqa: S108
      'type': 'safetensors',
    },
  ]
  with (
    mock.patch.object(api, 'Call', return_value=api_response),
    mock.patch('pathlib.Path.exists', return_value=True),
  ):
    models: list[db.AIModelType] = api.GetModels()
  assert len(models) == 1
  assert models[0]['name'] == 'my-model'
  assert models[0]['hash'] == 'hash1'


def testGetModelsEmptyResponseRaises() -> None:
  """GetModels raises Error on empty response."""
  api = sdnapi.API('http://localhost:7860')
  with (
    mock.patch.object(api, 'Call', return_value=[]),
    pytest.raises(sdnapi.Error, match='Invalid models response'),
  ):
    api.GetModels()


def testGetModelsNotListRaises() -> None:
  """GetModels raises Error on non-list response."""
  api = sdnapi.API('http://localhost:7860')
  with (
    mock.patch.object(api, 'Call', return_value={'not': 'a list'}),
    pytest.raises(sdnapi.Error, match='Invalid models response'),
  ):
    api.GetModels()


def testGetModelsInvalidEntryRaises() -> None:
  """GetModels raises Error on non-dict entry."""
  api = sdnapi.API('http://localhost:7860')
  with (
    mock.patch.object(api, 'Call', return_value=['not-a-dict']),
    pytest.raises(sdnapi.Error, match='Invalid model entry'),
  ):
    api.GetModels()


def testGetModelsFileNotFoundRaises() -> None:
  """GetModels raises Error when model file does not exist."""
  api = sdnapi.API('http://localhost:7860')
  api_response: tbase.JSONValue = [
    {
      'sha256': 'hash1',
      'model_name': 'my-model',
      'filename': '/nonexistent/model.safetensors',
      'type': 'safetensors',
    },
  ]
  with (
    mock.patch.object(api, 'Call', return_value=api_response),
    mock.patch('pathlib.Path.exists', return_value=False),
    pytest.raises(sdnapi.Error, match='Model file not found'),
  ):
    api.GetModels()


def testGetModelsMissingNameRaises() -> None:
  """GetModels raises Error when model name is missing."""
  api = sdnapi.API('http://localhost:7860')
  api_response: tbase.JSONValue = [
    {
      'sha256': 'hash1',
      'model_name': '',
      'filename': '/tmp/model.safetensors',  # noqa: S108
      'type': 'safetensors',
    },
  ]
  with (
    mock.patch.object(api, 'Call', return_value=api_response),
    mock.patch('pathlib.Path.exists', return_value=True),
    pytest.raises(sdnapi.Error, match='Missing model name or type'),
  ):
    api.GetModels()


def testGetModelsMissingTypeRaises() -> None:
  """GetModels raises Error when model type is missing."""
  api = sdnapi.API('http://localhost:7860')
  api_response: tbase.JSONValue = [
    {
      'sha256': 'hash1',
      'model_name': 'my-model',
      'filename': '/tmp/model.safetensors',  # noqa: S108
      'type': '',
    },
  ]
  with (
    mock.patch.object(api, 'Call', return_value=api_response),
    mock.patch('pathlib.Path.exists', return_value=True),
    pytest.raises(sdnapi.Error, match='Missing model name or type'),
  ):
    api.GetModels()


def testGetModelsNullHashAndType() -> None:
  """GetModels handles None values for hash and type via 'or' fallback."""
  api = sdnapi.API('http://localhost:7860')
  api_response: tbase.JSONValue = [
    {
      'sha256': None,
      'model_name': 'my-model',
      'filename': '/tmp/model.safetensors',  # noqa: S108
      'type': None,
    },
  ]
  with (
    mock.patch.object(api, 'Call', return_value=api_response),
    mock.patch('pathlib.Path.exists', return_value=True),
    # type is empty after stripping None -> ''
    pytest.raises(sdnapi.Error, match='Missing model name or type'),
  ):
    api.GetModels()


def testLoadModel() -> None:
  """LoadModel makes two API calls (OPTIONS then RELOAD)."""
  api = sdnapi.API('http://localhost:7860')
  with mock.patch.object(api, 'Call') as mock_call:
    api.LoadModel('my-model')
  assert mock_call.call_count == 2
  # first call is OPTIONS with model name, second is RELOAD_CHECKPOINT
  mock_call.assert_any_call(
    sdnapi.Endpoints.OPTIONS,
    {'sd_model_checkpoint': 'my-model'},
  )
  mock_call.assert_any_call(sdnapi.Endpoints.RELOAD_CHECKPOINT)


def testTxt2ImgSuccess() -> None:
  """Txt2Img returns valid DBImageType and image bytes."""
  meta: db.AIMetaType = _MakeMeta({'width': 64, 'height': 64})
  model: db.AIModelType = _MakeModel()
  png_bytes: bytes = _MakePNG(64, 64)
  raw_hash: str = hashes.Hash256(
    Image.open(io.BytesIO(png_bytes)).convert('RGBA').tobytes(),
  ).hex()
  api = sdnapi.API('http://localhost:7860')
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.return_value = _Txt2ImgAPIData(64, 64)
    # first two calls are LoadModel, third is TXT2IMG
    mock_call.side_effect = [None, None, _Txt2ImgAPIData(64, 64)]
    db_img, img_data = api.Txt2Img(model, meta)
  assert db_img['width'] == 64
  assert db_img['height'] == 64
  assert db_img['format'] == 'PNG'
  assert db_img['raw_hash'] == raw_hash
  assert len(img_data) > 0


def testTxt2ImgWithDirRoot(tmp_path: mock.Mock) -> None:
  """Txt2Img saves image to disk when dir_root is provided."""
  meta: db.AIMetaType = _MakeMeta({'width': 64, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api = sdnapi.API('http://localhost:7860')
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [None, None, _Txt2ImgAPIData(64, 64)]
    db_img, _img_data = api.Txt2Img(model, meta, dir_root=tmp_path)
  assert db_img['path'] is not None
  assert db_img['path'].endswith('.png')


def testTxt2ImgHashMismatchRaises() -> None:
  """Txt2Img raises Error when model hash doesn't match meta."""
  meta: db.AIMetaType = _MakeMeta({'model_hash': 'expected'})
  model: db.AIModelType = _MakeModel(h='different')
  api = sdnapi.API('http://localhost:7860')
  with pytest.raises(sdnapi.Error, match='Model hash mismatch'):
    api.Txt2Img(model, meta)


def testTxt2ImgInvalidDimensionsRaises() -> None:
  """Txt2Img raises Error for invalid image dimensions (not multiple of 16)."""
  meta: db.AIMetaType = _MakeMeta({'width': 15, 'height': 64})  # 15 not divisible by 16
  model: db.AIModelType = _MakeModel()
  api = sdnapi.API('http://localhost:7860')
  with (
    mock.patch.object(api, 'Call'),
    pytest.raises(sdnapi.Error, match='Invalid image dimensions'),
  ):
    api.Txt2Img(model, meta)


def testTxt2ImgZeroDimensionsRaises() -> None:
  """Txt2Img raises Error for zero-size dimensions."""
  meta: db.AIMetaType = _MakeMeta({'width': 0, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api = sdnapi.API('http://localhost:7860')
  with (
    mock.patch.object(api, 'Call'),
    pytest.raises(sdnapi.Error, match='Invalid image dimensions'),
  ):
    api.Txt2Img(model, meta)


def testTxt2ImgInvalidResponseRaises() -> None:
  """Txt2Img raises Error when response is missing expected fields."""
  meta: db.AIMetaType = _MakeMeta({'width': 64, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api = sdnapi.API('http://localhost:7860')
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [None, None, {'no-info': True}]
    with pytest.raises(sdnapi.Error, match='Invalid image metadata'):
      api.Txt2Img(model, meta)


def testTxt2ImgParametersSizeMismatchRaises() -> None:
  """Txt2Img raises Error when parameters size doesn't match meta."""
  meta: db.AIMetaType = _MakeMeta({'width': 64, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api = sdnapi.API('http://localhost:7860')
  bad_data: dict[str, object] = {
    'images': [_B64PNG(64, 64)],
    'info': json.dumps({'width': 64, 'height': 64}),
    'parameters': {'width': 128, 'height': 64},  # mismatch!
  }
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [None, None, bad_data]
    with pytest.raises(sdnapi.Error, match='Expected image of size'):
      api.Txt2Img(model, meta)


def testTxt2ImgInfoSizeMismatchRaises() -> None:
  """Txt2Img raises Error when info dimensions don't match meta."""
  meta: db.AIMetaType = _MakeMeta({'width': 64, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api = sdnapi.API('http://localhost:7860')
  bad_data: dict[str, object] = {
    'images': [_B64PNG(64, 64)],
    'info': json.dumps({'width': 128, 'height': 64}),  # mismatch!
    'parameters': {'width': 64, 'height': 64},
  }
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [None, None, bad_data]
    with pytest.raises(sdnapi.Error, match='Expected image of size'):
      api.Txt2Img(model, meta)


def testTxt2ImgNoDirRootNoSave() -> None:
  """Txt2Img sets path to None when no dir_root is given."""
  meta: db.AIMetaType = _MakeMeta({'width': 64, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api = sdnapi.API('http://localhost:7860')
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [None, None, _Txt2ImgAPIData(64, 64)]
    db_img, _img_data = api.Txt2Img(model, meta)
  assert db_img['path'] is None


# ─── _ExtractImageData ───────────────────────────────────────────────────────


def testValidPNG() -> None:
  """Extracts valid PNG image data."""
  png_b64: str = _B64PNG(64, 64)
  data: tbase.JSONDict = {
    'images': [png_b64],
    'parameters': {'width': 64, 'height': 64},
  }
  img_data, raw_hash = sdnapi._ExtractImageData(data)
  assert len(img_data) > 0
  assert len(raw_hash) > 0


def testMissingImagesRaises() -> None:
  """Raises Error when 'images' key is missing."""
  with pytest.raises(sdnapi.Error, match='No images received'):
    sdnapi._ExtractImageData({'parameters': {}})


def testMissingParametersRaises() -> None:
  """Raises Error when 'parameters' key is missing."""
  with pytest.raises(sdnapi.Error, match='Image metadata not received'):
    sdnapi._ExtractImageData({'images': [_B64PNG()]})


def testWrongImageCountRaises() -> None:
  """Raises Error when image count is not exactly 1."""
  data: tbase.JSONDict = {
    'images': [_B64PNG(), _B64PNG()],
    'parameters': {'width': 64, 'height': 64},
  }
  with pytest.raises(sdnapi.Error, match='Expected exactly 1 image'):
    sdnapi._ExtractImageData(data)


def testEmptyImagesListRaises() -> None:
  """Raises Error when images list is empty."""
  data: tbase.JSONDict = {
    'images': [],
    'parameters': {'width': 64, 'height': 64},
  }
  with pytest.raises(sdnapi.Error, match='Expected exactly 1 image'):
    sdnapi._ExtractImageData(data)


def testCommaInBase64Raises() -> None:
  """Raises Error when base64 string contains comma."""
  data: tbase.JSONDict = {
    'images': ['data:image/png;base64,' + _B64PNG()],
    'parameters': {'width': 64, 'height': 64},
  }
  with pytest.raises(sdnapi.Error, match='Unexpected comma'):
    sdnapi._ExtractImageData(data)


def testEmptyImageDataRaises() -> None:
  """Raises Error when decoded image data is empty."""
  data: tbase.JSONDict = {
    'images': [''],  # base64.b64decode('') == b''
    'parameters': {'width': 64, 'height': 64},
  }
  with pytest.raises(sdnapi.Error, match='Image data empty'):
    sdnapi._ExtractImageData(data)


def testNonPNGImageRaises() -> None:
  """Raises Error when image format is not PNG."""
  # create a JPEG image
  img: Image.Image = Image.new('RGB', (64, 64), color=(255, 0, 0))
  buf = io.BytesIO()
  img.save(buf, format='JPEG')
  jpeg_b64: str = base64.b64encode(buf.getvalue()).decode('ascii')
  data: tbase.JSONDict = {
    'images': [jpeg_b64],
    'parameters': {'width': 64, 'height': 64},
  }
  with pytest.raises(sdnapi.Error, match='Expected PNG image'):
    sdnapi._ExtractImageData(data)


def testImageSizeMismatchRaises() -> None:
  """Raises Error when image dimensions don't match parameters."""
  data: tbase.JSONDict = {
    'images': [_B64PNG(64, 64)],
    'parameters': {'width': 128, 'height': 128},
  }
  with pytest.raises(sdnapi.Error, match='Expected image of size'):
    sdnapi._ExtractImageData(data)


# ─── _Call ────────────────────────────────────────────────────────────────────


def testCallSuccess() -> None:
  """_Call returns JSON response on success."""
  mock_method = mock.Mock(return_value=_MockCallResponse({'ok': True}))
  mock_method.__name__ = 'post'
  result: tbase.JSONValue = sdnapi._Call(mock_method, 'http://localhost:7860', '/test', {'data': 1})
  assert result == {'ok': True}


def testCallBadStatusRaises() -> None:
  """_Call raises Error on non-200 status."""
  mock_method = mock.Mock(return_value=_MockCallResponse('err', status=500))
  mock_method.__name__ = 'post'
  with pytest.raises(sdnapi.Error, match='Status 500'):
    sdnapi._Call(mock_method, 'http://localhost:7860', '/test')


def testCallConnectionErrorRaises() -> None:
  """_Call raises Error on connection failure."""
  mock_method = mock.Mock(side_effect=requests.exceptions.ConnectionError('refused'))
  mock_method.__name__ = 'post'
  with pytest.raises(sdnapi.Error, match='Failed to connect'):
    sdnapi._Call(mock_method, 'http://localhost:7860', '/test')


def testCallGenericExceptionRaises() -> None:
  """_Call raises Error on unexpected exception."""
  mock_method = mock.Mock(side_effect=RuntimeError('unexpected'))
  mock_method.__name__ = 'post'
  with pytest.raises(sdnapi.Error, match='Error calling SDNext API'):
    sdnapi._Call(mock_method, 'http://localhost:7860', '/test')


def testCallNoPayload() -> None:
  """_Call works with None payload."""
  mock_method = mock.Mock(return_value=_MockCallResponse({'ok': True}))
  mock_method.__name__ = 'get'
  result: tbase.JSONValue = sdnapi._Call(mock_method, 'http://localhost:7860', '/test')
  assert result == {'ok': True}
  mock_method.assert_called_once_with(
    'http://localhost:7860/test',
    json=None,
    timeout=300,
    verify=False,
  )
