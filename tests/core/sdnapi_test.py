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
from PIL import Image, PngImagePlugin
from transcrypto.utils import base as tbase

from transnext import __version__ as _transnext_version
from transnext.core import base, db, sdnapi

# --- helpers ---


def _MakePNG(
  width: int = 64,
  height: int = 64,
  info_text: str = 'some info text',
) -> bytes:
  """Create a minimal valid PNG in memory with embedded info text."""  # noqa: DOC201
  img: Image.Image = Image.new('RGBA', (width, height), color=(255, 0, 0, 255))
  png_info = PngImagePlugin.PngInfo()
  png_info.add_text('parameters', info_text)
  buf = io.BytesIO()
  img.save(buf, format='PNG', pnginfo=png_info)
  return buf.getvalue()


def _B64PNG(
  width: int = 64,
  height: int = 64,
  info_text: str = 'some info text',
) -> str:
  """Create base64-encoded PNG string."""  # noqa: DOC201
  return base64.b64encode(_MakePNG(width, height, info_text)).decode('ascii')


def _MakeMeta(overrides: dict[str, object] | None = None) -> db.AIMetaType:
  """Create an AIMetaType with a valid fixed seed for testing."""  # noqa: DOC201
  defaults: dict[str, object] = {'seed': 42, 'model_hash': 'abc123'}
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
    path=path,
    model_type=db.ModelType.safetensors.value,
    function=db.ModelFunction.Model.value,
    metadata={},
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
  info_text: str = 'some info text',
) -> dict[str, object]:
  """Build a full valid Txt2Img API response dict."""  # noqa: DOC201
  return {
    'images': [_B64PNG(width, height, info_text)],
    'info': json.dumps(
      {
        'width': width,
        'height': height,
        'sd_model_checkpoint': 'test-model',
      }
    ),
    'parameters': {
      'width': width,
      'height': height,
      'sd_model_checkpoint': 'test-model',
    },
  }


def _MockAPI(url: str = 'http://localhost:7860', **kwargs: object) -> sdnapi.API:
  """Create an API instance with mocked ServerVersion."""  # noqa: DOC201
  with mock.patch.object(
    sdnapi.API,
    'ServerVersion',
    return_value=('abc123', '2026-01-01'),
  ):
    return sdnapi.API(url, **kwargs)  # type: ignore[arg-type]


# --- Endpoints ---


def testEndpointValues() -> None:
  """Endpoints have correct string values."""
  assert sdnapi.Endpoints.MODELS.value == 'sd-models'
  assert sdnapi.Endpoints.OPTIONS.value == 'options'
  assert sdnapi.Endpoints.RELOAD_CHECKPOINT.value == 'reload-checkpoint'
  assert sdnapi.Endpoints.TXT2IMG.value == 'txt2img'
  assert sdnapi.Endpoints.LORA.value == 'loras'
  assert sdnapi.Endpoints.SYSTEM_STATUS.value == 'system-info'


# --- APIVersions / APICalls ---


def testAPIVersionsValues() -> None:
  """APIVersions enum has expected paths."""
  assert sdnapi.APIVersions.V1.value == '/sdapi/v1/'
  assert sdnapi.APIVersions.V2.value == '/sdapi/v2/'


def testAPICallsValues() -> None:
  """APICalls enum covers all calls."""
  values: set[int] = {c.value for c in sdnapi.APICalls}
  assert values == {1, 2, 3, 4, 5, 6, 7}


def testAPICallMatrixComplete() -> None:
  """Every APICalls value has an entry in the call matrix."""
  for call in sdnapi.APICalls:
    assert call in sdnapi._API_CALL_MATRIX


# --- Error / APIConnectionError ---


def testErrorIsSubclass() -> None:
  """Error inherits from Exception."""  # noqa: DOC501
  with pytest.raises(sdnapi.Error):
    raise sdnapi.Error('test')
  assert issubclass(sdnapi.Error, Exception)


def testAPIConnectionErrorIsSubclass() -> None:
  """APIConnectionError inherits from Error and ConnectionError."""  # noqa: DOC501
  assert issubclass(sdnapi.APIConnectionError, sdnapi.Error)
  assert issubclass(sdnapi.APIConnectionError, ConnectionError)
  with pytest.raises(sdnapi.APIConnectionError):
    raise sdnapi.APIConnectionError('conn fail')


# --- API.__init__ ---


def testAPIInit() -> None:
  """API initializes with correct attributes after calling ServerVersion."""
  api: sdnapi.API = _MockAPI()
  assert api._api_url == 'http://localhost:7860'
  assert api._version == 'abc123'
  assert api._server_save_images is False


def testAPIInitWithSaveImages() -> None:
  """API initializes with server_save_images=True."""
  api: sdnapi.API = _MockAPI(server_save_images=True)
  assert api._server_save_images is True
  assert api._version == 'abc123'


def testAPIInitServerVersionFailRaises() -> None:
  """API init raises APIConnectionError when ServerVersion returns empty."""
  with (
    mock.patch.object(
      sdnapi.API,
      'ServerVersion',
      return_value=('', '2026-01-01'),
    ),
    pytest.raises(sdnapi.APIConnectionError, match='Failed to get version'),
  ):
    sdnapi.API('http://localhost:7860')


# --- API.version ---


def testVersionProperty() -> None:
  """Version property returns the stored version."""
  api: sdnapi.API = _MockAPI()
  assert api.version == 'abc123'


# --- API.ServerVersion ---


def testServerVersionValid() -> None:
  """ServerVersion returns (commit, updated) tuple."""
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(
    api,
    'Call',
    return_value={
      'version': {'app': 'sd.next', 'commit': '0eb4a98e0', 'updated': '2026-04-04'},
    },
  ):
    commit, updated = api.ServerVersion()
  assert commit == '0eb4a98e0'
  assert updated == '2026-04-04'


def testServerVersionInvalidResponseRaises() -> None:
  """ServerVersion raises Error on invalid response."""
  api: sdnapi.API = _MockAPI()
  with (
    mock.patch.object(api, 'Call', return_value='not a dict'),
    pytest.raises(sdnapi.Error, match='Invalid system status'),
  ):
    api.ServerVersion()


def testServerVersionMissingFieldsRaises() -> None:
  """ServerVersion raises Error when version fields are missing."""
  api: sdnapi.API = _MockAPI()
  with (
    mock.patch.object(api, 'Call', return_value={'version': {'app': 'sd.next'}}),
    pytest.raises(sdnapi.Error, match='Invalid version info'),
  ):
    api.ServerVersion()


def testServerVersionWrongAppRaises() -> None:
  """ServerVersion raises Error when app name is unexpected."""
  api: sdnapi.API = _MockAPI()
  with (
    mock.patch.object(
      api,
      'Call',
      return_value={
        'version': {'app': 'wrong-app', 'commit': 'x', 'updated': 'y'},
      },
    ),
    pytest.raises(sdnapi.Error, match='Unexpected app'),
  ):
    api.ServerVersion()


# --- API.Call ---


def testCallRoutesToCorrectEndpoint() -> None:
  """Call routes APICalls to correct HTTP method and endpoint."""
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(sdnapi, '_Call', return_value=[{'model': 'data'}]) as mock_call:
    result: tbase.JSONValue = api.Call(sdnapi.APICalls.MODELS)
  assert isinstance(result, list)
  mock_call.assert_called_once_with(
    requests.get,
    'http://localhost:7860',
    '/sdapi/v2/sd-models',
    None,
  )


def testCallTxt2ImgWithPayload() -> None:
  """Call passes payload through to _Call."""
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(sdnapi, '_Call', return_value={'result': 'ok'}) as mock_call:
    result: tbase.JSONValue = api.Call(sdnapi.APICalls.TXT2IMG, {'prompt': 'hello'})
  assert result == {'result': 'ok'}
  args = mock_call.call_args
  assert args[0][3] == {'prompt': 'hello'}


# --- API.options ---


def testOptionsGetter() -> None:
  """Options property returns dict from API."""
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call', return_value={'sd_model_checkpoint': 'foo'}):
    opts: tbase.JSONDict = api.options
  assert opts == {'sd_model_checkpoint': 'foo'}


def testOptionsGetterInvalidRaises() -> None:
  """Options property raises Error on non-dict response."""
  api: sdnapi.API = _MockAPI()
  with (
    mock.patch.object(api, 'Call', return_value='not a dict'),
    pytest.raises(sdnapi.Error, match='Invalid options response'),
  ):
    _ = api.options


def testOptionsSetter() -> None:
  """Options setter calls SET_OPTIONS."""
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call') as mock_call:
    api.options = {'sd_model_checkpoint': 'bar'}
  mock_call.assert_called_once_with(
    sdnapi.APICalls.SET_OPTIONS,
    {'sd_model_checkpoint': 'bar'},
  )


# --- API.GetModels ---


def testGetModelsValid() -> None:
  """GetModels parses valid response with new dict format."""
  api: sdnapi.API = _MockAPI()
  api_response: tbase.JSONValue = {
    'items': [
      {
        'sha256': 'hash1',
        'model_name': 'my-model',
        'title': 'my-model [hash1]',
        'filename': '/tmp/model.safetensors',  # noqa: S108
        'type': 'safetensors',
      },
    ],
  }
  with (
    mock.patch.object(api, 'Call', return_value=api_response),
    mock.patch('pathlib.Path.exists', return_value=True),
  ):
    models: list[db.AIModelType] = api.GetModels()
  assert len(models) == 1
  assert models[0] == {
    'hash': 'hash1',
    'name': 'my-model',
    'alias': 'my-model [hash1]',
    'path': '/tmp/model.safetensors',  # noqa: S108
    'model_type': 'safetensors',
    'function': db.ModelFunction.Model.value,
    'metadata': {
      'sha256': 'hash1',
      'model_name': 'my-model',
      'title': 'my-model [hash1]',
      'filename': '/tmp/model.safetensors',  # noqa: S108
      'type': 'safetensors',
    },
    'description': None,
  }


def testGetModelsEmptyResponseRaises() -> None:
  """GetModels raises Error on empty response."""
  api: sdnapi.API = _MockAPI()
  with (
    mock.patch.object(api, 'Call', return_value={}),
    pytest.raises(sdnapi.Error, match='Invalid models response'),
  ):
    api.GetModels()


def testGetModelsNotDictRaises() -> None:
  """GetModels raises Error on non-dict response."""
  api: sdnapi.API = _MockAPI()
  with (
    mock.patch.object(api, 'Call', return_value=[]),
    pytest.raises(sdnapi.Error, match='Invalid models response'),
  ):
    api.GetModels()


def testGetModelsInvalidEntryRaises() -> None:
  """GetModels raises Error on non-dict entry in items."""
  api: sdnapi.API = _MockAPI()
  with (
    mock.patch.object(api, 'Call', return_value={'items': ['not-a-dict']}),
    pytest.raises(sdnapi.Error, match='Invalid model entry'),
  ):
    api.GetModels()


def testGetModelsFileNotFoundRaises() -> None:
  """GetModels raises Error when model file does not exist."""
  api: sdnapi.API = _MockAPI()
  api_response: tbase.JSONValue = {
    'items': [
      {
        'sha256': 'hash1',
        'model_name': 'my-model',
        'title': 'my-model',
        'filename': '/nonexistent/model.safetensors',
        'type': 'safetensors',
      },
    ],
  }
  with (
    mock.patch.object(api, 'Call', return_value=api_response),
    mock.patch('pathlib.Path.exists', return_value=False),
    pytest.raises(sdnapi.Error, match='Model file not found'),
  ):
    api.GetModels()


def testGetModelsMissingNameRaises() -> None:
  """GetModels raises Error when model name is missing."""
  api: sdnapi.API = _MockAPI()
  api_response: tbase.JSONValue = {
    'items': [
      {
        'sha256': 'hash1',
        'model_name': '',
        'title': '',
        'filename': '/tmp/model.safetensors',  # noqa: S108
        'type': 'safetensors',
      },
    ],
  }
  with (
    mock.patch.object(api, 'Call', return_value=api_response),
    mock.patch('pathlib.Path.exists', return_value=True),
    pytest.raises(sdnapi.Error, match='Missing model name'),
  ):
    api.GetModels()


# --- API.GetLora ---


def testGetLoraValid() -> None:
  """GetLora parses valid response."""
  api: sdnapi.API = _MockAPI()
  api_response: tbase.JSONValue = [
    {
      'name': 'my-lora',
      'alias': 'lora-alias',
      'path': '/tmp/lora.safetensors',  # noqa: S108
      'metadata': {},
    },
  ]
  with (
    mock.patch.object(api, 'Call', return_value=api_response),
    mock.patch('pathlib.Path.exists', return_value=True),
  ):
    loras: list[db.AIModelType] = api.GetLora()
  assert len(loras) == 1
  assert loras[0] == {
    'hash': '',
    'name': 'my-lora',
    'alias': 'lora-alias',
    'path': '/tmp/lora.safetensors',  # noqa: S108
    'model_type': 'safetensors',
    'function': db.ModelFunction.Lora.value,
    'metadata': {},
    'description': None,
  }


def testGetLoraLycorisDetected() -> None:
  """GetLora sets function to Lycoris when ss_network_module contains lyco."""
  api: sdnapi.API = _MockAPI()
  api_response: tbase.JSONValue = [
    {
      'name': 'lyco-model',
      'alias': 'lyco-alias',
      'path': '/tmp/lora.safetensors',  # noqa: S108
      'metadata': {'ss_network_module': 'lycoris.kohya'},
    },
  ]
  with (
    mock.patch.object(api, 'Call', return_value=api_response),
    mock.patch('pathlib.Path.exists', return_value=True),
  ):
    loras: list[db.AIModelType] = api.GetLora()
  assert loras[0]['function'] == db.ModelFunction.Lycoris.value


def testGetLoraEmptyRaises() -> None:
  """GetLora raises Error on empty response."""
  api: sdnapi.API = _MockAPI()
  with (
    mock.patch.object(api, 'Call', return_value=[]),
    pytest.raises(sdnapi.Error, match='Invalid models response'),
  ):
    api.GetLora()


def testGetLoraNotListRaises() -> None:
  """GetLora raises Error on non-list response."""
  api: sdnapi.API = _MockAPI()
  with (
    mock.patch.object(api, 'Call', return_value={'foo': 'bar'}),
    pytest.raises(sdnapi.Error, match='Invalid models response'),
  ):
    api.GetLora()


def testGetLoraFileNotFoundRaises() -> None:
  """GetLora raises Error when lora file does not exist."""
  api: sdnapi.API = _MockAPI()
  api_response: tbase.JSONValue = [
    {
      'name': 'missing-lora',
      'alias': 'alias',
      'path': '/nonexistent/lora.safetensors',
      'metadata': {},
    },
  ]
  with (
    mock.patch.object(api, 'Call', return_value=api_response),
    mock.patch('pathlib.Path.exists', return_value=False),
    pytest.raises(sdnapi.Error, match='Model file not found'),
  ):
    api.GetLora()


def testGetLoraMissingNameRaises() -> None:
  """GetLora raises Error when lora name is missing."""
  api: sdnapi.API = _MockAPI()
  api_response: tbase.JSONValue = [
    {
      'name': '',
      'alias': 'alias',
      'path': '/tmp/lora.safetensors',  # noqa: S108
      'metadata': {},
    },
  ]
  with (
    mock.patch.object(api, 'Call', return_value=api_response),
    mock.patch('pathlib.Path.exists', return_value=True),
    pytest.raises(sdnapi.Error, match='Missing model name'),
  ):
    api.GetLora()


# --- API.Txt2Img ---


def testTxt2ImgSuccess() -> None:
  """Txt2Img returns valid DBImageType and image bytes."""
  meta: db.AIMetaType = _MakeMeta({'width': 64, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call') as mock_call:
    # Call sequence: options get, options set, txt2img
    mock_call.side_effect = [
      {'sd_checkpoint_hash': 'abc123', 'sd_model_checkpoint': 'test-model'},
      None,  # options set
      _Txt2ImgAPIData(64, 64),  # txt2img
    ]
    db_img, img_data = api.Txt2Img(model, meta)
  assert len(img_data) > 0
  # pop variable fields and check them separately
  assert db_img.pop('created_at') > 1000000  # type: ignore[misc]
  assert db_img == {
    'path': None,
    'alt_path': [],
    'width': 64,
    'height': 64,
    'size': 217,
    'hash': '75d704ad7cf296cc1f21cfea8dfb8a1b829559b88ba832885d8fa68d080659f7',
    'raw_hash': '0bcc07de1631aa0395013f35790f719bd754f61e4bb2847a86fdc2365d3d8d63',
    'format': 'PNG',
    'origin': db.ImageOrigin.TransNext.value,
    'version': f'abc123/{_transnext_version}',
    'info': 'some info text',
    'ai_meta': _MakeMeta({'width': 64, 'height': 64}),
    'sd_info': {
      'width': 64,
      'height': 64,
      'sd_model_checkpoint': 'test-model',
    },
    'sd_params': {
      'width': 64,
      'height': 64,
      'sd_model_checkpoint': 'test-model',
    },
    'parse_errors': {},
  }


def testTxt2ImgModelChange() -> None:
  """Txt2Img triggers model reload when model changes."""
  meta: db.AIMetaType = _MakeMeta({'width': 64, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [
      {'sd_checkpoint_hash': 'different_hash', 'sd_model_checkpoint': 'old-model'},
      None,  # options set
      None,  # reload checkpoint
      _Txt2ImgAPIData(64, 64),  # txt2img
    ]
    db_img, _img_data = api.Txt2Img(model, meta)
  assert db_img['width'] == 64
  assert mock_call.call_count == 4


def testTxt2ImgWithDirRoot(tmp_path: mock.Mock) -> None:
  """Txt2Img saves image to disk when dir_root is provided."""
  meta: db.AIMetaType = _MakeMeta({'width': 64, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [
      {'sd_checkpoint_hash': 'abc123', 'sd_model_checkpoint': 'test-model'},
      None,
      _Txt2ImgAPIData(64, 64),
    ]
    db_img, _img_data = api.Txt2Img(model, meta, dir_root=tmp_path)
  assert db_img['path'] is not None
  assert db_img['path'].endswith('.png')


def testTxt2ImgHashMismatchRaises() -> None:
  """Txt2Img raises Error when model hash does not match meta."""
  meta: db.AIMetaType = _MakeMeta({'model_hash': 'expected'})
  model: db.AIModelType = _MakeModel(h='different')
  api: sdnapi.API = _MockAPI()
  with pytest.raises(sdnapi.Error, match='Model hash mismatch'):
    api.Txt2Img(model, meta)


def testTxt2ImgIncompatibleFunctionRaises() -> None:
  """Txt2Img raises Error when model function is not Model."""
  meta: db.AIMetaType = _MakeMeta()
  model: db.AIModelType = _MakeModel()
  model['function'] = db.ModelFunction.Lora.value
  api: sdnapi.API = _MockAPI()
  with pytest.raises(sdnapi.Error, match='Incompatible model'):
    api.Txt2Img(model, meta)


def testTxt2ImgImg2ImgRaises() -> None:
  """Txt2Img raises Error when meta has img2img set."""
  meta: db.AIMetaType = _MakeMeta(
    {
      'img2img': db.AIImg2ImgType(input_hash='x', denoising=50),
    }
  )
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  with pytest.raises(sdnapi.Error, match='img2img is not supported'):
    api.Txt2Img(model, meta)


def testTxt2ImgInvalidDimensionsRaises() -> None:
  """Txt2Img raises Error for invalid image dimensions (not multiple of 16)."""
  meta: db.AIMetaType = _MakeMeta({'width': 15, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [
      {'sd_checkpoint_hash': 'abc123', 'sd_model_checkpoint': 'test-model'},
      None,
    ]
    with pytest.raises(sdnapi.Error, match='Invalid image dimensions'):
      api.Txt2Img(model, meta)


def testTxt2ImgZeroDimensionsRaises() -> None:
  """Txt2Img raises Error for zero-size dimensions."""
  meta: db.AIMetaType = _MakeMeta({'width': 0, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [
      {'sd_checkpoint_hash': 'abc123', 'sd_model_checkpoint': 'test-model'},
      None,
    ]
    with pytest.raises(sdnapi.Error, match='Invalid image dimensions'):
      api.Txt2Img(model, meta)


def testTxt2ImgA1111SamplerRaises() -> None:
  """Txt2Img raises Error when sampler is A1111-only."""
  sampler_name: str = next(iter(base.SamplerA1111)).value
  meta: db.AIMetaType = _MakeMeta(
    {'width': 64, 'height': 64, 'sampler': sampler_name},
  )
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [
      {'sd_checkpoint_hash': 'abc123', 'sd_model_checkpoint': 'test-model'},
      None,
    ]
    with pytest.raises(sdnapi.Error, match='not supported by SDNext'):
      api.Txt2Img(model, meta)


def testTxt2ImgInvalidResponseRaises() -> None:
  """Txt2Img raises Error when response is missing expected fields."""
  meta: db.AIMetaType = _MakeMeta({'width': 64, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [
      {'sd_checkpoint_hash': 'abc123', 'sd_model_checkpoint': 'test-model'},
      None,
      {'no-info': True},
    ]
    with pytest.raises(sdnapi.Error, match='Invalid image metadata'):
      api.Txt2Img(model, meta)


def testTxt2ImgParametersSizeMismatchRaises() -> None:
  """Txt2Img raises Error when parameters size does not match meta."""
  meta: db.AIMetaType = _MakeMeta({'width': 64, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  bad_data: dict[str, object] = {
    'images': [_B64PNG(64, 64)],
    'info': json.dumps(
      {
        'width': 64,
        'height': 64,
        'sd_model_checkpoint': 'test-model',
      }
    ),
    'parameters': {
      'width': 128,
      'height': 64,
      'sd_model_checkpoint': 'test-model',
    },
  }
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [
      {'sd_checkpoint_hash': 'abc123', 'sd_model_checkpoint': 'test-model'},
      None,
      bad_data,
    ]
    with pytest.raises(sdnapi.Error, match='Expected image of size'):
      api.Txt2Img(model, meta)


def testTxt2ImgNoDirRootNoSave() -> None:
  """Txt2Img sets path to None when no dir_root is given."""
  meta: db.AIMetaType = _MakeMeta({'width': 64, 'height': 64})
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [
      {'sd_checkpoint_hash': 'abc123', 'sd_model_checkpoint': 'test-model'},
      None,
      _Txt2ImgAPIData(64, 64),
    ]
    db_img, _img_data = api.Txt2Img(model, meta)
  assert db_img['path'] is None


# --- _ExtractImageData ---


def testExtractValidPNG() -> None:
  """Extracts valid PNG image data with info text."""
  data: tbase.JSONDict = {
    'images': [_B64PNG(64, 64)],
    'parameters': {'width': 64, 'height': 64},
  }
  img_data, raw_hash, info_text = sdnapi._ExtractImageData(data)
  assert len(img_data) > 0
  assert len(raw_hash) > 0
  assert info_text == 'some info text'


def testExtractMissingImagesRaises() -> None:
  """Raises Error when images key is missing."""
  with pytest.raises(sdnapi.Error, match='No images received'):
    sdnapi._ExtractImageData({'parameters': {}})


def testExtractMissingParametersRaises() -> None:
  """Raises Error when parameters key is missing."""
  with pytest.raises(sdnapi.Error, match='Image metadata not received'):
    sdnapi._ExtractImageData({'images': [_B64PNG()]})


def testExtractWrongImageCountRaises() -> None:
  """Raises Error when image count is not exactly 1."""
  data: tbase.JSONDict = {
    'images': [_B64PNG(), _B64PNG()],
    'parameters': {'width': 64, 'height': 64},
  }
  with pytest.raises(sdnapi.Error, match='Expected exactly 1 image'):
    sdnapi._ExtractImageData(data)


def testExtractEmptyImagesListRaises() -> None:
  """Raises Error when images list is empty."""
  data: tbase.JSONDict = {
    'images': [],
    'parameters': {'width': 64, 'height': 64},
  }
  with pytest.raises(sdnapi.Error, match='Expected exactly 1 image'):
    sdnapi._ExtractImageData(data)


def testExtractCommaInBase64Raises() -> None:
  """Raises Error when base64 string contains comma."""
  data: tbase.JSONDict = {
    'images': ['data:image/png;base64,' + _B64PNG()],
    'parameters': {'width': 64, 'height': 64},
  }
  with pytest.raises(sdnapi.Error, match='Unexpected comma'):
    sdnapi._ExtractImageData(data)


def testExtractEmptyImageDataRaises() -> None:
  """Raises Error when decoded image data is empty."""
  data: tbase.JSONDict = {
    'images': [''],
    'parameters': {'width': 64, 'height': 64},
  }
  with pytest.raises(sdnapi.Error, match='Image data empty'):
    sdnapi._ExtractImageData(data)


def testExtractNonPNGImageRaises() -> None:
  """Raises Error when image format is not PNG."""
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


def testExtractImageSizeMismatchRaises() -> None:
  """Raises Error when image dimensions do not match parameters."""
  data: tbase.JSONDict = {
    'images': [_B64PNG(64, 64)],
    'parameters': {'width': 128, 'height': 128},
  }
  with pytest.raises(sdnapi.Error, match='Expected image of size'):
    sdnapi._ExtractImageData(data)


def testExtractNoInfoTextRaises() -> None:
  """Raises Error when PNG has no info text metadata."""
  img: Image.Image = Image.new('RGBA', (64, 64), color=(0, 0, 255, 255))
  buf = io.BytesIO()
  img.save(buf, format='PNG')
  b64: str = base64.b64encode(buf.getvalue()).decode('ascii')
  data: tbase.JSONDict = {
    'images': [b64],
    'parameters': {'width': 64, 'height': 64},
  }
  with pytest.raises(sdnapi.Error, match='No info text found'):
    sdnapi._ExtractImageData(data)


# --- _Call ---


def testCallSuccess() -> None:
  """_Call returns JSON response on success."""
  mock_method = mock.Mock(return_value=_MockCallResponse({'ok': True}))
  mock_method.__name__ = 'post'
  result: tbase.JSONValue = sdnapi._Call(
    mock_method,
    'http://localhost:7860',
    '/test',
    {'data': 1},
  )
  assert result == {'ok': True}


def testCallBadStatusRaises() -> None:
  """_Call raises Error on non-200 status."""
  mock_method = mock.Mock(return_value=_MockCallResponse('err', status=500))
  mock_method.__name__ = 'post'
  with pytest.raises(sdnapi.Error, match='Status 500'):
    sdnapi._Call(mock_method, 'http://localhost:7860', '/test')


def testCallConnectionErrorRaises() -> None:
  """_Call raises APIConnectionError on connection failure."""
  mock_method = mock.Mock(
    side_effect=requests.exceptions.ConnectionError('refused'),
  )
  mock_method.__name__ = 'post'
  with pytest.raises(sdnapi.APIConnectionError, match='Failed to connect'):
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
  result: tbase.JSONValue = sdnapi._Call(
    mock_method,
    'http://localhost:7860',
    '/test',
  )
  assert result == {'ok': True}
  mock_method.assert_called_once_with(
    'http://localhost:7860/test',
    json=None,
    timeout=300,
    verify=False,
  )
