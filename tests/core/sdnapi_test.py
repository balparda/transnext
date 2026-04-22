# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: transnext.core.sdnapi module."""

from __future__ import annotations

import base64
import io
import json
import pathlib
from unittest import mock

import pytest
import requests
from PIL import Image
from transcrypto.utils import base as tbase

from transnext.core import base, db, sdnapi

# --- real test PNG (stable on-disk image, version-independent) ---

_TEST_PNG_PATH: pathlib.Path = (
  pathlib.Path(__file__).parent.parent
  / 'data'
  / 'images'
  / '6db2ba7302bd-20260417102202-e6bb9ea8-80-40-512-256-666-db088cdca097.png'
)
_TEST_PNG_BYTES: bytes = _TEST_PNG_PATH.read_bytes()
_TEST_PNG_B64: str = base64.b64encode(_TEST_PNG_BYTES).decode('ascii')
_TEST_PNG_WIDTH: int = 512
_TEST_PNG_HEIGHT: int = 256
_TEST_PNG_SIZE: int = len(_TEST_PNG_BYTES)
_TEST_PNG_HASH: str = 'db088cdca09796cadee02ec7eef8dd8e2227490a5afbb353461ab34d1ddbd8b9'
_TEST_PNG_RAW_HASH: str = 'dcf3c5cacfddc7b23f3314c680263c03ace7fedc456f6daa1907d5f7ed30af2e'
_TEST_PNG_INFO_TEXT: str = (
  'dark knight in moody rain\n'
  'Negative prompt: batman, comic, text\n'
  'Steps: 40, Size: 512x256, Sampler: DPM SDE, Scheduler: DPMSolverSDEScheduler, Seed: 666, '
  'CFG scale: 8.0, CFG rescale: 0.8, CFG end: 0.9, Clip skip: 2, App: SD.Next, '
  'Version: 0eb4a98, Parser: a1111, Pipeline: StableDiffusionXLPipeline, '
  'Operations: txt2img, Model: SDXL_00_XLB_v10VAEFix, Model hash: e6bb9ea85b, '
  'Variation seed: 999, Variation strength: 0.3, Sampler spacing: linspace, '
  'Sampler sigma: karras, Sampler type: epsilon, Sampler beta schedule: linear, '
  'FreeU: b1=1.1 b2=1.15 s1=0.7 s2=0.6'
)

# --- helpers ---


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
    autov3=None,
    path=path,
    model_type=db.ModelType.safetensors.value,
    function=db.ModelFunction.Model.value,
    metadata={},
    sidecar=None,
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


def _Txt2ImgAPIData() -> dict[str, object]:
  """Build a full valid Txt2Img API response dict using the real test PNG."""  # noqa: DOC201
  return {
    'images': [_TEST_PNG_B64],
    'info': json.dumps(
      {
        'width': _TEST_PNG_WIDTH,
        'height': _TEST_PNG_HEIGHT,
        'sd_model_checkpoint': 'test-model',
      }
    ),
    'parameters': {
      'width': _TEST_PNG_WIDTH,
      'height': _TEST_PNG_HEIGHT,
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


# --- APIVersions / APICalls ---


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
    record_list=None,
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
    'autov3': None,
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
    'sidecar': None,
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
    'autov3': None,
    'path': '/tmp/lora.safetensors',  # noqa: S108
    'model_type': 'safetensors',
    'function': db.ModelFunction.Lora.value,
    'metadata': {},
    'sidecar': None,
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
  meta: db.AIMetaType = _MakeMeta({'width': _TEST_PNG_WIDTH, 'height': _TEST_PNG_HEIGHT})
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call') as mock_call:
    # Call sequence: options get, options set, txt2img
    mock_call.side_effect = [
      {'sd_checkpoint_hash': 'abc123', 'sd_model_checkpoint': 'test-model'},
      None,  # options set
      _Txt2ImgAPIData(),  # txt2img
    ]
    db_img, img_data = api.Txt2Img(model, meta)
  assert len(img_data) == _TEST_PNG_SIZE
  # the return uses the new paths-based structure
  assert db_img['paths'][''].pop('created_at') > 1000000  # type: ignore[misc]
  assert db_img == {
    'format': 'PNG',
    'hash': _TEST_PNG_HASH,
    'height': _TEST_PNG_HEIGHT,
    'info': _TEST_PNG_INFO_TEXT,
    'paths': {
      '': {
        'ai_meta': {
          'cfg_end': 8,
          'cfg_rescale': 0,
          'cfg_scale': 60,
          'cfg_skip': None,
          'clip_skip': 10,
          'freeu': {
            'b1': 105,
            'b2': 110,
            's1': 75,
            's2': 65,
          },
          'height': _TEST_PNG_HEIGHT,
          'img2img': None,
          'lora': None,
          'model_hash': 'abc123',
          'n_embeddings': None,
          'negative': None,
          'ngms': None,
          'p_embeddings': None,
          'parser': 'a1111',
          'positive': '',
          'sampler': 'DPM++ SDE',
          'sch_beta': None,
          'sch_sigma': None,
          'sch_spacing': None,
          'sch_type': None,
          'seed': 42,
          'steps': 20,
          'v_seed': None,
          'width': _TEST_PNG_WIDTH,
        },
        'main': False,
        'origin': db.ImageOrigin.TransNext.value,
        'parse_errors': None,
        'sd_info': {
          'height': _TEST_PNG_HEIGHT,
          'sd_model_checkpoint': 'test-model',
          'width': _TEST_PNG_WIDTH,
        },
        'sd_params': {
          'height': _TEST_PNG_HEIGHT,
          'sd_model_checkpoint': 'test-model',
          'width': _TEST_PNG_WIDTH,
        },
        'version': 'abc123/1.1.0',
      },
    },
    'raw_hash': _TEST_PNG_RAW_HASH,
    'size': _TEST_PNG_SIZE,
    'width': _TEST_PNG_WIDTH,
  }


def testTxt2ImgModelChange() -> None:
  """Txt2Img triggers model reload when model changes."""
  meta: db.AIMetaType = _MakeMeta({'width': _TEST_PNG_WIDTH, 'height': _TEST_PNG_HEIGHT})
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [
      {'sd_checkpoint_hash': 'different_hash', 'sd_model_checkpoint': 'old-model'},
      None,  # options set
      None,  # reload checkpoint
      _Txt2ImgAPIData(),  # txt2img
    ]
    db_img, _img_data = api.Txt2Img(model, meta)
  assert db_img['width'] == _TEST_PNG_WIDTH
  assert mock_call.call_count == 4


def testTxt2ImgWithDirRoot(tmp_path: mock.Mock) -> None:
  """Txt2Img saves image to disk when dir_root is provided."""
  meta: db.AIMetaType = _MakeMeta({'width': _TEST_PNG_WIDTH, 'height': _TEST_PNG_HEIGHT})
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [
      {'sd_checkpoint_hash': 'abc123', 'sd_model_checkpoint': 'test-model'},
      None,
      _Txt2ImgAPIData(),
    ]
    db_img, _img_data = api.Txt2Img(model, meta, dir_root=tmp_path)
  saved_path: str = next(iter(db_img['paths']))
  assert saved_path
  assert saved_path.endswith('.png')


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
  meta: db.AIMetaType = _MakeMeta({'width': _TEST_PNG_WIDTH, 'height': _TEST_PNG_HEIGHT})
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  bad_data: dict[str, object] = {
    'images': [_TEST_PNG_B64],
    'info': json.dumps(
      {
        'width': _TEST_PNG_WIDTH,
        'height': _TEST_PNG_HEIGHT,
        'sd_model_checkpoint': 'test-model',
      }
    ),
    'parameters': {
      'width': _TEST_PNG_WIDTH * 2,  # mismatch: not what meta requested
      'height': _TEST_PNG_HEIGHT,
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
  meta: db.AIMetaType = _MakeMeta({'width': _TEST_PNG_WIDTH, 'height': _TEST_PNG_HEIGHT})
  model: db.AIModelType = _MakeModel()
  api: sdnapi.API = _MockAPI()
  with mock.patch.object(api, 'Call') as mock_call:
    mock_call.side_effect = [
      {'sd_checkpoint_hash': 'abc123', 'sd_model_checkpoint': 'test-model'},
      None,
      _Txt2ImgAPIData(),
    ]
    db_img, _img_data = api.Txt2Img(model, meta)
  assert '' in db_img['paths']  # empty string key when no dir_root


# --- _ExtractImageData ---


def testExtractValidPNG() -> None:
  """Extracts valid PNG image data with info text."""
  data: tbase.JSONDict = {
    'images': [_TEST_PNG_B64],
    'parameters': {'width': _TEST_PNG_WIDTH, 'height': _TEST_PNG_HEIGHT},
  }
  img_data, raw_hash, info_text = sdnapi._ExtractImageData(data)
  assert len(img_data) == _TEST_PNG_SIZE
  assert raw_hash == _TEST_PNG_RAW_HASH
  assert info_text == _TEST_PNG_INFO_TEXT


def testExtractMissingImagesRaises() -> None:
  """Raises Error when images key is missing."""
  with pytest.raises(sdnapi.Error, match='No images received'):
    sdnapi._ExtractImageData({'parameters': {}})


def testExtractMissingParametersRaises() -> None:
  """Raises Error when parameters key is missing."""
  with pytest.raises(sdnapi.Error, match='Image metadata not received'):
    sdnapi._ExtractImageData({'images': [_TEST_PNG_B64]})


def testExtractWrongImageCountRaises() -> None:
  """Raises Error when image count is not exactly 1."""
  data: tbase.JSONDict = {
    'images': [_TEST_PNG_B64, _TEST_PNG_B64],
    'parameters': {'width': _TEST_PNG_WIDTH, 'height': _TEST_PNG_HEIGHT},
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
    'images': ['data:image/png;base64,' + _TEST_PNG_B64],
    'parameters': {'width': _TEST_PNG_WIDTH, 'height': _TEST_PNG_HEIGHT},
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
    'images': [_TEST_PNG_B64],
    'parameters': {'width': 128, 'height': 128},  # real PNG is 512x256, not 128x128
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
