# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""TransNext core SDNext API module.

Look at <http://127.0.0.1:7860/docs> on a running server to see the API calls and details.
"""

from __future__ import annotations

import base64
import copy
import enum
import json
import logging
import pathlib
import time
from collections import abc
from typing import cast

import requests
import urllib3
from transcrypto.core import hashes
from transcrypto.utils import base as tbase
from transcrypto.utils import human, timer

from transnext import __version__
from transnext.core import base, db

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_SUCCESS_STATUSES: set[int] = {200}

_APP_NAME: str = 'sd.next'


class APIVersions(enum.Enum):
  """SDNext API endpoints."""

  V1 = '/sdapi/v1/'
  V2 = '/sdapi/v2/'


class Endpoints(enum.Enum):
  """SDNext API endpoints."""

  SYSTEM_STATUS = 'system-info'
  MODELS = 'sd-models'
  LORA = 'loras'
  EMBEDDINGS = 'embeddings'
  OPTIONS = 'options'
  COMMAND_FLAGS = 'cmd-flags'
  RELOAD_CHECKPOINT = 'reload-checkpoint'
  TXT2IMG = 'txt2img'


class APICalls(enum.Enum):
  """SDNext API calls."""

  STATUS = 1
  MODELS = 2
  LORA = 3
  EMBEDDINGS = 4
  READ_OPTIONS = 5
  SET_OPTIONS = 6
  COMMAND_FLAGS = 7
  RELOAD_CHECKPOINT = 8
  TXT2IMG = 9


_API_CALL_MATRIX: dict[
  APICalls, tuple[APIVersions, Endpoints, abc.Callable[..., requests.Response]]
] = {
  # see <http://127.0.0.1:7860/docs> for details on the API
  APICalls.STATUS: (APIVersions.V2, Endpoints.SYSTEM_STATUS, requests.get),
  APICalls.MODELS: (APIVersions.V2, Endpoints.MODELS, requests.get),
  APICalls.LORA: (APIVersions.V1, Endpoints.LORA, requests.get),
  APICalls.EMBEDDINGS: (APIVersions.V2, Endpoints.EMBEDDINGS, requests.get),
  APICalls.READ_OPTIONS: (APIVersions.V2, Endpoints.OPTIONS, requests.get),
  APICalls.SET_OPTIONS: (APIVersions.V2, Endpoints.OPTIONS, requests.post),
  APICalls.COMMAND_FLAGS: (APIVersions.V1, Endpoints.COMMAND_FLAGS, requests.get),
  APICalls.RELOAD_CHECKPOINT: (APIVersions.V1, Endpoints.RELOAD_CHECKPOINT, requests.post),
  APICalls.TXT2IMG: (APIVersions.V1, Endpoints.TXT2IMG, requests.post),
}


class Error(base.Error):
  """TransNext SDNext API exception."""


class APIConnectionError(Error, ConnectionError):
  """TransNext SDNext API connection exception."""


class API(db.APIProtocol):
  """SDNext API client."""

  def __init__(
    self, api_url: str, *, server_save_images: bool = False, record: bool = False
  ) -> None:
    """Initialize SDNext API client.

    Args:
      api_url: Base URL of the SDNext API, e.g. "http://127.0.0.1:5000"
      server_save_images: Whether if the server will save a copy of the images too (default False)
      record: Whether to record API calls and responses (default False)

    Raises:
      APIConnectionError: If there is a connection error to the SDNext API.

    """
    self._api_url: str = api_url
    self._server_save_images: bool = server_save_images
    self._record: bool = record
    self._call_record: list[tbase.JSONDict] = []
    commit, updated = self.ServerVersion()
    if not commit or not updated:
      raise APIConnectionError(f'Failed to get version from SDNext API: {commit!r}, {updated!r}')
    self._version: str = commit
    logging.info(
      f'API/{commit}/{updated} @ {self._api_url}{" + SAVE" if self._server_save_images else ""}'
    )
    if record:
      logging.warning('Recording of API calls and responses is enabled!')

  @property
  def version(self) -> str:
    """Get the SDNext API server version."""
    return self._version

  def Call(self, api_call: APICalls, payload: tbase.JSONDict | None = None) -> tbase.JSONValue:
    """Call SDNext API endpoint with given payload.

    Args:
      api_call: API call to make
      payload: JSON-serializable payload to send in the request body (default is None)

    Returns:
      The JSON response from the API as a dictionary.

    Raises:
      APIConnectionError: If there is a connection error to the SDNext API.

    """  # noqa: DOC502
    version, endpoint, method = _API_CALL_MATRIX[api_call]
    return _Call(
      method,
      self._api_url,
      version.value + endpoint.value,
      payload.copy() if payload else None,
      record_list=self._call_record if self._record else None,
    )

  def SaveRecordToFile(self, path: pathlib.Path) -> None:
    """Save recorded API calls and responses to a JSON file.

    Args:
      path: Path to the JSON file to save the record to.

    """
    if not self._record or not self._call_record:
      logging.error('No call recorded, nothing to save.')
      return
    with path.open('w', encoding='utf-8') as obj:
      json.dump(self._call_record, obj, indent=2)
    logging.warning(f'Saved API call record to {path}')

  def ServerVersion(self) -> tuple[str, str]:
    """Get SDNext API server version info.

    Schema:

    {
      'version': {
        'app': 'sd.next',
        'updated': '2026-04-04',
        'commit': '0eb4a98e0',
        'branch': 'master',
        'url': 'https://github.com/vladmandic/sdnext/tree/master',
        'kanvas': 'main',
        'ui': 'main',
      },
      'uptime': 'Tue Apr 14 12:28:21 2026',
      'timestamp': 'Tue Apr 14 14:04:16 2026',
      'platform': {
        'arch': 'arm64',
        'cpu': 'arm',
        'system': 'Darwin',
        'release': '25.3.0',
        'python': '3.12.13',
        'locale': "('en_IE', 'UTF-8')",
        'setuptools': '69.5.1',
        'docker': 'False',
      },
      'torch': '2.7.1',
      'gpu': {},
      'device': {
        'device': 'mps',
        'dtype': 'torch.float32',
        'dtype_vae': 'torch.float32',
        'dtype_unet': 'torch.float32',
      },
      'libs': {
        'torch': '2.7.1',
        'diffusers': '0.38.0.dev0',
        'transformers': '5.5.0.dev0',
        'gradio': '3.43.2',
        'accelerate': '1.13.0',
      },
      'backend': 'DIFFUSERS',
      'pipeline': 'True',
      'cross_attention': 'Scaled-Dot-Product',
      'flags': [],
    }

    Returns:
      A tuple containing the server version hash and updated timestamp, example:
      ('0eb4a98e0', '2026-04-04')

    Raises:
      Error: If there is an error with the API call or if the response is invalid.

    """
    info: tbase.JSONValue = self.Call(APICalls.STATUS)
    if not isinstance(info, dict) or 'version' not in info:
      raise Error(f'Invalid system status response from SDNext API: {info}')
    version: dict[str, str] = cast('dict[str, str]', info['version'])
    if (
      not isinstance(version, dict)  # pyright: ignore[reportUnnecessaryIsInstance]
      or 'app' not in version
      or 'updated' not in version
      or 'commit' not in version
    ):
      raise Error(f'Invalid version info in system status response from SDNext API: {info}')
    if version['app'] != _APP_NAME:
      raise Error(f'Unexpected app in version info from SDNext API: {version}')
    return (version['commit'], version['updated'])

  @property
  def options(self) -> tbase.JSONDict:
    """Get current SDNext API options. Costs an API call.

    Schema:

    {
      'sd_model_checkpoint': 'SDXL_00_XLB_v10VAEFix',                 ==> CONTEMPLATED / INDIRECT
      'sd_checkpoint_hash': 'e6bb9ea8....',                           ==> CONTEMPLATED
      'samples_filename_pattern': '[prompt_hash]-[image_hash]',
      'show_progress_every_n_steps': 10,
      'live_preview_refresh_period': 5000,
      'samples_format': 'png',
      'grid_format': 'png',
      'save_images_add_number': False,
      'no_half': True,                                                ==> CONTEMPLATED / FIXED
      'diffusers_generator_device': 'CPU',
      'olive_vae_encoder_float32': True,
      'save_to_dirs': True,
      'theme_style': 'Dark',
      'autolaunch': True,
      'font_size': 16,
      'ui_columns': 6,
      'diffusers_model_load_variant': 'fp32',
      'diffusers_vae_load_variant': 'fp32',
      'cfgzero_enabled': True,
      'cfgzero_steps': 2,
      'gradio_theme': 'Default',
      'prompt_attention': 'a1111',                                    ==> CONTEMPLATED
      'clip_skip_enabled': True,                                      ==> CONTEMPLATED / FIXED
      'show_progress_type': 'Approximate',
      'lora_add_hashes_to_infotext': True,                            ==> CONTEMPLATED / FIXED
      'lora_fuse_native': False,
      'lora_in_memory_limit': 2,                                      ==> CONTEMPLATED / FIXED
      'schedulers_sigma': 'karras',                                   ==> CONTEMPLATED
      'schedulers_timestep_spacing': 'linspace',                      ==> CONTEMPLATED
      'schedulers_beta_schedule': 'scaled',                           ==> CONTEMPLATED
      'schedulers_prediction_type': 'epsilon',                        ==> CONTEMPLATED
      'clip_skip': 1,                                                 ==> CONTEMPLATED
      'uni_pc_lower_order_final': True,
      'uni_pc_order': 2,
      'sdnq_dequantize_compile': False,
      'ckpt_dir': 'models/Stable-diffusion',
      'diffusers_dir': 'models/Diffusers',
      'hfcache_dir': 'models/huggingface',
      'vae_dir': 'models/VAE',
      'unet_dir': 'models/UNET',
      'te_dir': 'models/Text-encoder',
      'lora_dir': 'models/Lora',
      'tunable_dir': 'models/tunable',
      'embeddings_dir': 'models/embeddings',
      'onnx_temp_dir': 'models/ONNX/temp',
      'outdir_txt2img_samples': 'outputs/text',
      'outdir_img2img_samples': 'outputs/image',
      'outdir_control_samples': 'outputs/control',
      'outdir_extras_samples': 'outputs/extras',
      'outdir_init_images': 'outputs/inputs',
      'outdir_txt2img_grids': 'outputs/grids',
      'outdir_img2img_grids': 'outputs/grids',
      'outdir_control_grids': 'outputs/grids',
      'outdir_save': 'outputs/save',
      'outdir_video': 'outputs/video',
      'styles_dir': 'models/styles',
      'yolo_dir': 'models/yolo',
      'wildcards_dir': 'models/wildcards',
      'theme_type': 'Modern',
      'ui_disabled': [],
      'lora_functional': False,
    }

    Returns:
      A dictionary containing the current options.

    Raises:
      Error: If there is an error with the API call or if the response is invalid.

    """
    options: tbase.JSONValue = self.Call(APICalls.READ_OPTIONS)
    if not isinstance(options, dict):
      raise Error(f'Invalid options response from SDNext API: {options}')
    return options

  @options.setter
  def options(self, new_options: tbase.JSONDict) -> None:
    """Set SDNext API options. Costs an API call.

    Args:
      new_options: A dictionary containing the options to set.

    """
    self.Call(APICalls.SET_OPTIONS, new_options)
    logging.info(f'Options set in SDNext API: {new_options}')

  @property
  def flags(self) -> tbase.JSONDict:
    """Get current SDNext API command flags. Costs an API call.

    Schema:

    {
      'ckpt': null,
      'data_dir': '',
      'models_dir': null,
      'embeddings_dir': 'models/embeddings',
      'vae_dir': 'models/VAE',
      'lora_dir': 'models/Lora',
      'extensions_dir': null,
      'config': 'config.json',
      'secrets': 'secrets.json',
      'ui_config': 'ui-config.json',
      'freeze': false,
      'medvram': false,
      'lowvram': false,
      'disable': '',
      'device_id': null,
      'use_cuda': false,
      'use_ipex': false,
      'use_rocm': false,
      'use_zluda': false,
      'use_openvino': false,
      'use_directml': false,
      'use_xformers': false,
      'use_nightly': false,
      'no_half': false,
      'no_half_vae': false,
      'theme': null,
      'locale': null,
      'enso': false,
      'server_name': null,
      'tls_keyfile': null,
      'tls_certfile': null,
      'tls_selfsign': false,
      'cors_origins': null,
      'cors_regex': null,
      'subpath': null,
      'autolaunch': false,
      'auth': null,
      'auth_file': null,
      'insecure': false,
      'listen': false,
      'port': 7861,
      'experimental': false,
      'ignore': false,
      'new': false,
      'safe': false,
      'test': false,
      'version': false,
      'monitor': 0.0,
      'status': 120.0,
      'log': null,
      'debug': true,
      'trace': false,
      'profile': false,
      'docs': true,
      'api_log': true,
      'backend': null,
      'enable_insecure_extension_access': false,
      'api_only': false,
      'disable_queue': false,
      'no_hashing': false,
      'no_metadata': false,
      'precision': 'autocast',
      'upcast_sampling': false,
      'hypernetwork_dir': null,
      'share': false,
      'quick': false,
      'reset': false,
      'upgrade': false,
      'requirements': false,
      'reinstall': false,
      'uv': false,
      'optional': false,
      'skip_requirements': false,
      'skip_extensions': false,
      'skip_git': false,
      'skip_torch': true,
      'skip_all': false,
      'skip_env': false,
      'agent_scheduler_sqlite_file': 'task_scheduler.sqlite3',
      'allow_code': false,
      'use_cpu': [],
      'f': false,
      'vae': null,
      'ui_settings_file': 'config.json',
      'ui_config_file': 'ui-config.json',
      'hide_ui_dir_config': false,
      'disable_console_progressbars': true,
      'disable_safe_unpickle': true,
      'lowram': false,
      'disable_extension_access': false,
      'allowed_paths': [],
      'api': true,
      'api_auth': null,
    }

    Returns:
      A dictionary containing the current command flags.

    Raises:
      Error: If there is an error with the API call or if the response is invalid.

    """
    flags: tbase.JSONValue = self.Call(APICalls.READ_OPTIONS)
    if not isinstance(flags, dict):
      raise Error(f'Invalid command flags response from SDNext API: {flags}')
    return flags

  def GetModels(self) -> list[db.AIModelType]:
    """Get list of available models from SDNext API.

    Schema:

    {
      'items': [
        {
          'title': 'SDXL_00_XLB_v10VAEFix [e6bb9ea85b]',
          'model_name': 'SDXL_00_XLB_v10VAEFix',
          'filename': '/foo/bar/models/Stable-diffusion/SDXL_00_XLB_v10VAEFix.safetensors',
          'type': 'safetensors',
          'hash': 'e6bb9ea85b',
          'sha256': 'e6bb9ea85bbf7bf6478a7c6d18b71246f22e95d41bcdd80ed40aa212c33cfeff',
          'size': 6938078334,
          'mtime': '2023-09-08T18:06:07',
          'version': '',
          'subfolder': None,
        },
        # ...
      ],
    }

    Returns:
      A list of AIModelType objects representing the available models. BEWARE: hash may be ''!

    Raises:
      Error: If there is an error with the API call or if the response is invalid.

    """
    # call
    models: tbase.JSONValue = self.Call(APICalls.MODELS)
    if (
      not isinstance(models, dict)
      or not models
      or 'items' not in models
      or not isinstance(models['items'], list)
    ):
      raise Error(f'Invalid models response from SDNext API: {models}')
    # we got a valid response, parse it
    parsed: list[db.AIModelType] = []
    for model in models['items']:
      if not isinstance(model, dict):
        raise Error(f'Invalid model entry in SDNext API response: {model}')
      # make the struct
      model_path = pathlib.Path(cast('str', model.get('filename', '')).strip())
      new_model: db.AIModelType = db.AIModelType(
        hash=cast('str', model.get('sha256', '') or '').strip(),
        name=cast('str', model.get('model_name', '') or '').strip(),
        alias=cast('str', model.get('title', '') or '').strip(),
        path=str(model_path),
        model_type=db.ModelType(cast('str', model.get('type', '') or '').strip()).value,
        function=db.ModelFunction.Model.value,
        metadata=copy.deepcopy(model.copy()),
        sidecar=None,  # this, if present, will be filled in by the DB
        description=None,
        autov3=None,  # models don't have autov3 hashes, only lora/lyco
      )
      # check name and path sanity; hash may be empty for now, so we don't check it
      if not model_path.exists():
        raise Error(f'Model file not found for model from SDNext API: {new_model}')
      if not new_model['name']:
        raise Error(f'Missing model name for model from SDNext API: {new_model}')
      # done, so add to list
      parsed.append(new_model)
    # done, return
    return parsed

  def GetLora(self) -> list[db.AIModelType]:
    """Get list of available lora/lycoris from SDNext API.

    Schema:

    [
      {
        'name': 'XL-CLR-colorful-fractal',
        'alias': 'tr03_colorful_04_colorful_fractal',
        'path': '/foo/bar/models/Lora/XL-CLR-colorful-fractal.safetensors',
        'metadata': {
          # ... HUGE JSON, LOTS OF FIELDS, INCLUDING TAG FREQUENCIES, e.g.:
          'ss_network_module': 'lycoris.kohya',  # or "networks.lora"
          'ss_tag_frequency': {
            'img': {
              'green background': 2,
              'no humans': 7,
              'solo': 22,
              'simple background': 7,
            },
          },
        },
      },
    ]

    Returns:
      A list of AIModelType objects representing the available lora/lycoris. BEWARE: hash will be ''

    Raises:
      Error: If there is an error with the API call or if the response is invalid.

    """
    # call
    loras: tbase.JSONValue = self.Call(APICalls.LORA)
    if not isinstance(loras, list) or not loras:
      raise Error(f'Invalid models response from SDNext API: {loras}')
    # we got a valid response, parse it
    parsed: list[db.AIModelType] = []
    for lora in loras:
      if not isinstance(lora, dict):
        raise Error(f'Invalid model entry in SDNext API response: {lora}')
      # make the struct
      lora_path = pathlib.Path(cast('str', lora.get('path', '')).strip())
      new_lora: db.AIModelType = db.AIModelType(
        hash='',  # computed later
        name=cast('str', lora.get('name', '') or '').strip(),
        alias=cast('str', lora.get('alias', '') or '').strip(),
        path=str(lora_path),
        model_type=db.ModelType(lora_path.suffix.lower()[1:]).value,
        function=db.ModelFunction.Lora.value,  # start with lora and decide below in metadata
        metadata=(
          copy.deepcopy(cast('tbase.JSONDict', lora.get('metadata', {})))
          if isinstance(lora.get('metadata'), dict)
          else {}
        ),
        sidecar=None,  # lora/lyco don't have sidecar info, only models do
        description=None,
        autov3=None,  # computed later
      )
      # check name and path sanity; hash may be empty for now, so we don't check it
      if not lora_path.exists():
        raise Error(f'Model file not found for model from SDNext API: {new_lora}')
      if not new_lora['name']:
        raise Error(f'Missing model name for model from SDNext API: {new_lora}')
      # figure out network type
      ss_network: str = cast('str', new_lora['metadata'].get('ss_network_module', '') or '').lower()
      if 'lyco' in ss_network:
        new_lora['function'] = db.ModelFunction.Lycoris.value
      elif ss_network.strip() and 'lora' not in ss_network:
        logging.error(f'Unknown `ss_network_module` from SDNext API: {ss_network!r} in {new_lora}')
      # done, so add to list
      parsed.append(new_lora)
    # done, return
    return parsed

  def GetEmbeddings(self) -> dict[str, str]:
    """Get list of available (loaded) embeddings from SDNext API.

    Schema:

    {
      'loaded': [
        {
          'name': 'zPDXL3',
          'filename': '/foo/bar/models/embeddings/zPDXL3.pt',
          'step': null,
          'shape': null,
          'vectors': 0,
          'sd_checkpoint': null,
          'sd_checkpoint_name': null,
        },
        # ...
      ],
      'skipped': [
        {
          'name': 'zPDXLrl-neg',
          'filename': '/foo/bar/models/embeddings/zPDXLrl-neg.pt',
          'step': null,
          'shape': null,
          'vectors': 0,
          'sd_checkpoint': null,
          'sd_checkpoint_name': null,
        },
        # ...
      ],
    }

    Returns:
      {name: path}

    Raises:
      Error: If there is an error with the API call or if the response is invalid.

    """
    # call
    embeddings: tbase.JSONValue = self.Call(APICalls.EMBEDDINGS)
    if (
      not isinstance(embeddings, dict)
      or not embeddings
      or 'loaded' not in embeddings
      or not isinstance(embeddings['loaded'], list)
    ):
      raise Error(f'Invalid embeddings response from SDNext API: {embeddings}')
    # parse and return loaded and valid embeddings
    return {
      n: str(p)
      for n, p in (
        (cast('str', e['name']), pathlib.Path(cast('str', e['filename'])).expanduser().resolve())
        for e in embeddings['loaded']
        if isinstance(e, dict) and 'name' in e and 'filename' in e
      )
      if p.exists() and p.is_file()
    }

  def Txt2Img(  # noqa: C901, PLR0914, PLR0915
    self,
    model: db.AIModelType,
    meta: db.AIMetaType,
    *,
    dir_root: pathlib.Path | None = None,
    tm: int | None = None,
  ) -> tuple[db.DBImageType, bytes]:
    """Generate image from text prompt using SDNext API.

    See: <https://github.com/vladmandic/sdnext/blob/master/cli/api-txt2img.py>

    Schema:

    {
      "sd_model_checkpoint": "string",                     ==> CONTEMPLATED
      "prompt": "",                                        ==> CONTEMPLATED
      "negative_prompt": "",                               ==> CONTEMPLATED
      "seed": -1,                                          ==> CONTEMPLATED
      "subseed": -1,                                       ==> CONTEMPLATED
      "subseed_strength": 0,                               ==> CONTEMPLATED
      "seed_resize_from_h": -1,
      "seed_resize_from_w": -1,
      "batch_size": 1,                              ==> CONTEMPLATED / FIXED
      "n_iter": 1,                                  ==> CONTEMPLATED / FIXED
      "steps": 20,                                  ==> CONTEMPLATED
      "clip_skip": 1,                               ==> CONTEMPLATED
      "width": 1024,                                ==> CONTEMPLATED
      "height": 1024,                               ==> CONTEMPLATED
      "sampler_index": 0,
      "sampler_name": "Default",                    ==> CONTEMPLATED
      "hr_sampler_name": "Same as primary",
      "eta": 0,
      "guidance_name": "Default",
      "guidance_scale": 6,                          ==> IGNORED BY txt2img
      "guidance_rescale": 0,                        ==> IGNORED BY txt2img
      "guidance_start": 0,                          ==> IGNORED BY txt2img
      "guidance_stop": 1,                           ==> IGNORED BY txt2img
      "cfg_scale": 6,                               ==> CONTEMPLATED
      "cfg_end": 1,                                 ==> CONTEMPLATED
      "diffusers_guidance_rescale": 0,              ==> CONTEMPLATED
      "pag_scale": 0,                               ==> CONTEMPLATED / FIXED
      "pag_adaptive": 0.5,                          ==> CONTEMPLATED / FIXED
      "styles": [                                   ==> CONTEMPLATED / FIXED
        "string"
      ],
      "tiling": false,                              ==> CONTEMPLATED / FIXED
      "vae_type": "Full",                           ==> CONTEMPLATED / FIXED
      "hidiffusion": false,                         ==> CONTEMPLATED / FIXED
      "do_not_reload_embeddings": false,
      "detailer_enabled": false,
      "detailer_prompt": "",
      "detailer_negative": "",
      "detailer_steps": 10,
      "detailer_strength": 0.3,
      "detailer_resolution": 1024,
      "detailer_segmentation": true,
      "detailer_include_detections": true,
      "detailer_merge": true,
      "detailer_sort": true,
      "detailer_classes": "string",
      "detailer_conf": 0,
      "detailer_iou": 0,
      "detailer_max": 0,
      "detailer_min_size": 0,
      "detailer_max_size": 0,
      "detailer_blur": 0,
      "detailer_padding": 0,
      "detailer_sigma_adjust": 0,
      "detailer_sigma_adjust_max": 0,
      "detailer_models": [
        "string"
      ],
      "detailer_augment": true,
      "img2img_color_correction": true,
      "color_correction_method": "string",
      "img2img_background_color": "string",
      "img2img_fix_steps": true,
      "mask_apply_overlay": true,
      "include_mask": true,
      "inpainting_mask_weight": 0,
      "samples_save": true,
      "samples_format": "string",
      "save_images_before_highres_fix": true,
      "save_images_before_refiner": true,
      "save_images_before_detailer": true,
      "save_images_before_color_correction": true,
      "grid_save": true,
      "grid_format": "string",
      "return_grid": true,
      "save_mask": true,
      "save_mask_composite": true,
      "return_mask": true,
      "return_mask_composite": true,
      "keep_incomplete": true,
      "image_metadata": true,                            ==> CONTEMPLATED / FIXED
      "jpeg_quality": 0,
      "lora_fuse_native": true,
      "lora_fuse_diffusers": true,
      "lora_force_reload": true,
      "extra_networks_default_multiplier": 0,
      "lora_apply_tags": 0,
      "hdr_mode": 0,
      "hdr_brightness": 0,
      "hdr_color": 0,
      "hdr_sharpen": 0,
      "hdr_clamp": false,
      "hdr_boundary": 4,
      "hdr_threshold": 0.95,
      "hdr_maximize": false,
      "hdr_max_center": 0.6,
      "hdr_max_boundary": 1,
      "hdr_color_picker": "#000000",
      "hdr_tint_ratio": 0,
      "hdr_apply_hires": true,
      "grading_brightness": 0,
      "grading_contrast": 0,
      "grading_saturation": 0,
      "grading_hue": 0,
      "grading_gamma": 1,
      "grading_sharpness": 0,
      "grading_color_temp": 6500,
      "grading_shadows": 0,
      "grading_midtones": 0,
      "grading_highlights": 0,
      "grading_clahe_clip": 0,
      "grading_clahe_grid": 8,
      "grading_shadows_tint": "#000000",
      "grading_highlights_tint": "#ffffff",
      "grading_split_tone_balance": 0.5,
      "grading_vignette": 0,
      "grading_grain": 0,
      "grading_lut_file": "",
      "grading_lut_strength": 1,
      "denoising_strength": 0.3,
      "init_images": [
        "string"
      ],
      "init_control": [
        "string"
      ],
      "image_cfg_scale": 0,
      "initial_noise_multiplier": 0,
      "scale_by": 1,
      "selected_scale_tab": 0,
      "resize_mode": 0,
      "resize_name": "None",
      "resize_context": "None",
      "width_before": 0,
      "width_after": 0,
      "width_mask": 0,
      "height_before": 0,
      "height_after": 0,
      "height_mask": 0,
      "resize_name_before": "None",
      "resize_name_after": "None",
      "resize_name_mask": "None",
      "resize_mode_before": 0,
      "resize_mode_after": 0,
      "resize_mode_mask": 0,
      "resize_context_before": "None",
      "resize_context_after": "None",
      "resize_context_mask": "None",
      "selected_scale_tab_before": 0,
      "selected_scale_tab_after": 0,
      "selected_scale_tab_mask": 0,
      "scale_by_before": 1,
      "scale_by_after": 1,
      "scale_by_mask": 1,
      "mask": "string",
      "latent_mask": "string",
      "mask_for_overlay": "string",
      "mask_blur": 4,
      "paste_to": "string",
      "inpainting_fill": 1,
      "inpaint_full_res": false,
      "inpaint_full_res_padding": 0,
      "inpainting_mask_invert": 0,
      "overlay_images": "string",
      "enable_hr": false,                          ==> CONTEMPLATED / FIXED
      "firstphase_width": 0,
      "firstphase_height": 0,
      "hr_scale": 2,
      "hr_force": false,
      "hr_resize_mode": 0,
      "hr_resize_context": "None",
      "hr_second_pass_steps": 0,
      "hr_resize_x": 0,
      "hr_resize_y": 0,
      "hr_denoising_strength": 0,
      "refiner_steps": 5,                          ==> CONTEMPLATED / FIXED
      "hr_upscaler": "string",
      "refiner_start": 0,
      "refiner_prompt": "",
      "refiner_negative": "",
      "hr_refiner_start": 0,
      "enhance_prompt": false,
      "do_not_save_samples": false,
      "do_not_save_grid": false,
      "xyz": false,
      "script_args": [],
      "schedulers_prediction_type": "string",                     ==> CONTEMPLATED
      "schedulers_beta_schedule": "string",                       ==> CONTEMPLATED
      "schedulers_timesteps": "string",
      "schedulers_sigma": "string",                               ==> CONTEMPLATED
      "schedulers_use_thresholding": true,
      "schedulers_use_loworder": true,
      "schedulers_solver_order": 0,
      "uni_pc_variant": "string",
      "schedulers_beta_start": 0,
      "schedulers_beta_end": 0,
      "schedulers_shift": 0,
      "schedulers_dynamic_shift": true,
      "schedulers_base_shift": 0,
      "schedulers_max_shift": 0,
      "schedulers_rescale_betas": true,
      "schedulers_timestep_spacing": "string",                    ==> CONTEMPLATED
      "schedulers_timesteps_range": 0,
      "schedulers_sigma_adjust": 0,
      "schedulers_sigma_adjust_min": 0,
      "schedulers_sigma_adjust_max": 0,
      "scheduler_eta": 0,                                         ==> CONTEMPLATED / FIXED
      "eta_noise_seed_delta": 0,
      "enable_batch_seeds": true,
      "diffusers_generator_device": "string",
      "nan_skip": true,
      "sequential_seed": true,
      "prompt_attention": "string",                               ==> CONTEMPLATED
      "prompt_mean_norm": true,
      "diffusers_zeros_prompt_pad": true,
      "te_pooled_embeds": true,
      "lora_apply_te": true,
      "te_complex_human_instruction": "string",
      "te_use_mask": true,
      "freeu_enabled": true,
      "freeu_b1": 0,
      "freeu_b2": 0,
      "freeu_s1": 0,
      "freeu_s2": 0,
      "hypertile_unet_enabled": true,
      "hypertile_hires_only": true,                      ==> CONTEMPLATED / FIXED
      "hypertile_unet_tile": 0,                          ==> CONTEMPLATED / FIXED
      "hypertile_unet_min_tile": 0,
      "hypertile_unet_swap_size": 0,                     ==> CONTEMPLATED / FIXED
      "hypertile_unet_depth": 0,                         ==> CONTEMPLATED / FIXED
      "hypertile_vae_enabled": true,                     ==> CONTEMPLATED / FIXED
      "hypertile_vae_tile": 0,
      "hypertile_vae_swap_size": 0,
      "teacache_enabled": true,                          ==> CONTEMPLATED / FIXED
      "teacache_thresh": 0,
      "token_merging_method": "string",                  ==> CONTEMPLATED / FIXED
      "tome_ratio": 0,                                   ==> CONTEMPLATED / FIXED
      "todo_ratio": 0,                             ==> CONTEMPLATED / FIXED
      "override_settings_restore_afterwards": true,
      "override_settings": {
        "additionalProp1": {}
      },
      "script_name": "",
      "send_images": true,                                ==> CONTEMPLATED / FIXED
      "save_images": false,                               ==> CONTEMPLATED
      "alwayson_scripts": {},
      "ip_adapter": [
        {
          "adapter": "Base",
          "images": [],
          "masks": [],
          "scale": 0.5,
          "start": 0,
          "end": 1,
          "crop": false
        }
      ],
      "control_units": [
        {
          "process": "",
          "model": "",
          "strength": 1,
          "start": 0,
          "end": 1,
          "override": "string",
          "unit_type": "string",
          "image": "string"
        }
      ],
      "face": {
        "mode": "FaceID",
        "source_images": [
          "string"
        ],
        "ip_model": "FaceID Base",
        "ip_override_sampler": true,
        "ip_cache_model": true,
        "ip_strength": 1,
        "ip_structure": 1,
        "id_strength": 1,
        "id_conditioning": 0.5,
        "id_cache": true,
        "pm_trigger": "person",
        "pm_strength": 1,
        "pm_start": 0.5,
        "fs_cache": true
      },
      "extra": {}
    }

    Args:
      model: AIModelType object representing the model to use for generation
      meta: AIMetaType object containing the generation metadata (e.g., prompt, steps,
          seed, width, height, sampler_id, model_key)
      dir_root: (default: None) Directory root to save the generated image, if None don't save
      tm: (default: None) Optional timestamp to use for the generated image metadata;
          if None, uses time we  get image back from API

    Returns:
      A tuple containing the DBImageType object and the raw image data.
      The file path in the DBImageType will be '' if dir_root is None, otherwise it will be
      the path to the saved image. The path will be marked main==False, so if it is to be a
      main it has to be flipped to main==True.

    Raises:
      Error: If there is an error with the API call, if the response is invalid,
          or if the image data is invalid.

    """
    meta = copy.deepcopy(meta)  # make a copy of the meta to modify without affecting caller
    # set options; most importantly, set the model if needed
    if meta['img2img'] is not None:
      raise Error('img2img is not supported by Txt2Img() call')
    if model['hash'] != meta['model_hash']:
      raise Error(f'Model hash mismatch: expected {meta["model_hash"]}, got {model["hash"]}')
    if model['function'] != db.ModelFunction.Model.value:
      raise Error(f'Incompatible model: expected model, got {model}')
    clip_skip: int = meta['clip_skip'] // 10  # TODO: in future, when accepts float do regular div
    base_options: tbase.JSONDict = {
      # FIXED OPTIONS
      'clip_skip_enabled': clip_skip > 1,  # only enable if clip_skip > 1
      'no_half': True,
      'lora_add_hashes_to_infotext': True,  # TODO: not having any effect: investigate!
      'lora_in_memory_limit': 3,
      # VARIABLE OPTIONS
      'sd_checkpoint_hash': model['hash'],  # set the model hash to trigger load if needed
      'prompt_attention': meta['parser'],
      # TODO: investigate why the server returns error when we change clip, then it behaves again
      'clip_skip': clip_skip,
      'schedulers_sigma': 'default' if meta['sch_sigma'] is None else meta['sch_sigma'],
      'schedulers_timestep_spacing': (
        'default' if meta['sch_spacing'] is None else meta['sch_spacing']
      ),
      'schedulers_beta_schedule': 'default' if meta['sch_beta'] is None else meta['sch_beta'],
      'schedulers_prediction_type': 'default' if meta['sch_type'] is None else meta['sch_type'],
    }
    with timer.Timer(emit_log=False) as tmr_load:
      api_options = self.options  # get current options from API before sending any changes
      if str(api_options['sd_checkpoint_hash'] or '') != model['hash']:
        logging.info(f'Change models: {api_options["sd_model_checkpoint"]!r} -> {model["name"]!r}')
      self.options = base_options  # set the options above; will call API and update model if needed
      if str(api_options['sd_checkpoint_hash'] or '') != model['hash']:
        logging.info('Reloading checkpoint')
        self.Call(APICalls.RELOAD_CHECKPOINT)  # needed if running in api-only to trigger new load
    logging.info(f'Options loaded in SDNext API in {tmr_load}')
    # sanity check some options
    if (
      meta['width'] <= 0 or meta['height'] <= 0 or meta['width'] % 8 != 0 or meta['height'] % 8 != 0
    ):
      raise Error(f'Invalid image dimensions or not divisible by 8: {meta}')
    if meta['sampler'] in {s.value for s in base.SamplerA1111}:
      raise Error(f'Sampler {meta["sampler"]!r} not supported by SDNext')
    # set the options
    options: tbase.JSONDict = {
      # FIXED OPTIONS
      'send_images': True,
      'image_metadata': True,
      'n_iter': 1,
      'batch_size': 1,
      'tiling': False,
      'vae_type': 'Full',  # TODO: see if this should be variable
      'hidiffusion': False,
      'enable_hr': False,
      'styles': [],
      'refiner_steps': 0,
      'schedulers_solver_order': 0,
      'scheduler_eta': 0,
      'pag_scale': 0,  # TODO: make real option? shows up as 'CFG true' when != 0
      'pag_adaptive': 0.5,  # TODO: make real option? shows up as 'CFG adaptive' when != 0.5
      'hypertile_unet_enabled': False,
      'hypertile_hires_only': False,
      'hypertile_unet_tile': 0,  # 0 in sdnext means automatic
      'hypertile_unet_swap_size': 2,
      'hypertile_unet_depth': 0,
      'hypertile_vae_enabled': False,
      'teacache_enabled': False,  # unnecessary optimization
      'token_merging_method': None,  # unnecessary optimization
      'tome_ratio': 0,  # unnecessary optimization
      'todo_ratio': 0,  # unnecessary optimization
      # VARIABLE OPTIONS
      # TODO: respect vae
      # TODO: respect pony
      # TODO: respect clip2
      'save_images': self._server_save_images,
      'sd_model_checkpoint': model['name'],
      'prompt': meta['positive'],  # TODO: "[foo:bar:0.1]" this pattern is failing
      'negative_prompt': meta['negative'],
      'steps': meta['steps'],
      'seed': meta['seed'],
      'subseed': base.SeedGen() if meta['v_seed'] is None else meta['v_seed']['seed'],
      'subseed_strength': 0 if meta['v_seed'] is None else meta['v_seed']['percent'] / 100,  # /100
      'width': meta['width'],
      'height': meta['height'],
      'sampler_name': meta['sampler'],
      'cfg_scale': meta['cfg_scale'] / 10,  # remember to divide by 10
      'cfg_end': meta['cfg_end'] / 10,  # remember to divide by 10
      'diffusers_guidance_rescale': meta['cfg_rescale'] / 100,  # remember to divide by 100
      'freeu_enabled': bool(meta['freeu']),
      'freeu_b1': meta['freeu']['b1'] / 100 if meta['freeu'] else 1.0,  # remember to divide by 100
      'freeu_b2': meta['freeu']['b2'] / 100 if meta['freeu'] else 1.0,  # remember to divide by 100
      'freeu_s1': meta['freeu']['s1'] / 100 if meta['freeu'] else 1.0,  # remember to divide by 100
      'freeu_s2': meta['freeu']['s2'] / 100 if meta['freeu'] else 1.0,  # remember to divide by 100
      # OPTIONS THAT ARE 1:1 WITH API OPTIONS (so we copy)
      'prompt_attention': base_options['prompt_attention'],
      'clip_skip_enabled': base_options['clip_skip_enabled'],
      'clip_skip': base_options['clip_skip'],
      'schedulers_sigma': base_options['schedulers_sigma'],
      'schedulers_timestep_spacing': base_options['schedulers_timestep_spacing'],
      'schedulers_beta_schedule': base_options['schedulers_beta_schedule'],
      'schedulers_prediction_type': base_options['schedulers_prediction_type'],
    }
    # make the call to the APIs
    logging.info(f'Generating image with SDNext API, options: {options}')
    with timer.Timer(emit_log=False) as tmr_generate:
      data: tbase.JSONValue = self.Call(APICalls.TXT2IMG, options)
    tm_created: int = tm if tm and tm > 0 else timer.Now()  # time is as soon as we get the response
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
    img_data, raw_hash, info_text = _ExtractImageData(data)
    # hash, log
    img_hash: str = hashes.Hash256(img_data).hex()
    logging.info(f'Got {human.HumanizedBytes(len(img_data))} image in {tmr_generate}: {img_hash}')
    # if we are going to save the image, figure out the full path
    full_path: pathlib.Path | None = None
    if dir_root is not None:
      date_str: str = time.strftime('%Y-%m-%d', time.gmtime(tm_created))
      tm_str: str = time.strftime('%Y%m%d%H%M%S', time.gmtime(tm_created))
      out_dir: pathlib.Path = dir_root / date_str
      out_dir.mkdir(exist_ok=True)  # make sure the date dir exists, create it if not
      filename: str = (
        f'{tm_str}-'
        f'{base.PromptHash(meta["positive"], meta["negative"])}-'
        f'{model["hash"][:10]}-'
        f'{meta["cfg_scale"]}-{meta["steps"]}-'
        f'{meta["width"]}-{meta["height"]}-'
        f'{meta["seed"]}-'
        f'{img_hash[:12]}.png'
      )
      full_path = out_dir / filename
      full_path.write_bytes(img_data)
      logging.info(f'SDNext API image saved: {full_path}, {human.HumanizedBytes(len(img_data))}')
    # create the metadata
    path_key: str = str(full_path) if full_path else ''
    path_obj = db.DBImagePathType(
      main=False,  # to be safe
      created_at=tm_created,
      origin=db.ImageOrigin.TransNext.value,
      parse_errors=None,
      version=f'{self.version}/{__version__}',  # TransNext version is like '0eb4a98e0/1.0.0'
      ai_meta=meta,  # no need to copy here since we already copied above
      sd_info=json.loads(cast('str', data['info'])),
      sd_params=cast('tbase.JSONDict', data['parameters']),
    )
    db_image: db.DBImageType = db.DBImageType(
      hash=img_hash,
      raw_hash=raw_hash,
      size=len(img_data),
      width=meta['width'],
      height=meta['height'],
      format=base.ImageFormat.PNG.value,
      info=info_text,
      paths={path_key: path_obj},
    )
    # make sure the model and dimensions are coherent as sanity checks
    if (gen_model := path_obj['sd_params']['sd_model_checkpoint']) != model['name']:  # type: ignore[index]
      raise Error(f'Model name mismatch: expected {model["name"]}, got {gen_model}: {data}')
    if (
      meta['width'] != path_obj['sd_info']['width']  # type: ignore[index]
      or meta['height'] != path_obj['sd_info']['height']  # type: ignore[index]
    ):
      raise Error(
        f'Expected image of size {path_obj["sd_info"]["width"]}x{path_obj["sd_info"]["height"]} '  # type: ignore[index]
        f'from SDNext API, got {meta["width"]}x{meta["height"]}: {data}'
      )
    # all is good, save the image to disk and return the DB entry and raw image data
    logging.debug(f'SDNext API image metadata: {db_image}')
    return (db_image, img_data)


def _ExtractImageData(data: tbase.JSONDict) -> tuple[bytes, str, str]:
  """Extract and validate image data from SDNext API response.

  Args:
    data: JSON response from SDNext API containing the image data and metadata

  Returns:
    (PNG image data as bytes, the computed hash of the raw image data, info text from AI)

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
  # open, do some basic validation on the image and metadata, prepare the DB entry for it
  fmt, width, height, raw_hash, info_text = base.GetBasicDataFromImage(img_data)
  if fmt != base.ImageFormat.PNG:
    raise Error(f'Expected PNG image from SDNext API, got {fmt}: {data}')
  if not info_text:
    raise Error(f'No info text found in image metadata from SDNext API: {data}')
  if width != data['parameters']['width'] or height != data['parameters']['height']:  # type: ignore[index,call-overload]
    raise Error(
      f'Expected image of size {data["parameters"]["width"]}x{data["parameters"]["height"]} from '  # type: ignore[index,call-overload]
      f'SDNext API, got {width}x{height}: {data}'
    )
  # passed all checks
  return (img_data, raw_hash, info_text)


def _Call(
  method: abc.Callable[..., requests.Response],
  sd_url: str,
  endpoint: str,
  payload: tbase.JSONDict | None = None,
  *,
  record_list: list[tbase.JSONDict] | None = None,
) -> tbase.JSONValue:
  """Call SDNext API endpoint with given payload.

  Args:
    method: HTTP method function from requests library (e.g., requests.post)
    sd_url: Base URL of the SDNext API, e.g. "http://127.0.0.1:5000"
    endpoint: API endpoint to call
    payload: JSON-serializable payload to send in the request body (default is None)
    record_list: (default None) optional MUTABLE list to append the call and response data

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
    response: tbase.JSONValue = cast('tbase.JSONValue', req.json())
    if record_list is not None:
      record_list.append(
        {
          'call': {
            'method': method.__name__.upper(),
            'url': full_url,
            'payload': copy.deepcopy(payload),
          },
          'response': copy.deepcopy(response),
        }
      )
      logging.info(f'Call recorded: {method.__name__.upper()}/{full_url}')
    return response
  except requests.exceptions.ConnectionError as err:
    raise APIConnectionError(f'Failed to connect to SDNext API {method} {full_url}') from err
  except Exception as err:
    raise Error(f'Error calling SDNext API {method} {full_url}: {err}') from err
