# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""TransNext core SDNext API module.

Look at <http://127.0.0.1:7860/docs> on a running server to see the API calls and details.
"""

from __future__ import annotations

import base64
import copy
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
  OPTIONS = 'options'
  RELOAD_CHECKPOINT = 'reload-checkpoint'
  TXT2IMG = 'txt2img'


class APICalls(enum.Enum):
  """SDNext API calls."""

  STATUS = 1
  MODELS = 2
  READ_OPTIONS = 3
  SET_OPTIONS = 4
  RELOAD_CHECKPOINT = 5
  TXT2IMG = 6


_API_CALL_MATRIX: dict[
  APICalls, tuple[APIVersions, Endpoints, abc.Callable[..., requests.Response]]
] = {
  # see <http://127.0.0.1:7860/docs> for details on the API
  APICalls.STATUS: (APIVersions.V2, Endpoints.SYSTEM_STATUS, requests.get),
  APICalls.MODELS: (APIVersions.V1, Endpoints.MODELS, requests.get),
  APICalls.READ_OPTIONS: (APIVersions.V2, Endpoints.OPTIONS, requests.get),
  APICalls.SET_OPTIONS: (APIVersions.V2, Endpoints.OPTIONS, requests.post),
  APICalls.RELOAD_CHECKPOINT: (APIVersions.V1, Endpoints.RELOAD_CHECKPOINT, requests.post),
  APICalls.TXT2IMG: (APIVersions.V1, Endpoints.TXT2IMG, requests.post),
}


class Error(base.Error):
  """TransNext SDNext API exception."""


class APIConnectionError(Error, ConnectionError):
  """TransNext SDNext API connection exception."""


class API(db.APIProtocol):
  """SDNext API client."""

  def __init__(self, api_url: str, *, server_save_images: bool = False) -> None:
    """Initialize SDNext API client.

    Args:
      api_url: Base URL of the SDNext API, e.g. "http://127.0.0.1:5000"
      server_save_images: Whether if the server will save a copy of the images too (default False)

    Raises:
      APIConnectionError: If there is a connection error to the SDNext API.

    """
    self._api_url: str = api_url
    self._server_save_images: bool = server_save_images
    commit, updated = self.ServerVersion()
    if not commit or not updated:
      raise APIConnectionError(f'Failed to get version from SDNext API: {commit!r}, {updated!r}')
    self._version: str = commit
    logging.info(
      f'API/{commit}/{updated} @ {self._api_url}{" + SAVE" if self._server_save_images else ""}'
    )

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
    return _Call(method, self._api_url, version.value + endpoint.value, payload)

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

  def GetModels(self) -> list[db.AIModelType]:
    """Get list of available models from SDNext API.

    Returns:
      A list of AIModelType objects representing the available models. BEWARE: hash may be ''!

    Raises:
      Error: If there is an error with the API call or if the response is invalid.

    """
    models: tbase.JSONValue = self.Call(APICalls.MODELS)
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

  def Txt2Img(  # noqa: C901
    self, model: db.AIModelType, meta: db.AIMetaType, *, dir_root: pathlib.Path | None = None
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
      "pag_scale": 0,
      "pag_adaptive": 0.5,
      "styles": [
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
      "image_metadata": true,
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
      "enable_hr": false,
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
      "refiner_steps": 5,
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
      "scheduler_eta": 0,
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
      "hypertile_hires_only": true,
      "hypertile_unet_tile": 0,
      "hypertile_unet_min_tile": 0,
      "hypertile_unet_swap_size": 0,
      "hypertile_unet_depth": 0,
      "hypertile_vae_enabled": true,
      "hypertile_vae_tile": 0,
      "hypertile_vae_swap_size": 0,
      "teacache_enabled": true,
      "teacache_thresh": 0,
      "token_merging_method": "string",
      "tome_ratio": 0,
      "todo_ratio": 0,
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

    Returns:
      A tuple containing the DBImageType object and the raw image data.

    Raises:
      Error: If there is an error with the API call, if the response is invalid,
          or if the image data is invalid.

    """
    # set options; most importantly, set the model if needed
    if meta['img2img'] is not None:
      raise Error('img2img is not supported by Txt2Img() call')
    if model['hash'] != meta['model_hash']:
      raise Error(f'Model hash mismatch: expected {meta["model_hash"]}, got {model["hash"]}')
    base_options: tbase.JSONDict = {
      # FIXED OPTIONS
      'clip_skip_enabled': True,
      'no_half': True,
      'lora_add_hashes_to_infotext': True,
      'lora_in_memory_limit': 3,
      # VARIABLE OPTIONS
      'sd_checkpoint_hash': model['hash'],  # set the model hash to trigger load if needed
      'prompt_attention': meta['parser'],
      'clip_skip': meta['clip_skip'] // 10,  # TODO: in future, when accepts float do regular div
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
      meta['width'] <= 0
      or meta['height'] <= 0
      or meta['width'] % 16 != 0
      or meta['height'] % 16 != 0
    ):
      raise Error(f'Invalid image dimensions: {meta}')
    if meta['sampler'] in {s.value for s in base.SamplerA1111}:
      raise Error(f'Sampler {meta["sampler"]!r} not supported by SDNext')
    # set the options
    options: tbase.JSONDict = {
      # FIXED OPTIONS
      'send_images': True,
      'n_iter': 1,
      'batch_size': 1,
      'tiling': False,
      'vae_type': 'Full',  # TODO: see if this should be variable
      'hidiffusion': False,
      # VARIABLE OPTIONS
      'save_images': self._server_save_images,
      'sd_model_checkpoint': model['name'],
      'prompt': meta['positive'],
      'negative_prompt': meta['negative'],
      'steps': meta['steps'],
      'seed': meta['seed'],
      'subseed': base.SeedGen() if meta['v_seed'] is None else meta['v_seed'][0],
      'subseed_strength': 0 if meta['v_seed'] is None else meta['v_seed'][1] / 100,  # divide by 100
      'width': meta['width'],
      'height': meta['height'],
      'sampler_name': meta['sampler'],
      'prompt_attention': meta['parser'],
      'cfg_scale': meta['cfg_scale'] / 10,  # remember to divide by 10
      'cfg_end': meta['cfg_end'] / 10,  # remember to divide by 10
      'diffusers_guidance_rescale': meta['cfg_rescale'] / 100,  # remember to divide by 100
      'clip_skip': meta['clip_skip'] // 10,  # TODO: in future, when accepts float do regular div
      'schedulers_sigma': 'default' if meta['sch_sigma'] is None else meta['sch_sigma'],
      'schedulers_timestep_spacing': (
        'default' if meta['sch_spacing'] is None else meta['sch_spacing']
      ),
      'schedulers_beta_schedule': 'default' if meta['sch_beta'] is None else meta['sch_beta'],
      'schedulers_prediction_type': 'default' if meta['sch_type'] is None else meta['sch_type'],
    }
    # make the call to the APIs
    logging.info(f'Generating image with SDNext API, options: {options}')
    with timer.Timer(emit_log=False) as tmr_generate:
      data: tbase.JSONValue = self.Call(APICalls.TXT2IMG, options)
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
    logging.info(f'Got {human.HumanizedBytes(len(img_data))} image in {tmr_generate}: {img_hash}')
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
    db_image: db.DBImageType = db.DBImageType(
      hash=img_hash,
      raw_hash=raw_hash,
      path=str(full_path) if full_path else None,
      alt_path=[],
      size=len(img_data),
      width=meta['width'],
      height=meta['height'],
      format=db.ImageFormat.PNG.value,
      created_at=tm_created,
      origin=db.ImageOrigin.TransNext.value,
      version=f'{self.version}/{__version__}',  # TransNext version is like '0eb4a98e0/1.0.0'
      ai_meta=copy.deepcopy(meta),
      sd_info=json.loads(cast('str', data['info'])),
      sd_params=cast('tbase.JSONDict', data['parameters']),
      parse_errors=[],
    )
    # make sure the model and dimensions are coherent as sanity checks
    if (gen_model := db_image['sd_params']['sd_model_checkpoint']) != model['name']:
      raise Error(f'Model name mismatch: expected {model["name"]}, got {gen_model}: {data}')
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
