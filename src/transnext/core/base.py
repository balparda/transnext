# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""TransNext core base module."""

from __future__ import annotations

import dataclasses
import enum
import io
import logging
import os
import pathlib
import re
from collections import abc

import typer
from PIL import Image
from transcrypto.cli import clibase
from transcrypto.core import hashes
from transcrypto.utils import base as tbase
from transcrypto.utils import saferandom


class Error(tbase.Error):
  """TransNext base exception."""


SD_URL: str | None = os.environ.get('SDAPI_URL', None)
DEFAULT_PORT: int = 7860
DEFAULT_HOST: str = 'http://127.0.0.1'
try:
  DEFAULT_PORT = int(SD_URL.rsplit(':', 1)[-1]) if SD_URL else DEFAULT_PORT  # pyright: ignore[reportConstantRedefinition]
  DEFAULT_HOST = SD_URL.rsplit(':', 1)[0] if SD_URL else DEFAULT_HOST  # pyright: ignore[reportConstantRedefinition]
except ValueError:
  logging.exception(f'Invalid SDAPI_URL environment variable: {SD_URL!r}; falling back to defaults')

MakeURL: abc.Callable[[str, int], str] = lambda host, port: f'{host}:{port}'


class ImageFormat(enum.Enum):
  """Image format enum."""

  JPEG = 'JPEG'
  PNG = 'PNG'
  GIF = 'GIF'


_PIL_FORMAT_MAP: dict[str, ImageFormat] = {
  'JPEG': ImageFormat.JPEG,
  'PNG': ImageFormat.PNG,
  'GIF': ImageFormat.GIF,
}


class Sampler(enum.Enum):
  """Image generation sampler enum.

  List of all is in:
  <https://github.com/vladmandic/sdnext/blob/master/modules/sd_samplers_diffusers.py>
  """

  UniPC = 'UniPC'
  DDIM = 'DDIM'

  Euler = 'Euler'
  Euler_A = 'Euler a'  # ancestral
  Euler_SGM = 'Euler SGM'
  Euler_EDM = 'Euler EDM'
  Euler_FlowMatch = 'Euler FlowMatch'

  DPM_P = 'DPM++'
  DPM_P_2M = 'DPM++ 2M'
  DPM_P_3M = 'DPM++ 3M'
  DPM_P_1S = 'DPM++ 1S'
  DPM_P_SDE = 'DPM++ SDE'
  DPM_P_2M_SDE = 'DPM++ 2M SDE'
  DPM_P_2M_EDM = 'DPM++ 2M EDM'
  DPM_P_Cosine = 'DPM++ Cosine'
  DPM_SDE = 'DPM SDE'

  DPM_P_Inverse = 'DPM++ Inverse'
  DPM_P_2M_Inverse = 'DPM++ 2M Inverse'
  DPM_P_3M_Inverse = 'DPM++ 3M Inverse'

  UniPC_FlowMatch = 'UniPC FlowMatch'
  DPM2_FlowMatch = 'DPM2 FlowMatch'
  DPM2a_FlowMatch = 'DPM2a FlowMatch'
  DPM2_P_2M_FlowMatch = 'DPM2++ 2M FlowMatch'
  DPM2_P_2S_FlowMatch = 'DPM2++ 2S FlowMatch'
  DPM2_P_SDE_FlowMatch = 'DPM2++ SDE FlowMatch'
  DPM2_P_2M_SDE_FlowMatch = 'DPM2++ 2M SDE FlowMatch'
  DPM2_P_3M_SDE_FlowMatch = 'DPM2++ 3M SDE FlowMatch'

  Heun = 'Heun'
  Heun_FlowMatch = 'Heun FlowMatch'
  LCM = 'LCM'
  LCM_FlowMatch = 'LCM FlowMatch'

  DEIS = 'DEIS'
  SA_Solver = 'SA Solver'
  DC_Solver = 'DC Solver'
  VDM_Solver = 'VDM Solver'
  TCD = 'TCD'
  TDD = 'TDD'
  Flash_FlowMatch = 'Flash FlowMatch'
  PeRFlow = 'PeRFlow'
  UFOGen = 'UFOGen'
  BDIA_DDIM = 'BDIA DDIM'

  PNDM = 'PNDM'
  IPNDM = 'IPNDM'
  DDPM = 'DDPM'
  LMSD = 'LMSD'
  KDPM2 = 'KDPM2'
  KDPM2_a = 'KDPM2 a'  # ancestral
  CMSI = 'CMSI'
  CogX_DDIM = 'CogX DDIM'
  DDIM_Parallel = 'DDIM Parallel'
  DDPM_Parallel = 'DDPM Parallel'

  # ATTENTION! The samplers below are not in the official SDNext codebase,
  # they are legacy A1111 samplers; we must also list them in the SamplerA1111 enum!
  # The reason is that they are not supported in SDNext, so they can be in the DB, but
  # they CANNOT be used for generation, unless they have a converter to a supported sampler
  # in SAMPLER_EQUIVALENCE_A1111_TO_SDNEXT
  DPM_ADAPTIVE = 'DPM adaptive'
  DPM_FAST = 'DPM fast'
  DPM_P_2S_A = 'DPM++ 2S a'  # ancestral
  DPM_P_2S_A_KARRAS = 'DPM++ 2S a Karras'  # ancestral
  DPM_P_2M_KARRAS = 'DPM++ 2M Karras'
  DPM_P_3M_SDE = 'DPM++ 3M SDE'
  DPM_P_3M_SDE_KARRAS = 'DPM++ 3M SDE Karras'


class SamplerA1111(enum.Enum):
  """Image generation sampler from A1111 enum: do NOT have a 1-to-1 mapping to SDNext samplers."""

  DPM_ADAPTIVE = 'DPM adaptive'
  DPM_FAST = 'DPM fast'
  DPM_P_2S_A = 'DPM++ 2S a'
  DPM_P_2S_A_KARRAS = 'DPM++ 2S a Karras'
  DPM_P_2M_KARRAS = 'DPM++ 2M Karras'
  DPM_P_3M_SDE = 'DPM++ 3M SDE'
  DPM_P_3M_SDE_KARRAS = 'DPM++ 3M SDE Karras'


class QueryParser(enum.Enum):
  """Query processing handler enum."""

  SDNextNative = 'native'
  Compel = 'compel'
  XHinker = 'xhinker'
  A1111 = 'a1111'
  Fixed = 'fixed'


class SchedulerSigma(enum.Enum):
  """Sampler sigma schedule enum (`schedulers_sigma`)."""

  default = 'default'
  karras = 'karras'
  betas = 'betas'
  exponential = 'exponential'
  lambdas = 'lambdas'
  flowmatch = 'flowmatch'


class SchedulerSpacing(enum.Enum):
  """Sampler spacing enum (`schedulers_timestep_spacing`)."""

  default = 'default'
  linspace = 'linspace'
  leading = 'leading'
  trailing = 'trailing'


class SchedulerBeta(enum.Enum):
  """Sampler beta schedule enum (`schedulers_beta_schedule`)."""

  default = 'default'
  linear = 'linear'
  scaled = 'scaled'
  cosine = 'cosine'
  sigmoid = 'sigmoid'
  laplace = 'laplace'


class SchedulerPredictionType(enum.Enum):
  """Sampler type enum (`schedulers_prediction_type`)."""

  default = 'default'
  epsilon = 'epsilon'
  sample = 'sample'
  v_prediction = 'v_prediction'
  flow_prediction = 'flow_prediction'


# defaults: what we think are good defaults for most cases
SD_MAX_SEED: int = 2**64 - 1
SD_DEFAULT_VARIATION_STRENGTH: float = 0.5
SD_DEFAULT_ITERATIONS: int = 20
SD_MAX_ITERATIONS: int = 200
SD_DEFAULT_WIDTH: int = 1024
SD_DEFAULT_HEIGHT: int = 1024
SD_DEFAULT_CFG_SCALE: int = 60  # default to 6.0 (multiply by 10 for CLI option)
SD_MAX_CFG_SCALE: int = 300  # max 30.0 (multiply by 10 for CLI option)
SD_DEFAULT_CFG_END: int = 8  # default to 0.8 (end at 80% of the steps; multiply by 10)
SD_DEFAULT_CFG_RESCALE: int = 0  # default to 0.0 (no rescaling; multiply by 100)
SD_DEFAULT_CLIP_SKIP: int = 10  # default to 1.0 (multiply by 10 for CLI option)
SD_MAX_CLIP_SKIP: int = 120  # max 12.0 (multiply by 10 for CLI option)
SD_DEFAULT_FREEU: bool = True  # FreeU enabled by default
SD_MAX_FREEU: float = 300  # 3.0 max FreeU scale (times 100 for CLI option)
SD_DEFAULT_FREEU_B1: int = 105  # FreeU b1 backbone scale default = 1.05 (times 100 for int storage)
SD_DEFAULT_FREEU_B2: int = 110  # FreeU b2 backbone scale default = 1.10 (times 100 for int storage)
SD_DEFAULT_FREEU_S1: int = 75  # FreeU s1 skip scale default = 0.75 (times 100 for int storage)
SD_DEFAULT_FREEU_S2: int = 65  # FreeU s2 skip scale default = 0.65 (times 100 for int storage)
SD_DEFAULT_QUERY_PARSER: QueryParser = QueryParser.A1111
SD_DEFAULT_SAMPLER: Sampler = Sampler.DPM_P_SDE
SD_DEFAULT_DENOISING: int = 50  # IMG2IMG: how much to de-noise 0.5 (multiply by 100 for CLI option)

# empty: for images we import that have some empty metadata, then these are the defaults
# these are strings to be inserted in metadata, so they are NOT multiplied by 10 like CLI options
SD_EMPTY_QUERY_PARSER: str = QueryParser.A1111.value
SD_EMPTY_CFG_END: str = '1.0'  # for empty prompts, default to 1.0 CFG end (i.e., full guidance)
SD_EMPTY_CFG_RESCALE: str = '0.0'  # for empty prompts, default to 0.0 CFG rescale (no rescaling)
SD_EMPTY_CLIP_SKIP: str = '1.0'  # for empty prompts, default to 1.0 CLIP skip
SD_EMPTY_V_SEED: str = '-1'  # for empty prompts, default to -1 variation seed
SD_EMPTY_V_STRENGTH: str = '0'  # for empty prompts, default to 0.0 variation strength

SeedGen: abc.Callable[[], int] = lambda: saferandom.RandInt(2**16 - 1, SD_MAX_SEED)

# basic options

SD_HOST_OPTION: typer.models.OptionInfo = typer.Option(
  DEFAULT_HOST,
  '--host',
  help=(f'Host for the SDNext API; default: {DEFAULT_HOST}'),
)
SD_PORT_OPTION: typer.models.OptionInfo = typer.Option(
  DEFAULT_PORT,
  '-p',
  '--port',
  min=0,
  max=65535,
  help=(f'Port number for the SDNext API; 0 ≤ p ≤ 65535; default: {DEFAULT_PORT}'),
)

SD_API_SERVER_SAVE: typer.models.OptionInfo = typer.Option(
  False,
  '--backup/--no-backup',
  help=(
    'If True, SDNext API server will save a backup copy of the generated images to its '
    'default local storage; default: False (images will only be saved in the TransNext DB)'
  ),
)

SD_FORCE_API: typer.models.OptionInfo = typer.Option(
  False,
  '--force-api/--no-force-api',
  help=(
    'If True, SDNext API server will be required; '
    'if False (default) will still TRY to connect to API, but if not found will proceed standalone'
  ),
)

# DB options

SD_DB_USE_OPTION: typer.models.OptionInfo = typer.Option(
  True,
  '--db/--no-db',
  help=(
    'If True, TransNext will use/update its internal DB; False means it will not load/use DB; '
    'default: True (DB will be used/updated)'
  ),
)
SD_IMAGES_OUTPUT_OPTION: typer.models.OptionInfo = typer.Option(
  None,
  '-o',
  '--out',
  exists=True,
  file_okay=False,
  dir_okay=True,
  readable=True,
  help=(
    'The local output root directory path, ex: "~/foo/bar/"; '
    'will create sub-directories in this directory in the format YYYY-MM-DD for the '
    'days where images are generated; if you do not use DB (i.e., `--no-db`) this is '
    'mandatory, but with the DB activated it will store the last used output and re-use it; '
    'default: with `--db` default is last used, with `--no-db` no default and you must provide it'
  ),
)

# image generation options

SD_POSITIVE_PROMPT_OPTION: typer.models.ArgumentInfo = typer.Argument(
  ..., help='Query input string to guide the image generation, positive prompt; "user prompt"'
)
SD_NEGATIVE_PROMPT_OPTION: typer.models.OptionInfo = typer.Option(
  None,
  '-n',
  '--negative',
  help=(
    'Negative prompt to guide the image generation; "negative prompt"; default: no negative prompt'
  ),
)

SD_STEPS_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_ITERATIONS,
  '-i',
  '--iterations',
  min=1,
  max=SD_MAX_ITERATIONS,
  help=(
    f'Number of steps (iterations) for the image generation; 1 ≤ i ≤ 200; '
    f'default: {SD_DEFAULT_ITERATIONS}'
  ),
)

SD_SEED_OPTION: typer.models.OptionInfo = typer.Option(
  None,
  '-s',
  '--seed',
  min=1,
  max=SD_MAX_SEED,
  help=(
    f'Seed for the image generation; 0 < s ≤ {SD_MAX_SEED}; '
    'if not provided (default), a random seed will be used'
  ),
)
SD_VARIATION_SEED_OPTION: typer.models.OptionInfo = typer.Option(
  None,
  '--vseed',
  min=1,
  max=SD_MAX_SEED,
  help=(
    f'Variation seed for the image generation; 0 < s ≤ {SD_MAX_SEED}; '
    'if not provided (default) variation seeds will not be used'
  ),
)
SD_VARIATION_STRENGTH_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_VARIATION_STRENGTH,
  '--vstrength',
  min=0.0,
  max=1.0,
  help=(
    'Variation strength for the image generation, i.e., '
    'how much to mix the variation seed with the base (regular) seed; 0.0 ≤ s ≤ 1.0; '
    f'default: {SD_DEFAULT_VARIATION_STRENGTH}; only used if variation seed is provided'
  ),
)

SD_WIDTH_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_WIDTH,
  '-w',
  '--width',
  min=16,
  max=4096,
  help=f'Width of the generated image; 16 ≤ i ≤ 4096, multiple of 8; default: {SD_DEFAULT_WIDTH}',
)
SD_HEIGHT_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_HEIGHT,
  '-h',
  '--height',
  min=16,
  max=4096,
  help=f'Height of the generated image; 16 ≤ i ≤ 4096, multiple of 8; default: {SD_DEFAULT_HEIGHT}',
)

SD_SAMPLER_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_SAMPLER,
  '--sampler',
  help=f'Sampler to use for the generation; default: {SD_DEFAULT_SAMPLER.value!r}',
)

SD_QUERY_PARSER_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_QUERY_PARSER,
  '--parser',
  help=f'Query parser to use for the generation; default: {SD_DEFAULT_QUERY_PARSER.value!r}',
)

SD_MODEL_KEY_OPTION: typer.models.OptionInfo = typer.Option(
  'XLB_v10',
  '-m',
  '--model',
  help='Model key to use for the generation; default: "_v10VAEFix"',
)

SD_CLIP_SKIP_OPTION: typer.models.OptionInfo = typer.Option(  # TODO: float in future
  SD_DEFAULT_CLIP_SKIP // 10,  # remember to convert for CLI option
  '--clip',
  min=1,
  max=SD_MAX_CLIP_SKIP // 10,  # remember to convert for CLI option
  help=(
    f'Clip skip value; 1 ≤ c ≤ {SD_MAX_CLIP_SKIP // 10}; default: {SD_DEFAULT_CLIP_SKIP // 10}'
  ),
)

SD_CFG_SCALE_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_CFG_SCALE / 10,
  '-g',
  '--cfg',
  min=1.0,
  max=SD_MAX_CFG_SCALE / 10,
  help=(
    f'CFG scale value (guidance scale); 1.0 ≤ c ≤ {SD_MAX_CFG_SCALE / 10}; '
    f'default: {SD_DEFAULT_CFG_SCALE / 10}'
  ),
)
SD_CFG_END_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_CFG_END / 10,
  '--cfg-end',
  min=0.0,
  max=1.0,
  help=(
    f'CFG scale application end (guidance end); 0.0 ≤ c ≤ 1.0; default: {SD_DEFAULT_CFG_END / 10}'
  ),
)
SD_CFG_RESCALE_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_CFG_RESCALE / 100,
  '--cfg-rescale',
  min=0.0,
  max=1.0,
  help=(
    'Adjusts the CFG guided result to reduce the tendency of high CFG to cause overexposure / '
    'oversaturation / burned highlights / harsh color shifts; '
    'you usually only want this for higher CFG scales `-g/--cfg` (e.g., > 7.0); '
    f'0.0 ≤ r ≤ 1.0; default: {SD_DEFAULT_CFG_RESCALE / 100}'
  ),
)

SD_FREEU_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_FREEU,
  '--freeu/--no-freeu',
  help=f'Enable/disable FreeU backbone and skip feature scaling; default: {SD_DEFAULT_FREEU}',
)
SD_FREEU_B1_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_FREEU_B1 / 100,
  '--b1',
  min=0.0,
  max=SD_MAX_FREEU / 100,
  help=(
    f'FreeU b1 backbone feature scale; 0.0 ≤ b ≤ {SD_MAX_FREEU / 100}; '
    f'default: {SD_DEFAULT_FREEU_B1 / 100}'
  ),
)
SD_FREEU_B2_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_FREEU_B2 / 100,
  '--b2',
  min=0.0,
  max=SD_MAX_FREEU / 100,
  help=(
    f'FreeU b2 backbone feature scale; 0.0 ≤ b ≤ {SD_MAX_FREEU / 100}; '
    f'default: {SD_DEFAULT_FREEU_B2 / 100}'
  ),
)
SD_FREEU_S1_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_FREEU_S1 / 100,
  '--s1',
  min=0.0,
  max=SD_MAX_FREEU / 100,
  help=(
    f'FreeU s1 skip feature scale; 0.0 ≤ s ≤ {SD_MAX_FREEU / 100}; '
    'reduce over-smoothing / unnatural detail; '
    f'default: {SD_DEFAULT_FREEU_S1 / 100}'
  ),
)
SD_FREEU_S2_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_FREEU_S2 / 100,
  '--s2',
  min=0.0,
  max=SD_MAX_FREEU / 100,
  help=(
    f'FreeU s2 skip feature scale; 0.0 ≤ s ≤ {SD_MAX_FREEU / 100}; '
    'reduce over-smoothing / unnatural detail; '
    f'default: {SD_DEFAULT_FREEU_S2 / 100}'
  ),
)

SD_SCHEDULER_SIGMA_OPTION: typer.models.OptionInfo = typer.Option(
  None,
  '--sigma',
  help='Sampler sigma schedule to use for the generation; default: None (SDNext default)',
)
SD_SCHEDULER_SPACING_OPTION: typer.models.OptionInfo = typer.Option(
  None,
  '--spacing',
  help='Sampler spacing schedule to use for the generation; default: None (SDNext default)',
)
SD_SCHEDULER_BETA_OPTION: typer.models.OptionInfo = typer.Option(
  None,
  '--beta',
  help='Sampler beta schedule to use for the generation; default: None (SDNext default)',
)
SD_SCHEDULER_PREDICTION_TYPE_OPTION: typer.models.OptionInfo = typer.Option(
  None,
  '--prediction',
  help='Sampler prediction type to use for the generation; default: None (SDNext default)',
)


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class TransNextConfig(clibase.CLIConfig):
  """TransNext CLI global context, storing the configuration."""

  host: str
  port: int
  db: bool
  output: pathlib.Path | None


def PromptHash(positive: str, negative: str | None = None) -> str:
  """Hash a prompt with positive and negative parts.

  The hash is SHA256(positive + 2 zero bytes [+ negative])

  Args:
    positive: The positive prompt string.
    negative: The negative prompt string, optional.

  Returns:
    A string hash of the prompt, derived from the positive and negative parts.

  """
  hash_str: bytes = positive.encode('utf-8') + b'\x00\x00'
  if negative is not None:
    hash_str += negative.encode('utf-8')
  return hashes.Hash256(hash_str).hex()[-12:]


def GetFileCreation(path: pathlib.Path) -> int:
  """Get the file creation time as an integer timestamp.

  `st_birthtime` is available on some platforms (e.g., macOS), but not on others (e.g., Linux), so
  we fall back to `st_mtime` if `st_birthtime` is not available.

  Args:
    path: The path to the file.

  Returns:
    The file creation time as an integer timestamp.

  Raises:
    Error: If the file does not exist

  """
  if not path.exists():
    raise Error(f'File not found: {path}')
  stat_result: os.stat_result = path.stat()
  return int(getattr(stat_result, 'st_birthtime', stat_result.st_mtime))


def GetBasicDataFromImage(img_bytes: bytes) -> tuple[ImageFormat, int, int, str, str | None]:
  """Get basic data from an image, including format, size, hash, and metadata text.

  Args:
    img_bytes: The image data as bytes.

  Returns:
    (format, width, height, hash, metadata_text) where:
      - format: The image format as an ImageFormat enum.
      - width: The width of the image in pixels.
      - height: The height of the image in pixels.
      - hash: A hash of the image data (SHA256 of RGBA bytes).
      - metadata_text: The extracted metadata text from the image, if available; otherwise None.

  Raises:
    Error: If the image format is unsupported or if there are issues processing the image.

  """
  with Image.open(io.BytesIO(img_bytes)) as img:
    # make sure format is known
    fmt: ImageFormat | None = _PIL_FORMAT_MAP.get((img.format or '').upper())
    if not fmt:
      raise Error(f'Unsupported image format {img.format!r}')
    # get the internal data we need (size and hash)
    width: int = img.width
    height: int = img.height
    if width < 1 or height < 1:
      raise Error(f'Invalid image size {width}x{height}')
    raw_hash: str = hashes.Hash256(img.convert('RGBA').tobytes()).hex()
    # try to extract metadata from PNG info tags, either 'parameters' or 'UserComment'
    info_text: str = ''
    pil_info: tbase.JSONDict = img.info  # type: ignore[assignment]
    if 'parameters' in pil_info and isinstance(pil_info['parameters'], str):
      info_text = pil_info['parameters'].strip()
    elif 'UserComment' in pil_info and isinstance(pil_info['UserComment'], str):
      info_text = pil_info['UserComment'].strip()
  return (fmt, width, height, raw_hash, info_text or None)


_LORA_RE: re.Pattern[str] = re.compile(
  r'<(?P<kind>lora|lyco|lycora|lycoris):(?P<name>[^:>]+):(?P<strength>[^>]+)>'
)

LoraExtract: abc.Callable[[str], dict[str, tuple[str, str]]] = lambda q: {
  m['name'].lower().strip(): (m['kind'], m['strength'].strip()) for m in _LORA_RE.finditer(q)
}


def FindModelHash(tp: str, partial_hash: str, partial_name: str, models: dict[str, str]) -> str:
  """Find a full model hash in the DB by prefix-matching a partial hash or looking at names.

  Args:
    tp: The type of model we are looking for, for error messages (e.g., "model" or "lora")
    partial_hash: A partial (prefix) hash string to match against DB model hashes.
    partial_name: A partial (prefix) name string to match against DB model names/aliases.
    models: The DB models like {hash: model_name/alias}.

  Returns:
    The full model hash if a unique prefix match is found, or an empty string if not found

  Raises:
    Error: on error

  """
  # check not empty
  tp = tp.strip().lower()
  if tp not in {'model', 'lora'}:
    raise Error(f'invalid `tp`: {tp!r}')
  partial_hash, partial_name = partial_hash.strip().lower(), partial_name.strip().lower()
  if not partial_hash.strip() and not partial_name.strip():
    raise Error(f'{tp} empty query')
  hash_matches: list[str] = []
  if partial_hash:
    # check for exact HASH match, the best of all
    if partial_hash in models:
      return partial_hash  # exact match
    # check for partial hash match, second best, but still very confident
    hash_matches = [h for h in models if h.lower().startswith(partial_hash)]
    if len(hash_matches) == 1:
      logging.debug(f'Matched partial {tp} hash {partial_hash!r} -> {hash_matches[0]!r}')
      return hash_matches[0]  # found!
  # check for partial name/alias match if we have a name, not super reliable:
  # (could be a different version of the same model, for example)
  name_matches: list[str] = []
  if partial_name:
    name_matches = [h for h, v in models.items() if partial_name in v.lower()]
    if len(name_matches) == 1:
      logging.debug(f'Matched partial {tp} name {partial_name!r} -> {name_matches[0]!r}')
      return name_matches[0]  # found!
  # no match found, this is an error
  if len(hash_matches) > 1 or len(name_matches) > 1:
    raise Error(f'ambiguous {tp} #{partial_hash}/{partial_name}: {hash_matches}/{name_matches}')
  raise Error(f'{tp} #{partial_hash}/{partial_name} not found')
