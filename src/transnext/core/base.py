# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""TransNext core base module."""

from __future__ import annotations

import dataclasses
import enum
import logging
import os
import pathlib
from collections import abc

import typer
from transai.core import ai
from transcrypto.cli import clibase
from transcrypto.core import hashes
from transcrypto.utils import base


class Error(base.Error):
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


class Sampler(enum.Enum):
  """Image generation sampler enum.

  List of all is in:
  <https://github.com/vladmandic/sdnext/blob/master/modules/sd_samplers_diffusers.py>
  """

  UniPC = 'UniPC'
  DDIM = 'DDIM'

  Euler = 'Euler'
  Euler_A = 'Euler a'
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
  KDPM2_a = 'KDPM2 a'
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
  DPM_P_2S_A = 'DPM++ 2S a'
  DPM_P_2S_A_KARRAS = 'DPM++ 2S a Karras'
  DPM_P_3M_SDE = 'DPM++ 3M SDE'
  DPM_P_3M_SDE_KARRAS = 'DPM++ 3M SDE Karras'


class SamplerA1111(enum.Enum):
  """Image generation sampler from A1111 enum."""

  DPM_ADAPTIVE = 'DPM adaptive'
  DPM_FAST = 'DPM fast'
  DPM_P_2S_A = 'DPM++ 2S a'
  DPM_P_2S_A_KARRAS = 'DPM++ 2S a Karras'
  DPM_P_3M_SDE = 'DPM++ 3M SDE'
  DPM_P_3M_SDE_KARRAS = 'DPM++ 3M SDE Karras'


SAMPLER_EQUIVALENCE_A1111_TO_SDNEXT: dict[SamplerA1111, Sampler] = {
  # none for now
}


class QueryParser(enum.Enum):
  """Query processing handler enum."""

  SDNextNative = 'native'
  Compel = 'compel'
  XHinker = 'xhinker'
  A1111 = 'a1111'
  Fixed = 'fixed'


# defaults: what we think are good defaults for most cases
SD_DEFAULT_ITERATIONS: int = 20
SD_MAX_ITERATIONS: int = 200
SD_DEFAULT_WIDTH: int = 512
SD_DEFAULT_HEIGHT: int = 512
SD_DEFAULT_CFG_SCALE: int = 60  # default to 6.0 (multiply by 10 for CLI option)
SD_MAX_CFG_SCALE: int = 300  # max 30.0 (multiply by 10 for CLI option)
SD_DEFAULT_CFG_END: int = 8  # default to 0.8 (end at 80% of the steps; multiply by 10)
SD_DEFAULT_CLIP_SKIP: int = 10  # default to 1.0 (multiply by 10 for CLI option)
SD_MAX_CLIP_SKIP: int = 50  # max 5.0 (multiply by 10 for CLI option)
SD_DEFAULT_QUERY_PARSER: QueryParser = QueryParser.A1111
SD_DEFAULT_SAMPLER: Sampler = Sampler.DPM_P_SDE

# empty: for images we import that have some empty metadata, then these are the defaults
# these are strings to be inserted in metadata, so they are NOT multiplied by 10 like CLI options
SD_EMPTY_QUERY_PARSER: str = QueryParser.A1111.value
SD_EMPTY_CFG_END: str = '1.0'  # for empty prompts, default to 1.0 CFG end (i.e., full guidance)
SD_EMPTY_CLIP_SKIP: str = '1.0'  # for empty prompts, default to 1.0 CLIP skip

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
  min=2,
  max=ai.AI_MAX_SEED,
  help=(
    f'Seed for the image generation; 1 < s ≤ {ai.AI_MAX_SEED}; '
    'if not provided (default), a random seed will be used'
  ),
)

SD_WIDTH_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_WIDTH,
  '-w',
  '--width',
  min=16,
  max=4096,
  help=f'Width of the generated image; 16 ≤ i ≤ 4096; default: {SD_DEFAULT_WIDTH}',
)
SD_HEIGHT_OPTION: typer.models.OptionInfo = typer.Option(
  SD_DEFAULT_HEIGHT,
  '-h',
  '--height',
  min=16,
  max=4096,
  help=f'Height of the generated image; 16 ≤ i ≤ 4096; default: {SD_DEFAULT_HEIGHT}',
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
