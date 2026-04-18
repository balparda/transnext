# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""CLI: Make commands."""

from __future__ import annotations

import pathlib

import click
from transcrypto.cli import clibase

from transnext import gen
from transnext.core import base, db, sdnapi

# ===>>>> TO DEBUG calls: set True and run the CLI command
_DEBUG_RECORD: bool = False  # keep False!
# the result will be saved here:
_DEBUG_RECORD_SAVE_PATH: pathlib.Path = pathlib.Path('call_record.json')  # project root! careful!


@gen.app.command(
  'make',
  help='Query the model.',
  epilog=(
    'Example:\n\n\n\n'
    'poetry run gen -vv --out ~/foo/bar make "dark knight" -n batman '
    '--cfg 7.5 -m SDXL_model_1234 -i 30 --sampler "Euler a"'
  ),
)
@clibase.CLIErrorGuard
def Make(  # documentation is help/epilog/args # noqa: D103
  *,
  ctx: click.Context,
  positive_prompt: str = base.SD_POSITIVE_PROMPT_OPTION,  # type: ignore[assignment]
  negative_prompt: str | None = base.SD_NEGATIVE_PROMPT_OPTION,  # type: ignore[assignment]
  steps: int = base.SD_STEPS_OPTION,  # type: ignore[assignment]
  seed: int | None = base.SD_SEED_OPTION,  # type: ignore[assignment]
  vseed: int | None = base.SD_VARIATION_SEED_OPTION,  # type: ignore[assignment]
  vstrength: float = base.SD_VARIATION_STRENGTH_OPTION,  # type: ignore[assignment]
  width: int = base.SD_WIDTH_OPTION,  # type: ignore[assignment]
  height: int = base.SD_HEIGHT_OPTION,  # type: ignore[assignment]
  sampler: base.Sampler = base.SD_SAMPLER_OPTION,  # type: ignore[assignment]
  parser: base.QueryParser = base.SD_QUERY_PARSER_OPTION,  # type: ignore[assignment]
  model_key: str = base.SD_MODEL_KEY_OPTION,  # type: ignore[assignment]
  clip_skip: int = base.SD_CLIP_SKIP_OPTION,  # type: ignore[assignment]  # TODO: float in future
  cfg_scale: float = base.SD_CFG_SCALE_OPTION,  # type: ignore[assignment]
  cfg_end: float = base.SD_CFG_END_OPTION,  # type: ignore[assignment]
  cfg_rescale: float = base.SD_CFG_RESCALE_OPTION,  # type: ignore[assignment]
  sch_sigma: base.SchedulerSigma | None = base.SD_SCHEDULER_SIGMA_OPTION,  # type: ignore[assignment]
  sch_spacing: base.SchedulerSpacing | None = base.SD_SCHEDULER_SPACING_OPTION,  # type: ignore[assignment]
  sch_beta: base.SchedulerBeta | None = base.SD_SCHEDULER_BETA_OPTION,  # type: ignore[assignment]
  sch_type: base.SchedulerPredictionType | None = base.SD_SCHEDULER_PREDICTION_TYPE_OPTION,  # type: ignore[assignment]
  freeu_enabled: bool = base.SD_FREEU_OPTION,  # type: ignore[assignment]
  freeu_b1: float = base.SD_FREEU_B1_OPTION,  # type: ignore[assignment]
  freeu_b2: float = base.SD_FREEU_B2_OPTION,  # type: ignore[assignment]
  freeu_s1: float = base.SD_FREEU_S1_OPTION,  # type: ignore[assignment]
  freeu_s2: float = base.SD_FREEU_S2_OPTION,  # type: ignore[assignment]
  backup: bool = base.SD_API_SERVER_SAVE,  # type: ignore[assignment]
  redo: bool = base.SD_REDO_OPTION,  # type: ignore[assignment]
) -> None:
  # check sanity
  config: gen.GenConfig = ctx.obj
  if not config.db and not config.output:
    raise click.UsageError('With `--no-db` you must specify `--out`')
  if sampler.value in {s.value for s in base.SamplerA1111}:
    raise click.UsageError(
      f'Sampler {sampler.value!r} not supported by SDNext (it was valid in A1111 but not in SDNext)'
    )
  # open API and DB
  api = sdnapi.API(
    base.MakeURL(config.host, config.port), server_save_images=backup, record=_DEBUG_RECORD
  )
  with db.AIDatabase(config.appconfig, read_only=not config.db, api=api) as ai_db:
    # set output, if specified
    if config.output is not None:
      ai_db.output = config.output
    # run the generation and store the result in the DB
    ai_db.Txt2Img(
      db.AIMetaTypeFactory(
        {
          'positive': positive_prompt,
          'negative': negative_prompt,
          'steps': steps,
          'seed': seed,
          'v_seed': (vseed, round(vstrength * 100)) if vseed and vstrength > 0.0 else None,
          'width': width,
          'height': height,
          'sampler': sampler.value,
          'parser': parser.value,
          'model_hash': ai_db.GetModelHash(model_key, api=api),
          'clip_skip': round(clip_skip * 10),  # store as int (times 10)
          'cfg_scale': round(cfg_scale * 10),  # store as int (times 10)
          'cfg_end': round(cfg_end * 10),  # store as int (times 10)
          'cfg_rescale': round(cfg_rescale * 100),  # store as int (times 100)
          'sch_sigma': sch_sigma.value if sch_sigma else None,
          'sch_spacing': sch_spacing.value if sch_spacing else None,
          'sch_beta': sch_beta.value if sch_beta else None,
          'sch_type': sch_type.value if sch_type else None,
          'freeu': (
            round(freeu_b1 * 100),  # store as int (times 100)
            round(freeu_b2 * 100),  # store as int (times 100)
            round(freeu_s1 * 100),  # store as int (times 100)
            round(freeu_s2 * 100),  # store as int (times 100)
          )
          if freeu_enabled
          else None,
        }
      ),
      api,
      redo=redo,
    )
  # DB is closed and saved
  config.console.print()
  # debug only!
  if _DEBUG_RECORD:
    # this is debug only, we don't want tests here!
    api.SaveRecordToFile(_DEBUG_RECORD_SAVE_PATH)  # pragma: no cover
    raise click.UsageError('dont forget to set _DEBUG_RECORD to False!')  # pragma: no cover
