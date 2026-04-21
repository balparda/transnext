# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""CLI: Experiment commands."""

from __future__ import annotations

import logging
import pathlib

import click
from transcrypto.cli import clibase

from transnext import experiment
from transnext.core import base, db, newton, sdnapi

# ===>>>> TO DEBUG calls: set True and run the CLI command
_DEBUG_RECORD: bool = False  # keep False!
# the result will be saved here:
_DEBUG_RECORD_SAVE_PATH: pathlib.Path = pathlib.Path('experiment_record.json')  # root! careful!


@experiment.app.command(
  'new',
  help=(
    'Create and run a new experiment. '
    'An experiment varies one or more axes (CFG, sampler, model, positive/negative prompt) '
    'across a set of seed values. Provide at least one --axis and --seeds. '
    'The order of --axis options on the command line determines the axis order in the experiment.'
  ),
  epilog=(
    'Examples:\n\n\n\n'
    '$ poetry run experiment new "a photo of a cat" --seeds "666|-1|999" '
    '--axis "sampler:Euler|DPM++ SDE"\n\n'
    '<<runs 2x2 grid: 2 samplers x 2 seeds>>\n\n\n\n'
    '$ poetry run experiment new "a photo of a % animal" --seeds "42" '
    '--axis "positive:black cat|white dog" --axis "cfg_scale:6.0|9.0"\n\n'
    '<<runs 2x2x1 grid: 2 prompts x 2 CFGs x 1 seed>>'
  ),
)
@clibase.CLIErrorGuard
def New(  # documentation is help/epilog/args # noqa: D103
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
  seeds_raw: str = base.SD_EXPERIMENT_SEEDS_OPTION,  # type: ignore[assignment]
  raw_axes: list[str] = base.SD_EXPERIMENT_AXIS_OPTION,  # type: ignore[assignment]
  respect_vae: bool = base.SD_EXPERIMENT_RESPECT_VAE_OPTION,  # type: ignore[assignment]
  respect_pony: bool = base.SD_EXPERIMENT_RESPECT_PONY_OPTION,  # type: ignore[assignment]
  respect_clip2: bool = base.SD_EXPERIMENT_RESPECT_CLIP2_OPTION,  # type: ignore[assignment]
) -> None:
  # check sanity
  config: experiment.ExperimentConfig = ctx.obj
  if not config.db and not config.output:
    raise click.UsageError('With `--no-db` you must specify `--out`')
  if sampler.value in {s.value for s in base.SamplerA1111}:
    raise click.UsageError(
      f'Sampler {sampler.value!r} not supported by SDNext (it was valid in A1111 but not in SDNext)'
    )
  # parse experiment seeds and axes from CLI input
  seed_list: list[int] = [
    (s if s >= 1 else base.SeedGen())  # replace -1s with random seeds
    for s in base.ParseIntList(seeds_raw, name='--seeds', min_val=-1, max_val=base.SD_MAX_SEED)
  ]
  axes: list[newton.AxisType] = _BuildAxes(raw_axes)
  # open API and DB
  api = sdnapi.API(
    base.MakeURL(config.host, config.port), server_save_images=backup, record=_DEBUG_RECORD
  )
  with db.AIDatabase(
    config.appconfig, read_only=not config.db, sidecar=config.sidecar, api=api
  ) as ai_db:
    # set output, if specified
    if config.output is not None:
      ai_db.output = config.output
    exps = newton.Experiments(ai_db)
    exp: newton.Experiment = exps.Make(
      db.AIMetaTypeFactory(
        {
          'positive': positive_prompt,
          'negative': negative_prompt,
          'steps': steps,
          'seed': seed,
          'v_seed': (
            db.AIMetaVariationSeedType(seed=vseed, percent=round(vstrength * 100))
            if vseed and vstrength > 0.0
            else None
          ),
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
            (
              db.AIMetaFreeUType(
                b1=round(freeu_b1 * 100),  # store as int (times 100)
                b2=round(freeu_b2 * 100),  # store as int (times 100)
                s1=round(freeu_s1 * 100),  # store as int (times 100)
                s2=round(freeu_s2 * 100),  # store as int (times 100)
              )
            )
            if freeu_enabled
            else None
          ),
        }
      ),
      axes,
      seed_list,
      newton.ExperimentOptionsType(
        respect_vae=respect_vae,
        respect_pony=respect_pony,
        respect_clip2=respect_clip2,
      ),
    )
    for _ in exp.Run(api, redo=redo):
      pass
  # DB is closed and saved
  config.console.print()
  # debug only!
  if _DEBUG_RECORD:
    # this is debug only, we don't want tests here!
    api.SaveRecordToFile(_DEBUG_RECORD_SAVE_PATH)  # pragma: no cover
    raise click.UsageError('dont forget to set _DEBUG_RECORD to False!')  # pragma: no cover


def _BuildAxes(raw_axes: list[str]) -> list[newton.AxisType]:
  """Build experiment axes from repeatable --axis CLI option strings.

  Each raw string has the format "KEY:VAL1|VAL2|...". The order of the list
  determines the order of the experiment axes. Duplicate keys are rejected.

  Args:
    raw_axes: List of raw axis definition strings, in the user-provided order.

  Returns:
    A list of `newton.AxisType` objects for the experiment, in CLI order.

  Raises:
    click.UsageError: If no axes are provided or a key is duplicated.

  """
  if not raw_axes:
    raise click.UsageError(
      f'At least one --axis is required; valid keys: {", ".join(sorted(base.AXIS_KEYS))}'
    )
  axes: list[newton.AxisType] = []
  seen_keys: set[str] = set()
  for raw in raw_axes:
    key, values = base.ParseAxisDefinition(raw)
    if key in seen_keys:
      raise click.UsageError(f'Duplicate --axis key {key!r}; each axis can only appear once')
    seen_keys.add(key)
    axes.append(newton.AxisType(key=key, values=values))
  logging.info(
    f'Experiment axes ({len(axes)}): '  # noqa: G003
    + ', '.join(f'{a["key"]}({len(a["values"])} values)' for a in axes)
  )
  return axes
