# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""CLI: Make commands."""

from __future__ import annotations

import click
from transcrypto.cli import clibase

from transnext import gen
from transnext.core import base, db, sdnapi


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
  width: int = base.SD_WIDTH_OPTION,  # type: ignore[assignment]
  height: int = base.SD_HEIGHT_OPTION,  # type: ignore[assignment]
  sampler: base.Sampler = base.SD_SAMPLER_OPTION,  # type: ignore[assignment]
  parser: base.QueryParser = base.SD_QUERY_PARSER_OPTION,  # type: ignore[assignment]
  model_key: str = base.SD_MODEL_KEY_OPTION,  # type: ignore[assignment]
  clip_skip: int = base.SD_CLIP_SKIP_OPTION,  # type: ignore[assignment]  # TODO: float in future
  cfg_scale: float = base.SD_CFG_SCALE_OPTION,  # type: ignore[assignment]
  cfg_end: float = base.SD_CFG_END_OPTION,  # type: ignore[assignment]
  backup: bool = base.SD_API_SERVER_SAVE,  # type: ignore[assignment]
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
  api = sdnapi.API(base.MakeURL(config.host, config.port), server_save_images=backup)
  with db.AIDatabase(config.appconfig, read_only=not config.db) as ai_db:
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
          'width': width,
          'height': height,
          'sampler': sampler.value,
          'parser': parser.value,
          'model_hash': ai_db.GetModelHash(model_key, api=api),
          'clip_skip': round(clip_skip * 10),  # store as int (times 10)
          'cfg_scale': round(cfg_scale * 10),  # store as int (times 10)
          'cfg_end': round(cfg_end * 10),  # store as int (times 10)
        }
      ),
      api,
    )
  # DB is closed and saved
  config.console.print()
