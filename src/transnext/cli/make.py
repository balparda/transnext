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
    'poetry run gen make "What is the capital of France?"\n\n'
    'poetry run gen make --no-lms "Give me an onion soup recipe."'
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
  clip_skip: int = base.SD_CLIP_SKIP_OPTION,  # type: ignore[assignment]
  cfg_scale: float = base.SD_CFG_SCALE_OPTION,  # type: ignore[assignment]
  cfg_end: float = base.SD_CFG_END_OPTION,  # type: ignore[assignment]
  backup: bool = base.SD_API_SERVER_SAVE,  # type: ignore[assignment]
) -> None:
  config: gen.GenConfig = ctx.obj
  api = sdnapi.API(base.MakeURL(config.host, config.port), server_save_images=backup)
  if not config.db and not config.output:
    raise click.UsageError('With `--no-db` you must specify `--out`')
  api.GetModels()
  with db.AIDatabase(config.appconfig, read_only=not config.db) as ai_db:
    if config.output is not None:
      ai_db.output = config.output
    ai_db.Txt2Img(
      api,
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
          'clip_skip': clip_skip,
          'cfg_scale': int(round(cfg_scale * 10)),  # store as int (times 10)  # noqa: RUF046
          'cfg_end': int(round(cfg_end * 10)),  # store as int (times 10)  # noqa: RUF046
        }
      ),
    )

  # DB is closed and saved
  config.console.print()
