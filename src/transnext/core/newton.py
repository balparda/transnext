# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""TransNext core experiments module."""

from __future__ import annotations

import copy
import enum
import io
import itertools
import logging
import pathlib
import time
from collections import abc
from typing import Protocol, TypedDict, cast, overload

from PIL import Image, ImageDraw, ImageFont
from transcrypto.core import hashes
from transcrypto.utils import base as tbase
from transcrypto.utils import timer

from transnext import __version__
from transnext.core import base, db

_SAVE_EVERY: int = 5  # save to DB every N images generated, to avoid losing too much progress


class Error(base.Error):
  """TransNext experiments exception."""


type ExperimentKeyType = list[str | int]  # experiment instance [ax1_value, ax2_value, ...]
type AxisKindType = str | int  # axis value type
type ExperimentKeysType = dict[str, str | None]  # hash is the key b/c JSON can't have tuple keys
type ResultsType = dict[str, ExperimentKeysType]  # {str(seed): ExperimentKeysType}
type AxisMapType = dict[AxisField, tuple[type[AxisKindType], AxisFnType, AxisFnType]]


class AxisFnType(Protocol):
  """Protocol for axis value functions: preserves the input type (int→int, str→str)."""

  @overload
  def __call__(self, v: int) -> int: ...
  @overload
  def __call__(self, v: str) -> str: ...
  def __call__(self, v: AxisKindType) -> AxisKindType:
    """Call the function with the given value, preserving the input type."""
    ...


class AxisField(enum.Enum):
  """Experiment axis field enum."""

  Seed = 'seed'
  Model = 'model_hash'
  CFG = 'cfg_scale'
  Sampler = 'sampler'
  Positive = 'positive'
  Negative = 'negative'


class ExperimentType(TypedDict):
  """Experiment object type."""

  hash: str  # experiment hash, based on config and axes
  config: db.AIMetaType  # base config to vary along axes
  axes: list[AxisType]  # experiment axes to vary along
  options: ExperimentOptionsType  # options for the experiment that affect how axes behave
  results: ResultsType  # {str(seed): {experiment_key: result_hash}} (str key b/c JSON limitation)


class ExperimentOptionsType(TypedDict):
  """Experiment options type."""

  respect_vae: bool  # accept override of VAE option?
  respect_pony: bool  # accept override of Pony option?
  respect_clip2: bool  # accept override of CLIP2 option?


class AxisType(TypedDict):
  """Experiment axis object type."""

  key: str  # AxisField axis key (e.g. "model", "sampler", "cfg_scale")
  values: list[str] | list[int]  # axis values to iterate over (e.g. ["Euler", "DPMPP2MSampler"])


def Tunnel[T: AxisKindType](v: T) -> T:
  """Pass-through tunnel for any type T. Returns the input."""  # noqa: DOC201
  return v


def ReplaceMe[T: AxisKindType](v: T) -> T:
  """Pass-through tunnel for any type T. Returns the input."""
  raise NotImplementedError('This function is a placeholder and should be replaced')


TUNNEL: AxisFnType = cast('AxisFnType', Tunnel)
REPLACE_ME: AxisFnType = cast('AxisFnType', ReplaceMe)


def IntRangeTunnel(min_value: int, max_value: int) -> abc.Callable[[int], int]:
  """Pass-through tunnel for int within a range [min, max].

  Args:
    min_value: Minimum allowed value (inclusive).
    max_value: Maximum allowed value (inclusive).

  Returns:
    A function that takes an int and returns it if it's within the range, otherwise raises Error.

  """

  def _Tunnel(v: int) -> int:
    if not (min_value <= v <= max_value):
      raise Error(f'Value {v} is out of range [{min_value}, {max_value}]')
    return v

  return _Tunnel


_PROMPT_PLACEHOLDER: str = '%'
_IMPOSSIBLE_VALUE: str = 'b76b6a6aaf0fb66156fdec3f27777d396dd16c49e1767dd7cfca724c17eed4ee'


def PromptReplaceTunnel(prompt: str) -> abc.Callable[[str], str]:
  """Prompt replacement tunnel for str prompts with a placeholder.

  Args:
    prompt: The prompt string containing the placeholder. Ex:
        "A photo of a % cat" with placeholder "%" will be replaced with the axis value;
        every "%%" will be replaced with a single "%" if you need that

  Returns:
    A function that takes a string and replaces the placeholder with the given string.

  Raises:
    Error: if the prompt does not contain the required placeholder.

  """
  if _PROMPT_PLACEHOLDER not in prompt:
    raise Error(f'Prompt must contain 1+ placeholders {_PROMPT_PLACEHOLDER!r} for replacement')

  def _Tunnel(v: str) -> str:
    # replace all %% -> BLOB
    p2: str = prompt.replace(_PROMPT_PLACEHOLDER + _PROMPT_PLACEHOLDER, _IMPOSSIBLE_VALUE)
    # replace all % -> v
    p2 = p2.replace(_PROMPT_PLACEHOLDER, v)
    # replace all BLOB -> %
    return p2.replace(_IMPOSSIBLE_VALUE, _PROMPT_PLACEHOLDER)

  return _Tunnel


SeedValidate: AxisFnType = cast('AxisFnType', IntRangeTunnel(1, base.SD_MAX_SEED))
SamplerValidate: abc.Callable[[str], str] = lambda s: base.Sampler(s).value
CFGValidate: AxisFnType = cast(
  'AxisFnType', IntRangeTunnel(base.SD_MIN_CFG_SCALE, base.SD_MAX_CFG_SCALE)
)


_AXIS_MAP: AxisMapType = {
  # as: field_key: (name, type, validate_fn, apply_fn)
  # where validate_fn: (value: AxisKindType) -> value: AxisKindType or raises Error if invalid
  # and apply_fn: (value: AxisKindType) -> value: AxisKindType to apply a value
  #
  # READY_TO_USE
  AxisField.Seed: (int, SeedValidate, TUNNEL),
  AxisField.CFG: (int, CFGValidate, TUNNEL),
  AxisField.Sampler: (str, cast('AxisFnType', SamplerValidate), TUNNEL),
  #
  # REPLACE_ME values here have to be added later (usually need some query)
  AxisField.Model: (str, REPLACE_ME, TUNNEL),  # replace needs DB
  AxisField.Positive: (str, TUNNEL, REPLACE_ME),  # replace needs the prompt
  AxisField.Negative: (str, TUNNEL, REPLACE_ME),  # replace needs the prompt
}


def KeyHash(key: ExperimentKeyType) -> str:
  """Compute a hash for the given experiment key (combination of axis values).

  Args:
    key: A list of axis values representing a specific combination of axes for an experiment.

  Returns:
    A string hash of the key, computed as the SHA256 hash of the canonical JSON serialization

  """
  return base.CanonicalHash(key)


class Experiments:
  """Experiments class to manage experiments configurations and results."""

  def __init__(self, ai_db: db.AIDatabase) -> None:
    """Initialize the Experiments with the given configuration, axes, and seeds.

    Args:
      ai_db: AIDatabase instance to use for DB access and queries.

    """
    # create basic experiment structure
    self._ai_db: db.AIDatabase = ai_db
    self._experiments: dict[str, ExperimentType] = ai_db.experiments  # type: ignore[assignment]  # mutable, tied to DB
    self._objects: dict[str, Experiment] = {}  # in-memory objects for experiments, keyed by hash
    for exp in self._experiments.values():
      self._objects[exp['hash']] = Experiment(  # exp is MUTABLE!
        ai_db,
        exp['config'],
        exp['axes'],
        [int(s) for s in exp['results']],  # JSON round-trip turns int keys → str; convert back
        exp['options'],
        loaded=exp,
      )

  def Make(
    self,
    config: db.AIMetaType,
    axes: list[AxisType],
    seeds: list[int],
    options: ExperimentOptionsType,
  ) -> Experiment:
    """Create a new experiment with the given configuration, axes, and seeds (or load existing).

    Args:
      config: AIMetaType object containing the base generation metadata (e.g., prompt, steps
          seed, width, height, sampler_id, model_key) to be varied along the axes.
      axes: List of AxisType objects defining the experiment axes to vary along and their values.
      seeds: List of seed values to run the experiment with.
      options: ExperimentOptionsType object containing options for the experiment

    Returns:
      The created Experiment object.

    """
    config = self._ai_db.QueryNormalize(config)  # normalize the config to ensure consistent hashing
    config['seed'] = -1  # seed is fixed to -1 to ensure consistent hashing
    exp = Experiment(self._ai_db, config, axes, seeds, options)
    if exp.experiment_hash in self._experiments:
      logging.error(f'Experiment with hash {exp.experiment_hash!r} already exists')
      return self._objects[exp.experiment_hash]  # return existing object
    self._experiments[exp.experiment_hash] = exp.experiment  # add to DB-tied dict
    self._objects[exp.experiment_hash] = exp  # add to in-memory objects
    return exp


class Experiment:
  """Experiment class to manage experiment configurations and results."""

  def __init__(  # noqa: C901, PLR0912
    self,
    ai_db: db.AIDatabase,
    config: db.AIMetaType,
    axes: list[AxisType],
    seeds: list[int],
    options: ExperimentOptionsType,
    *,
    loaded: ExperimentType | None = None,
  ) -> None:
    """Initialize the Experiment with the given configuration, axes, and seeds.

    Args:
      ai_db: AIDatabase instance to use for DB access and queries.
      config: AIMetaType object containing the base generation metadata (e.g., prompt, steps
          seed, width, height, sampler_id, model_key) to be varied along the axes.
      axes: List of AxisType objects defining the experiment axes to vary along and their values.
      seeds: List of seed values to run the experiment with.
      options: ExperimentOptionsType object containing options for the experiment
      loaded: (optional) MUTABLE ExperimentType object containing pre-existing experiment data

    Raises:
      Error: if any axis key is unsupported or if any axis value is invalid according to its

    """
    # create basic experiment structure
    self._ai_db: db.AIDatabase = ai_db
    self._config: db.AIMetaType = copy.deepcopy(config)
    self._axes: list[AxisType] = copy.deepcopy(axes)
    self._seeds: list[int] = sorted(set(seeds))  # unique and sorted seeds for consistent order
    self._options: ExperimentOptionsType = copy.deepcopy(options)
    if any(s for s in self._seeds if not (1 <= s <= base.SD_MAX_SEED)):
      raise Error(f'Invalid seed in seeds {self._seeds}, must be in [1, {base.SD_MAX_SEED}]')
    # make internal axis map and replace the REPLACE_ME with actual validation functions
    self._axis_map: AxisMapType = copy.deepcopy(_AXIS_MAP)
    self._ValidateModel: abc.Callable[[str], str] = lambda h: self._ai_db.GetModel(h)['hash']
    self._axis_map[AxisField.Model] = (str, cast('AxisFnType', self._ValidateModel), TUNNEL)
    if any(a['key'] == AxisField.Positive.value for a in self._axes):
      # only if we need them b/c they raise on absence of placeholder _PROMPT_PLACEHOLDER
      self._axis_map[AxisField.Positive] = (
        str,
        TUNNEL,
        cast('AxisFnType', PromptReplaceTunnel(self._config['positive'])),
      )
    if any(a['key'] == AxisField.Negative.value for a in self._axes):
      # only if we need them b/c they raise on absence of placeholder _PROMPT_PLACEHOLDER
      self._axis_map[AxisField.Negative] = (
        str,
        TUNNEL,
        cast('AxisFnType', PromptReplaceTunnel(self._config['negative'] or '')),
      )
    # validate axes and values
    # TODO: respect vae
    # TODO: respect pony
    # TODO: respect clip2
    validate_fn: AxisFnType
    model_index: int | None = None
    for n, axis in enumerate(self._axes):
      # validate axis key
      if (field := AxisField(axis['key'])) not in self._axis_map:
        raise Error(f'Unsupported axis key {field.value!r}')
      if field == AxisField.Model:
        if model_index is not None:
          raise Error('Multiple model axes are not supported')
        model_index = n
      # validate values
      _, validate_fn, _ = self._axis_map[AxisField(axis['key'])]
      axis['values'] = [validate_fn(v) for v in axis['values']]  # type: ignore[typeddict-item]
    # create experiment entries
    keys: set[tuple[int | str, ...]] = {
      tuple(combination)
      for combination in itertools.product(
        *[
          [self._axis_map[AxisField(axis['key'])][2](v) for v in axis['values']]
          for axis in self._axes
        ]
      )
    }
    # sort, save, then add the keys we created to all seeds
    self._keys: list[tuple[int | str, ...]] = sorted(
      # sort keys for consistent order; if there's a model axis, sort by that
      # first since it's usually the most important one to group by, otherwise
      # just sort by the whole key and rely on pythons tuple ordering;
      # NOTE: always use full tuple (k) as tiebreaker to avoid non-determinism from
      # set iteration order (Python hash randomization), which would make ties break randomly
      keys,
      key=lambda k: (k[model_index], k) if model_index is not None else k,
    )
    self._k_dict: dict[str, ExperimentKeyType] = {KeyHash(list(k)): list(k) for k in self._keys}
    self.experiment: ExperimentType
    self._results: ResultsType
    eh: str = self.experiment_hash
    if loaded:
      # we have loaded data, validate it matches the keys and seeds we have
      if loaded['config'] != self._config:
        raise Error(f'Loaded experiment config {loaded["config"]!r} mismatch {self._config!r}')
      if loaded['axes'] != self._axes:
        raise Error(f'Loaded experiment axes {loaded["axes"]!r} mismatch {self._axes!r}')
      if {int(s) for s in loaded['results']} != set(self._seeds):
        raise Error(f'Loaded experiment seeds {set(loaded["results"])} mismatch {set(self._seeds)}')
      if loaded['hash'] != eh:
        raise Error(f'Loaded experiment hash {loaded["hash"]!r} does not match expected {eh!r}')
      # all good, save them
      self.experiment = loaded  # keep it the same mutable object tied to DB
      self._results = loaded['results']  # keep it the same mutable object tied to DB
    else:
      # create empty results
      self._results = {
        str(_AXIS_MAP[AxisField.Seed][1](seed)): dict.fromkeys(self._k_dict) for seed in self._seeds
      }
      # create main object, some logging and we're done
      self.experiment = ExperimentType(
        config=self._config,
        axes=self._axes,
        options=self._options,
        results=self._results,
        hash=self.experiment_hash,
      )
    logging.info(f'Experiment seeds: {self._seeds} ; options {self._options}')
    logging.info(f'Experiment keys:\n{"\n".join(map(str, self._keys))}')
    total: int = len(self._keys) * len(self._results)
    logging.info(f'Experiment {self.experiment["hash"]!r} created with {total} combinations (sxk)')

  @property
  def experiment_hash(self) -> str:
    """Get the experiment hash, based on the config and keys (axes)."""
    return base.CanonicalHash(
      {'config': self._config, 'axes': self._axes, 'seeds': self._seeds, 'options': self._options}  # type: ignore[dict-item]
    )

  def Run(
    self, api: db.APIProtocol, *, redo: bool = False
  ) -> abc.Generator[tuple[db.DBImageType, bytes]]:
    """Generate image from text prompt, store in DB.

    Beware that with no self.output the image will NOT be added to the DB (no path to track).

    Args:
      api: APIProtocol instance to use for making the API call
      redo: (default False) If True, forces re-generation of the image even if it exists in the DB

    Yields:
      A tuple containing the DBImageType object and the raw image data.
      The returned object in NOT the same object in the DB, the DB has a copy.

    """
    # log, get the time
    total: int = len(self._keys) * len(self._results)
    tm_created: int = timer.Now()  # time is as soon as we start
    logging.info(f'Running experiment with {total} combinations (keys x seeds), @{tm_created}')
    n_done: int = 0
    n_skip: int = 0
    image_obj: db.DBImageType
    image_data: bytes
    # go over the combinations of axes (called keys)
    with timer.Timer(emit_log=False) as tmr_total:  # noqa: PLR1702
      for key in self._keys:  # keys is sorted already
        # create the meta for this combination by applying the axis apply_fn to the base config
        meta: db.AIMetaType = copy.deepcopy(self.experiment['config'])
        kh: str = KeyHash(list(key))
        for n, axis in enumerate(self.experiment['axes']):
          field = AxisField(axis['key'])
          tp: type[AxisKindType] = self._axis_map[field][0]
          # find the value for this axis in the key tuple and apply it to the meta
          meta[field.value] = tp(key[n])  # pyright: ignore[reportGeneralTypeIssues]
        # go over the seeds and generate the image for this combination
        for seed in self._seeds:  # seeds is sorted already
          # check for pre-existing result
          if (res := self._results[str(seed)][kh]) is not None and (
            img := self._ai_db.Image(res)
          ) is not None:
            existing_paths: list[str] = sorted(p for p in img['paths'] if pathlib.Path(p).exists())
            if existing_paths:
              logging.warning(f'Existing result for {seed}/{key}: {res!r} -> {existing_paths}')
              if not redo:
                n_skip += 1
                continue
              logging.warning('Redo is enabled, re-generating image anyway...')
          # update seed; generate the image and store in DB
          meta['seed'] = seed
          logging.info(f'Generating {seed}/{key}...')
          image_obj, image_data = self._ai_db.Txt2Img(meta, api, redo=redo, tm=tm_created)
          self._results[str(seed)][kh] = image_obj['hash']  # store the result hash in results
          n_done += 1
          logging.info(f'Progress: {n_done + n_skip}/{total} combinations completed, skip {n_skip}')
          # save after _SAVE_EVERY
          if n_done and not n_done % _SAVE_EVERY:
            self._ai_db.Save()
          # yield the images as we generate them
          yield (image_obj, image_data)
    # ended, log, generate final images
    logging.info(f'Experiment completed in {tmr_total}, had to do {n_done}, skip {n_skip}')
    # make grid images for each combination of axes and save them if we have an output
    for n, (image_obj, image_data) in enumerate(
      self.Grid(output_dir=self._ai_db.output, tm=tm_created)
    ):
      path_data: db.DBImagePathType = next(iter(image_obj['paths'].values()))
      path_data['ai_meta'] = copy.deepcopy(self._config)
      path_data['sd_info'] = cast('tbase.JSONDict', copy.deepcopy(self._axes))
      path_data['sd_params'] = {'seeds': list(self._seeds), 'grid_index': n}
      yield (image_obj, image_data)

  def _ActualCellSize(self) -> tuple[int, int]:
    """Get actual image cell size from the first available result in the DB.

    Reads width/height from DB metadata without opening any file.

    Returns:
      (cell_width, cell_height) in pixels, including grid padding.

    Raises:
      Error: If no completed result images are found.

    """
    for seed_results in self._results.values():
      for img_hash in seed_results.values():
        if img_hash is not None:
          img_meta: db.DBImageType | None = self._ai_db.Image(
            img_hash,
          )
          if img_meta is not None:
            return (
              img_meta['width'] + _GRID_PADDING,
              img_meta['height'] + _GRID_PADDING,
            )
    raise Error('No result images found — experiment not yet run?')

  def _AxisValueLabel(
    self,
    field: AxisField,
    value: int | str,
  ) -> str:
    """Create a human-readable label for a single axis value.

    Args:
      field: The AxisField this value belongs to.
      value: The raw axis value to label.

    Returns:
      A short displayable string for the given value.

    """
    if field == AxisField.Seed:
      return f'Seed {value}'
    if field == AxisField.Model:
      try:
        mdl: db.AIModelType = self._ai_db.GetModel(
          str(value),
        )
        return mdl['alias'] or mdl['name']
      except db.Error:
        return str(value)[:12]
    if field == AxisField.CFG:
      return f'CFG {int(value) / 10:.1f}'
    if field in {AxisField.Positive, AxisField.Negative}:
      s: str = str(value)
      return s[:_LABEL_TRUNCATE] + '...' if len(s) > _LABEL_TRUNCATE else s
    return str(value)

  def _DimValues(
    self,
    field: AxisField,
  ) -> abc.Sequence[int | str]:
    """Get the values for a given experiment dimension.

    Args:
      field: The AxisField dimension to get values for.

    Returns:
      The ordered values for the dimension (seeds or axis values).

    Raises:
      Error: If the field is not a dimension of this experiment.

    """
    if field == AxisField.Seed:
      return self._seeds
    for ax in self._axes:
      if AxisField(ax['key']) == field:
        return ax['values']
    raise Error(
      f'Dimension {field.value!r} not in experiment axes',
    )

  def _LookupResultHash(
    self,
    dim_indices: dict[AxisField, int],
  ) -> str:
    """Look up the result image hash for a dimension-index combo.

    Each entry in ``dim_indices`` maps an AxisField to the value
    index for that dimension. Must include Seed and every axis.

    Args:
      dim_indices: Mapping of AxisField -> value index for every
          dimension of the experiment.

    Returns:
      The image hash string stored in the experiment results.

    Raises:
      Error: If the result is missing for the given combination.

    """
    seed: int = self._seeds[dim_indices[AxisField.Seed]]
    key_parts: list[int | str] = []
    for ax in self._axes:
      field: AxisField = AxisField(ax['key'])
      applied: int | str = self._axis_map[field][2](
        ax['values'][dim_indices[field]],
      )
      key_parts.append(applied)
    kh: str = KeyHash(key_parts)
    result: str | None = self._results[str(seed)].get(kh)
    if result is None:
      raise Error(
        f'Missing result for seed={seed}, key={key_parts} — not fully run?',
      )
    return result

  def _BuildOneGrid(  # noqa: C901
    self,
    x_dim: AxisField,
    y_dim: AxisField | None,
    fixed_dims: dict[AxisField, int],
    img_w: int,
    img_h: int,
  ) -> Image.Image:
    """Build a single XY grid image for fixed higher dimensions.

    Creates a labeled image grid with X values as columns and Y
    values as rows. Fixed dimensions are shown in the corner labels.

    Args:
      x_dim: AxisField for the X axis (columns).
      y_dim: AxisField for the Y axis (rows), or None for 1D strip.
      fixed_dims: AxisField -> value index for higher dimensions.
      img_w: Cell width in pixels (image width + padding).
      img_h: Cell height in pixels (image height + padding).

    Returns:
      The assembled XY grid as a PIL Image.

    Raises:
      Error: on error

    """
    x_vals: abc.Sequence[int | str] = self._DimValues(x_dim)
    y_vals: abc.Sequence[int | str] = []
    if y_dim is not None:
      y_vals = self._DimValues(y_dim)
    n_cols: int = len(x_vals)
    n_rows: int = max(len(y_vals), 1)
    # corner label shows fixed dimension values
    corner_text: str = (
      ' / '.join(
        self._AxisValueLabel(f, self._DimValues(f)[i])
        for f, i in sorted(
          fixed_dims.items(),
          key=lambda kv: kv[0].value,
        )
      )
      or self.experiment_hash[:12]
    )
    corner: Image.Image = _MakeTagImage(
      corner_text,
      img_w,
      img_h,
      background_color='#222222',
    )
    grid: Image.Image = Image.new(
      'RGB',
      size=((n_cols + 2) * img_w, (n_rows + 2) * img_h),
      color='black',
    )
    # paste corner labels in all 4 corners
    for cx, cy in (
      (0, 0),
      (0, (n_rows + 1) * img_h),
      ((n_cols + 1) * img_w, 0),
      ((n_cols + 1) * img_w, (n_rows + 1) * img_h),
    ):
      grid.paste(corner, box=(cx, cy))
    # paste X labels along top and bottom
    for col, v in enumerate(x_vals):
      lbl: Image.Image = _MakeTagImage(
        self._AxisValueLabel(x_dim, v),
        img_w,
        img_h,
      )
      grid.paste(lbl, box=((col + 1) * img_w, 0))
      grid.paste(
        lbl,
        box=((col + 1) * img_w, (n_rows + 1) * img_h),
      )
    # paste Y labels along left and right
    if y_dim is not None:
      for row, v in enumerate(y_vals):
        lbl = _MakeTagImage(
          self._AxisValueLabel(y_dim, v),
          img_w,
          img_h,
        )
        grid.paste(lbl, box=(0, (row + 1) * img_h))
        grid.paste(
          lbl,
          box=((n_cols + 1) * img_w, (row + 1) * img_h),
        )
    # paste experiment images in the grid cells
    for row in range(n_rows):
      for col in range(n_cols):
        dims: dict[AxisField, int] = dict(fixed_dims)
        dims[x_dim] = col
        if y_dim is not None:
          dims[y_dim] = row
        # get hash, find the image
        img_hash: str = self._LookupResultHash(dims)
        img: db.DBImageType | None = self._ai_db.Image(img_hash)
        if img is None:
          raise Error(f'Image {img_hash!r} not found in DB')
        existing_paths: list[str] = sorted(p for p in img['paths'] if pathlib.Path(p).exists())
        if not existing_paths:
          raise Error(f'No file found on disk for image {img_hash!r}')
        # open the image and paste it into the grid
        with Image.open(existing_paths[0]) as raw_img:
          pad: int = _GRID_PADDING // 2
          grid.paste(
            raw_img,
            box=((col + 1) * img_w + pad, (row + 1) * img_h + pad),
          )
    return grid

  def Grid(  # noqa: PLR0914
    self,
    *,
    grid_axes_override: list[AxisField] | None = None,
    output_dir: pathlib.Path | None = None,
    tm: int | None = None,
  ) -> abc.Generator[tuple[db.DBImageType, bytes]]:
    """Create XY grid images from a completed experiment.

    Arranges experiment results into labeled grid images. The caller
    specifies which experiment dimensions map to X (columns), Y (rows),
    and higher dimensions (each combo produces a separate XY grid).

    All experiment dimensions (``Seed`` + every axis in ``self._axes``)
    must appear exactly once in ``grid_axes``:

    - ``grid_axes[0]`` = X axis (columns) — **required**
    - ``grid_axes[1]`` = Y axis (rows) — optional (omit for 1D strip)
    - ``grid_axes[2:]`` = higher dims; the cartesian product of their
      values produces one XY grid per combination

    Args:
      grid_axes_override: (optional) Ordered list of AxisField values assigning every
          experiment dimension to a grid role (X, Y, higher...). If None, the default
          grid axes will be used, seed last.
      output_dir: (optional) Directory to save grid PNGs into. Each
          grid is saved as ``grid_NNN.png`` (0-padded index).
      tm: (optional) Timestamp to use for output file naming; None, the current time will be used

    Yields:
      (db.DBImageType, image_bytes): grid images, one per higher-dimension combination.

    Raises:
      Error: If ``grid_axes`` doesn't cover all dimensions, contains
          duplicates, or a result is missing (experiment not run).

    """
    out: pathlib.Path | None = (
      pathlib.Path(output_dir).expanduser().resolve() if output_dir else None
    )
    tm_created: int = tm if tm and tm > 0 else timer.Now()
    # validate grid_axes covers all dimensions exactly once
    all_dims: list[AxisField] = [AxisField(a['key']) for a in self._axes] + [AxisField.Seed]
    if grid_axes_override:
      if len(grid_axes_override) != len(set(grid_axes_override)):
        raise Error(
          f'Duplicate dimensions in grid_axes: {grid_axes_override}',
        )
      if set(grid_axes_override) != set(all_dims):
        raise Error(
          f'grid_axes {set(grid_axes_override)} does not match experiment dims {all_dims}',
        )
      all_dims = grid_axes_override
    # decompose into X, Y, and higher dimensions
    x_dim: AxisField = all_dims[0]
    y_dim: AxisField | None = all_dims[1] if len(all_dims) > 1 else None
    higher: list[AxisField] = all_dims[2:]
    img_w, img_h = self._ActualCellSize()
    # compute higher-dimension value-index combinations
    higher_combos: list[tuple[int, ...]] = (
      list(
        itertools.product(
          *(range(len(self._DimValues(d))) for d in higher),
        ),
      )
      if higher
      else [()]
    )
    logging.info(
      f'Grid: {len(all_dims)} dims, {len(higher_combos)} grid(s) to produce',
    )
    # build one grid per higher-dimension combination
    for combo_idx, combo in enumerate(higher_combos):
      # make the image
      fixed: dict[AxisField, int] = dict(
        zip(higher, combo, strict=True),
      )
      grid: Image.Image = self._BuildOneGrid(x_dim, y_dim, fixed, img_w, img_h)
      # save to buffer, hash
      img_data = io.BytesIO()
      grid.save(img_data, format='PNG')
      img_bytes: bytes = img_data.getvalue()
      hsh: str = hashes.Hash256(img_bytes).hex()
      raw_hash: str = hashes.Hash256(grid.convert('RGBA').tobytes()).hex()
      # save to disk if we were given a destination
      full_path: pathlib.Path | None = None
      if out:
        date_str: str = time.strftime('%Y-%m-%d', time.gmtime(tm_created))
        tm_str: str = time.strftime('%Y%m%d%H%M%S', time.gmtime(tm_created))
        out_dir: pathlib.Path = out / date_str
        out_dir.mkdir(exist_ok=True)  # make sure the date dir exists, create it if not
        filename: str = f'{tm_str}-grid-{combo_idx:03d}-{hsh[:12]}.png'
        full_path = out_dir / filename
        full_path.write_bytes(img_bytes)
      # done, return this grid and then continue
      yield (
        db.DBImageType(
          hash=hsh,
          raw_hash=raw_hash,
          size=len(img_bytes),
          width=grid.width,
          height=grid.height,
          format=base.ImageFormat.PNG.value,
          info=None,
          paths={
            str(full_path) if full_path else '': db.DBImagePathType(
              main=False,
              created_at=tm_created,
              origin=db.ImageOrigin.TransNext.value,
              parse_errors=None,
              version=__version__,
              ai_meta=None,
              sd_info=None,
              sd_params=None,
            )
          },
        ),
        img_bytes,
      )


_GRID_PADDING: int = 12  # pixels of padding around each image cell in the grid
_FONT_CONVERGE_ITERATIONS: int = 10  # max iterations for font size convergence
_FONT_MIN_SINGLE_LINE: float = 100.0  # min font size for single line text
_LABEL_TRUNCATE: int = 40  # max chars for prompt labels before truncating


def _MakeTagImage(
  tag_text: str,
  img_width: int,
  img_height: int,
  *,
  background_color: str = 'black',
  text_color: str = 'white',
) -> Image.Image:
  """Make a label tile with text, auto-sizing the font to fit the tile.

  If the text is too long for one line, it will be split into 3 lines.

  Args:
    tag_text: Text to render on the tile.
    img_width: Tile width in pixels.
    img_height: Tile height in pixels.
    background_color: (default 'black') Tile background color.
    text_color: (default 'white') Tile text color.

  Returns:
    A PIL Image tile with the text rendered on it.

  """
  tag_text = tag_text.strip() or '<empty>'
  tag_img: Image.Image = Image.new(
    'RGB',
    size=(img_width, img_height),
    color=background_color,
  )
  draw: ImageDraw.ImageDraw = ImageDraw.Draw(tag_img)
  # search for a font size that will fit to 85% of available width
  target_width: int = round(0.85 * img_width)
  target_precision: int = max(round(0.02 * img_width), 1)
  sz: float = 50.0
  text_width: float = draw.textlength(tag_text, font_size=sz)
  iterations: int = 0
  while (
    abs(target_width - text_width) > target_precision and iterations < _FONT_CONVERGE_ITERATIONS
  ):
    sz *= target_width / text_width if text_width > 0 else 2.0
    text_width = draw.textlength(tag_text, font_size=sz)
    iterations += 1
  # draw text in image if font is a reasonable size (single line fits)
  if sz > _FONT_MIN_SINGLE_LINE:
    font_size: int = round(sz)
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.load_default(size=font_size)
    draw.text(
      ((img_width - text_width) / 2, (img_height - font_size) / 2),
      tag_text,
      fill=text_color,
      font=font,
    )
    return tag_img
  # font is small, so text is big, so we divide it into 3 lines
  part1: int = len(tag_text) // 3
  part2: int = 2 * len(tag_text) // 3
  lines: tuple[str, str, str] = (
    tag_text[:part1],
    tag_text[part1:part2],
    tag_text[part2:],
  )
  sz = 50.0
  text_width = max(draw.textlength(line, font_size=sz) for line in lines)
  iterations = 0
  while (
    abs(target_width - text_width) > target_precision and iterations < _FONT_CONVERGE_ITERATIONS
  ):
    sz *= target_width / text_width if text_width > 0 else 2.0
    text_width = max(draw.textlength(line, font_size=sz) for line in lines)
    iterations += 1
  font_size = round(sz)
  spacing: int = max(round(sz / 3), 1)
  font = ImageFont.load_default(size=font_size)
  draw.multiline_text(
    (
      (img_width - text_width) / 2,
      (img_height - (3 * font_size + 2 * spacing)) / 2,
    ),
    '\n'.join(lines),
    fill=text_color,
    font=font,
    spacing=spacing,
    align='center',
  )
  return tag_img
