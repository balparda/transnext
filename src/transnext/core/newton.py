# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""TransNext core experiments module."""

from __future__ import annotations

import copy
import enum
import itertools
import logging
import pathlib
from collections import abc
from typing import Protocol, TypedDict, cast, overload

from transcrypto.utils import timer

from transnext.core import base, db


class Error(base.Error):
  """TransNext experiments exception."""


type ExperimentKeyType = list[str | int]  # experiment instance [ax1_value, ax2_value, ...]
type AxisKindType = str | int  # axis value type
type ExperimentKeysType = dict[str, str | None]  # hash is the key b/c JSON can't have tuple keys
type ResultsType = dict[int, ExperimentKeysType]
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
  results: ResultsType  # {seed: {experiment_key: result_hash}}


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
        ai_db, exp['config'], exp['axes'], list(exp['results']), loaded=exp
      )

  def New(self, config: db.AIMetaType, axes: list[AxisType], seeds: list[int]) -> Experiment:
    """Create a new experiment with the given configuration, axes, and seeds.

    Args:
      config: AIMetaType object containing the base generation metadata (e.g., prompt, steps
          seed, width, height, sampler_id, model_key) to be varied along the axes.
      axes: List of AxisType objects defining the experiment axes to vary along and their values.
      seeds: List of seed values to run the experiment with.

    Returns:
      The created Experiment object.

    Raises:
      Error: if an experiment with the same hash already exists

    """
    config = self._ai_db.QueryNormalize(config)  # normalize the config to ensure consistent hashing
    config['seed'] = -1  # seed is fixed to -1 to ensure consistent hashing
    exp = Experiment(self._ai_db, config, axes, seeds)
    if exp.experiment_hash in self._experiments:
      raise Error(f'Experiment with hash {exp.experiment_hash!r} already exists')
    self._experiments[exp.experiment_hash] = exp.experiment  # add to DB-tied dict
    self._objects[exp.experiment_hash] = exp  # add to in-memory objects
    self._ai_db.Save()  # save to DB immediately
    return exp


class Experiment:
  """Experiment class to manage experiment configurations and results."""

  def __init__(  # noqa: C901
    self,
    ai_db: db.AIDatabase,
    config: db.AIMetaType,
    axes: list[AxisType],
    seeds: list[int],
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
      loaded: (optional) MUTABLE ExperimentType object containing pre-existing experiment data

    Raises:
      Error: if any axis key is unsupported or if any axis value is invalid according to its

    """
    # create basic experiment structure
    self._ai_db: db.AIDatabase = ai_db
    self._config: db.AIMetaType = copy.deepcopy(config)
    self._axes: list[AxisType] = copy.deepcopy(axes)
    self._seeds: list[int] = sorted(set(seeds))  # unique and sorted seeds for consistent order
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
      # just sort by the whole key and rely on pythons tuple ordering
      keys,
      key=lambda k: k[model_index] if model_index is not None else k,
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
      if set(loaded['results']) != set(self._seeds):
        raise Error(f'Loaded experiment seeds {set(loaded["results"])} mismatch {set(self._seeds)}')
      if loaded['hash'] != eh:
        raise Error(f'Loaded experiment hash {loaded["hash"]!r} does not match expected {eh!r}')
      # all good, save them
      self.experiment = loaded  # keep it the same mutable object tied to DB
      self._results = loaded['results']  # keep it the same mutable object tied to DB
    else:
      # create empty results
      self._results = {
        _AXIS_MAP[AxisField.Seed][1](seed): dict.fromkeys(self._k_dict) for seed in self._seeds
      }
      # create main object, some logging and we're done
      self.experiment = ExperimentType(
        config=self._config,
        axes=self._axes,
        results=self._results,
        hash=self.experiment_hash,
      )
    logging.info(f'Experiment seeds: {sorted(self._results)}')
    logging.info(f'Experiment keys:\n{"\n".join(map(str, self._keys))}')
    total: int = len(self._keys) * len(self._results)
    logging.info(f'Experiment {self.experiment["hash"]!r} created with {total} combinations (sxk)')

  @property
  def experiment_hash(self) -> str:
    """Get the experiment hash, based on the config and keys (axes)."""
    return base.CanonicalHash(
      {'config': self._config, 'axes': self._axes, 'seeds': self._seeds}  # type: ignore[dict-item]
    )

  def Run(self, api: db.APIProtocol, *, redo: bool = False) -> tuple[db.DBImageType, bytes]:
    """Generate image from text prompt, store in DB.

    Beware that with no self.output the image will NOT be added to the DB (no path to track).

    Args:
      api: APIProtocol instance to use for making the API call
      redo: (default False) If True, forces re-generation of the image even if it exists in the DB

    Returns:
      A tuple containing the DBImageType object and the raw image data.
      The returned object in NOT the same object in the DB, the DB has a copy.

    """
    # log, get the time
    total: int = len(self._keys) * len(self._results)
    tm_created: int = timer.Now()  # time is as soon as we start
    logging.info(f'Running experiment with {total} combinations (keys x seeds), @{tm_created}')
    n_done: int = 0
    n_skip: int = 0
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
        for seed in sorted(self._results):  # sort seeds for consistent order
          # check for pre-existing result
          if (res := self._results[seed][kh]) is not None and (
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
          image: db.DBImageType = self._ai_db.Txt2Img(meta, api, redo=redo, tm=tm_created)[0]
          self._results[seed][kh] = image['hash']  # store the result hash in experiment results
          n_done += 1
          logging.info(f'Progress: {n_done + n_skip}/{total} combinations completed, skip {n_skip}')
        # save after each combination, so we have partial results
        self._ai_db.Save()
    # ended, log, generate final images
    logging.info(f'Experiment completed in {tmr_total}, had to do {n_done}, skip {n_skip}')
    return 1  # TODO: implement returning the final image objects and data, currently a placeholder
