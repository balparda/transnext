# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: transnext.core.base module."""

from __future__ import annotations

import importlib
import os
from unittest import mock

import pytest

from transnext.core import base


def testErrorIsSubclass() -> None:
  """Error is a subclass of base.Error from transcrypto."""
  assert issubclass(base.Error, Exception)


def testErrorRaise() -> None:
  """Error can be raised and caught."""  # noqa: DOC501
  with pytest.raises(base.Error, match='test error'):
    raise base.Error('test error')


def testDefaultPortNoEnvVar() -> None:
  """DEFAULT_PORT is 7860 when SDAPI_URL is not set."""
  assert base.DEFAULT_PORT == 7860


def testDefaultHostNoEnvVar() -> None:
  """DEFAULT_HOST is http://127.0.0.1 when SDAPI_URL is not set."""
  assert base.DEFAULT_HOST == 'http://127.0.0.1'


def testMakeURL() -> None:
  """MakeURL creates correct URL strings."""
  assert base.MakeURL('http://localhost', 8080) == 'http://localhost:8080'
  assert base.MakeURL('http://10.0.0.1', 7860) == 'http://10.0.0.1:7860'


def testSdapiUrlParsing() -> None:
  """Module correctly parses SDAPI_URL env var."""
  with mock.patch.dict(os.environ, {'SDAPI_URL': 'http://myhost:9999'}):
    importlib.reload(base)
  try:
    assert base.DEFAULT_PORT == 9999
    assert base.DEFAULT_HOST == 'http://myhost'
  finally:
    # restore to clean state
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop('SDAPI_URL', None)
      importlib.reload(base)


def testSdapiUrlInvalidPort() -> None:
  """Module handles invalid SDAPI_URL port by falling back to defaults."""
  with mock.patch.dict(os.environ, {'SDAPI_URL': 'http://host:notaport'}):
    importlib.reload(base)
  try:
    # fallback to defaults because ValueError was caught
    assert base.DEFAULT_PORT == 7860
    assert base.DEFAULT_HOST == 'http://127.0.0.1'
  finally:
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop('SDAPI_URL', None)
      importlib.reload(base)


def testDefaultConstants() -> None:
  """Default iteration/size/cfg constants have expected values."""
  assert base.SD_DEFAULT_ITERATIONS == 20
  assert base.SD_DEFAULT_WIDTH == 512
  assert base.SD_DEFAULT_HEIGHT == 512
  assert base.SD_DEFAULT_CFG_SCALE == 60
  assert base.SD_DEFAULT_CFG_END == 8
  assert base.SD_DEFAULT_CLIP_SKIP == 10
  assert base.SD_DEFAULT_QUERY_PARSER == base.QueryParser.A1111
  assert base.SD_DEFAULT_SAMPLER == base.Sampler.DPM_P_SDE


def testSamplerValues() -> None:
  """Sampler enum has correct string values."""
  assert base.Sampler.Euler.value == 'Euler'
  assert base.Sampler.EulerA.value == 'Euler a'
  assert base.Sampler.Unity.value == 'UniPC'
  assert base.Sampler.DPM_P_SDE.value == 'DPM++ SDE'
  assert base.Sampler.DPM_P_2M_SDE.value == 'DPM++ 2M SDE'


def testSamplerCount() -> None:
  """Sampler enum has 6 members."""
  assert len(base.Sampler) == 6


def testQueryParserValues() -> None:
  """QueryParser enum has correct string values."""
  assert base.QueryParser.SDNextNative.value == 'native'
  assert base.QueryParser.Compel.value == 'compel'
  assert base.QueryParser.XHinker.value == 'xhinker'
  assert base.QueryParser.A1111.value == 'a1111'
  assert base.QueryParser.Fixed.value == 'fixed'


def testQueryParserCount() -> None:
  """QueryParser enum has 5 members."""
  assert len(base.QueryParser) == 5


def testTransNextConfigCreation() -> None:
  """TransNextConfig can be created with all fields."""
  config = base.TransNextConfig(
    console=mock.MagicMock(),
    verbose=1,
    color=True,
    appconfig=mock.MagicMock(),
    host='http://localhost',
    port=7860,
    db=True,
    output=None,
  )
  assert config.host == 'http://localhost'
  assert config.port == 7860
  assert config.db is True
  assert config.output is None


def testTransNextConfigFrozen() -> None:
  """TransNextConfig is frozen (immutable)."""
  config = base.TransNextConfig(
    console=mock.MagicMock(),
    verbose=0,
    color=None,
    appconfig=mock.MagicMock(),
    host='http://localhost',
    port=7860,
    db=False,
    output=None,
  )
  with pytest.raises(AttributeError):
    config.host = 'http://other'  # type: ignore[misc]


def testPromptHashPositiveOnly() -> None:
  """PromptHash returns a 12-char hex string for positive-only prompt."""
  result = base.PromptHash('hello world')
  assert isinstance(result, str)
  assert len(result) == 12
  # should be deterministic
  assert result == base.PromptHash('hello world')


def testPromptHashWithNegative() -> None:
  """PromptHash includes negative prompt in hash."""
  result_pos: str = base.PromptHash('hello world')
  result_neg: str = base.PromptHash('hello world', 'bad stuff')
  assert result_pos != result_neg
  assert len(result_neg) == 12


def testPromptHashDeterministic() -> None:
  """PromptHash is deterministic for the same inputs."""
  assert base.PromptHash('a', 'b') == base.PromptHash('a', 'b')


def testPromptHashDifferentPositives() -> None:
  """Different positive prompts produce different hashes."""
  assert base.PromptHash('foo') != base.PromptHash('bar')


def testPromptHashNoneVsNoNegative() -> None:
  """None negative is the same as default (no negative)."""
  assert base.PromptHash('test', None) == base.PromptHash('test')
