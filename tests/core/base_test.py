# SPDX-FileCopyrightText: Copyright 2026 Daniel Balparda <balparda@github.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for: transnext.core.base module."""

from __future__ import annotations

import importlib
import io
import os
from unittest import mock

import pytest
from PIL import Image, PngImagePlugin

from transnext.core import base

# ─── Error ────────────────────────────────────────────────────────────────────


def testErrorIsSubclass() -> None:
  """Error is a subclass of Exception."""
  assert issubclass(base.Error, Exception)


# ─── Module-level defaults (SDAPI_URL parsing) ───────────────────────────────


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


# ─── Sampler ─────────────────────────────────────────────────────────────────


def testSamplerA1111SubsetOfSampler() -> None:
  """All SamplerA1111 values also exist as Sampler values."""
  sampler_values: set[str] = {s.value for s in base.Sampler}
  for a1111 in base.SamplerA1111:
    assert a1111.value in sampler_values, f'{a1111.value!r} not found in Sampler'


# ─── TransNextConfig ─────────────────────────────────────────────────────────


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


# ─── PromptHash ──────────────────────────────────────────────────────────────


def testPromptHashPositiveOnly() -> None:
  """PromptHash returns a 12-char hex string for positive-only prompt."""
  result: str = base.PromptHash('hello world')
  assert isinstance(result, str)
  assert len(result) == 12


def testPromptHashDeterministic() -> None:
  """PromptHash is deterministic for the same inputs."""
  assert base.PromptHash('hello world') == base.PromptHash('hello world')
  assert base.PromptHash('a', 'b') == base.PromptHash('a', 'b')


def testPromptHashWithNegative() -> None:
  """PromptHash includes negative prompt in hash."""
  result_pos: str = base.PromptHash('hello world')
  result_neg: str = base.PromptHash('hello world', 'bad stuff')
  assert result_pos != result_neg
  assert len(result_neg) == 12


def testPromptHashDifferentPositives() -> None:
  """Different positive prompts produce different hashes."""
  assert base.PromptHash('foo') != base.PromptHash('bar')


def testPromptHashNoneVsNoNegative() -> None:
  """None negative is the same as default (no negative)."""
  assert base.PromptHash('test', None) == base.PromptHash('test')


# ─── GetBasicDataFromImage ───────────────────────────────────────────────────


def testGetBasicDataFromImagePNG() -> None:
  """GetBasicDataFromImage returns correct format, size, hash for PNG."""
  img: Image.Image = Image.new('RGBA', (32, 48), color=(255, 0, 0, 255))
  buf = io.BytesIO()
  img.save(buf, format='PNG')
  img_bytes: bytes = buf.getvalue()
  fmt, width, height, raw_hash, info_text = base.GetBasicDataFromImage(img_bytes)
  assert fmt == base.ImageFormat.PNG
  assert width == 32
  assert height == 48
  assert isinstance(raw_hash, str)
  assert len(raw_hash) == 64  # SHA256 hex
  assert info_text is None  # no metadata embedded


def testGetBasicDataFromImageJPEG() -> None:
  """GetBasicDataFromImage handles JPEG images."""
  img: Image.Image = Image.new('RGB', (64, 64), color='blue')
  buf = io.BytesIO()
  img.save(buf, format='JPEG')
  img_bytes: bytes = buf.getvalue()
  fmt, width, height, _, _ = base.GetBasicDataFromImage(img_bytes)
  assert fmt == base.ImageFormat.JPEG
  assert width == 64
  assert height == 64


def testGetBasicDataFromImageWithMetadata() -> None:
  """GetBasicDataFromImage extracts 'parameters' metadata from PNG."""
  img: Image.Image = Image.new('RGB', (16, 16))
  png_info = PngImagePlugin.PngInfo()
  png_info.add_text('parameters', 'test metadata here')
  buf = io.BytesIO()
  img.save(buf, format='PNG', pnginfo=png_info)
  img_bytes: bytes = buf.getvalue()
  _, _, _, _, info_text = base.GetBasicDataFromImage(img_bytes)
  assert info_text == 'test metadata here'


def testGetBasicDataFromImageUserComment() -> None:
  """GetBasicDataFromImage extracts 'UserComment' metadata from PNG."""
  img: Image.Image = Image.new('RGB', (16, 16))
  png_info = PngImagePlugin.PngInfo()
  png_info.add_text('UserComment', 'user comment data')
  buf = io.BytesIO()
  img.save(buf, format='PNG', pnginfo=png_info)
  img_bytes: bytes = buf.getvalue()
  _, _, _, _, info_text = base.GetBasicDataFromImage(img_bytes)
  assert info_text == 'user comment data'


def testGetBasicDataFromImageUnsupportedFormat() -> None:
  """GetBasicDataFromImage raises Error for unsupported format."""
  img: Image.Image = Image.new('RGB', (8, 8))
  buf = io.BytesIO()
  img.save(buf, format='BMP')
  img_bytes: bytes = buf.getvalue()
  with pytest.raises(base.Error, match='Unsupported image format'):
    base.GetBasicDataFromImage(img_bytes)


# ─── LoraExtract ─────────────────────────────────────────────────────────────


def testLoraExtractSingleLora() -> None:
  """LoraExtract extracts a single lora reference."""
  query: str = 'test <lora:my-model:0.8> prompt'
  result: dict[str, tuple[str, str]] = base.LoraExtract(query)
  assert 'my-model' in result
  assert result['my-model'] == ('lora', '0.8')


def testLoraExtractMultiple() -> None:
  """LoraExtract extracts multiple lora/lyco references."""
  query: str = '<lora:model-a:1.0> <lyco:model-b:0.5>'
  result: dict[str, tuple[str, str]] = base.LoraExtract(query)
  assert result == {
    'model-a': ('lora', '1.0'),
    'model-b': ('lyco', '0.5'),
  }


def testLoraExtractNoMatch() -> None:
  """LoraExtract returns empty dict when no lora references."""
  result: dict[str, tuple[str, str]] = base.LoraExtract('just a prompt')
  assert result == {}


def testLoraExtractLycoris() -> None:
  """LoraExtract handles lycoris and lycora kinds."""
  query: str = '<lycoris:deep-model:1.2>'
  result: dict[str, tuple[str, str]] = base.LoraExtract(query)
  assert 'deep-model' in result
  assert result['deep-model'][0] == 'lycoris'


# ─── FindModelHash ───────────────────────────────────────────────────────────


def testFindModelHashExactMatch() -> None:
  """Exact hash match returns the hash unchanged."""
  models: dict[str, str] = {'abc123-full-hash': 'mymodel mymodel'}
  assert base.FindModelHash('model', 'abc123-full-hash', '', models) == 'abc123-full-hash'


def testFindModelHashPrefixMatch() -> None:
  """Prefix hash match returns the full model hash."""
  models: dict[str, str] = {'abc123-full-hash': 'mymodel mymodel'}
  assert base.FindModelHash('model', 'abc123', '', models) == 'abc123-full-hash'


def testFindModelHashNameMatch() -> None:
  """Name match returns the model hash."""
  models: dict[str, str] = {'abc123': 'my-model my-alias'}
  assert base.FindModelHash('model', '', 'my-model', models) == 'abc123'


def testFindModelHashNoMatchRaises() -> None:
  """No match raises Error."""
  models: dict[str, str] = {'abc123': 'my-model my-alias'}
  with pytest.raises(base.Error, match='not found'):
    base.FindModelHash('model', 'deadbeef', 'nonexistent', models)


def testFindModelHashAmbiguousRaises() -> None:
  """Ambiguous prefix match raises Error."""
  models: dict[str, str] = {
    'abc123aaa': 'model-a alias-a',
    'abc123bbb': 'model-b alias-b',
  }
  with pytest.raises(base.Error, match='ambiguous'):
    base.FindModelHash('model', 'abc123', '', models)


def testFindModelHashEmptyQueryRaises() -> None:
  """Empty hash and name raises Error."""
  with pytest.raises(base.Error, match='empty query'):
    base.FindModelHash('model', '', '', {})


def testFindModelHashInvalidTypeRaises() -> None:
  """Invalid type parameter raises Error."""
  with pytest.raises(base.Error, match='invalid'):
    base.FindModelHash('invalid', 'abc', '', {})


def testFindModelHashLoraType() -> None:
  """FindModelHash works with lora type."""
  models: dict[str, str] = {'lora-hash-123': 'my-lora my-lora-alias'}
  assert base.FindModelHash('lora', 'lora-hash-123', '', models) == 'lora-hash-123'


def testFindModelHashAmbiguousNameRaises() -> None:
  """Ambiguous name match raises Error."""
  models: dict[str, str] = {'aaa': 'shared-name x', 'bbb': 'shared-name y'}
  with pytest.raises(base.Error, match='ambiguous'):
    base.FindModelHash('model', '', 'shared-name', models)


# ─── SeedGen ─────────────────────────────────────────────────────────────────


def testSeedGenRange() -> None:
  """SeedGen returns a value in the valid seed range."""
  seed: int = base.SeedGen()
  assert 2**16 - 1 <= seed <= base.SD_MAX_SEED
