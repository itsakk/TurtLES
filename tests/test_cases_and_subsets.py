from __future__ import annotations

import math

import pytest

from turtles.data.cases import CaseParams, format_case_name, parse_case_name
from turtles.data.normalization import ParamNormalizer
from turtles.data.subsets import parse_subset


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Re1000_alpha0_a0p5_n2", CaseParams(1000, 0, 0.5, 2)),
        ("Re1089_alpha12_a1p302_n2", CaseParams(1089, 12, 1.302, 2)),
        ("Re1300_alpha19_a0p287_ninf", CaseParams(1300, 19, 0.287, math.inf)),
        ("Re1586_alpha16_a0p2_n2", CaseParams(1586, 16, 0.2, 2)),
        ("Re2000_alpha10_a1p0_ninf", CaseParams(2000, 10, 1.0, math.inf)),
    ],
)
def test_parse_case_name(name, expected):
    params = parse_case_name(name)
    assert params == expected
    assert params.name == name


def test_format_roundtrip_matches_pipeline_convention():
    assert format_case_name(1000, 0, 0.5, 2) == "Re1000_alpha0_a0p5_n2"
    assert format_case_name(1266, 24, 0.893, 5) == "Re1266_alpha24_a0p893_n5"
    assert format_case_name(1843, 0, 0.206, math.inf) == "Re1843_alpha0_a0p206_ninf"


@pytest.mark.parametrize("bad", ["foo", "Re1000_alpha0", "Re1000_alpha0_a0p5_n2_extra"])
def test_parse_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_case_name(bad)


def test_param_normalizer_ranges():
    norm = ParamNormalizer(["Re", "alpha", "a", "n"])
    lo = norm(CaseParams(1000, 0, 1 / 6, 1))
    hi = norm(CaseParams(3900, 45, 1.5, math.inf))
    assert lo.tolist() == pytest.approx([-1, -1, -1, 1])
    assert hi.tolist() == pytest.approx([1, 1, 1, -1])
    raw = ParamNormalizer(["Re"], enabled=False)(CaseParams(2450, 0, 0.5, 2))
    assert raw.tolist() == [2450.0]
    assert len(ParamNormalizer([])) == 0


def test_param_normalizer_rejects_unknown():
    with pytest.raises(ValueError):
        ParamNormalizer(["Mach"])


def test_parse_subset():
    full = parse_subset("full")
    assert full.num_frames == 400 and full.num_chunks == 4 and len(full.families) == 4
    g2 = parse_subset("g2-t200")
    assert g2.name == "G2-T200"
    assert g2.families == (2.0, math.inf)
    assert g2.chunk_suffixes == ("b0_99", "b100_199")
    assert g2.accepts("Re1300_alpha19_a0p287_ninf")
    assert not g2.accepts("Re1190_alpha27_a0p595_n1")
    assert not g2.accepts("not_a_case")
    debug = parse_subset("debug")
    assert debug.debug and debug.split_dir_name("train") == "reference"
    assert debug.accepts("Re1000_alpha0_a0p5_n2") and not debug.accepts("Re1089_alpha12_a1p302_n2")
    with pytest.raises(ValueError):
        parse_subset("G5-T100")


def test_hf_patterns():
    patterns = parse_subset("G1-T100").hf_patterns(["train"])
    assert "data/stats.yaml" in patterns
    assert "data/train/*_n2/*_coords.h5" in patterns
    assert "data/train/*_n2/*_b0_99.h5" in patterns
    assert not any("b100_199" in p for p in patterns)
    assert not any("template.vtu" in p for p in patterns)
    debug = parse_subset("debug").hf_patterns(["train", "val"], include_vtu=True)
    assert "data/reference/Re1000_alpha0_a0p5_n2/*_template.vtu" in debug
