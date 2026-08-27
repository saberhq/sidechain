"""Contract tests for sidechain.utils.naming — the model-name grammar (ADR 0005).

The grammar guards three gates: submit.build refuses a malformed build stem before an
hours-long build, mirror2026.score / loco refuse a run directory that misspells a model
name, and freeform ablation labels (h1_xatlas) pass every gate untouched.
"""
import pytest

from sidechain.utils import naming


def test_valid_short_names_parse():
    assert naming.parse_short("SER-1") == ("SER", 1, "")
    assert naming.parse_short("SER-3n") == ("SER", 3, "n")
    assert naming.parse_short("PRO-12np") == ("PRO", 12, "np")


@pytest.mark.parametrize("bad", ["ser-1", "SER1", "SER-1N", "XXX-1", "SER-1pn", "SER-1nn", "SER-"])
def test_invalid_short_names_report(bad):
    assert naming.parse_short(bad) is None
    assert naming.problems_short(bad)


def test_near_miss_suggests_the_fix():
    (msg,) = naming.problems_short("ser2n")
    assert "SER-2n" in msg


@pytest.mark.parametrize("stem", ["ser-2n_delta4_even_noshrink_v1", "ser-3n_delta4full_even_noshrink_v1", "ser-1p", "gly-0_null_v2"])
def test_valid_stems(stem):
    assert naming.problems_stem(stem) == []


@pytest.mark.parametrize("bad", ["SER-2n_delta_v1", "ser2_delta_v1", "ser-2pn_delta_v1", "ser-2n_v1", "ser-2n_delta"])
def test_invalid_stems_report(bad):
    assert naming.problems_stem(bad)


def test_short_from_stem():
    assert naming.short_from_stem("ser-2n_delta4_even_noshrink_v1") == "SER-2n"
    assert naming.short_from_stem("ser-2n") == "SER-2n"
    assert naming.short_from_stem("h1_xatlas") is None


@pytest.mark.parametrize("freeform", ["h1_xatlas", "baseline", "feng_only", "loco_jurkat_rule", "serine_test", "hepg2_flowtest"])
def test_freeform_labels_pass_every_gate(freeform):
    assert not naming.CLAIMS_RE.match(freeform)
    naming.check_out_leaf(freeform, context="t")  # must not raise


@pytest.mark.parametrize("bad", ["ser2_x", "SER-2N", "Ser-2_delta_v1", "ser-2pn_delta_v1"])
def test_claiming_but_malformed_leaves_die(bad):
    with pytest.raises(SystemExit):
        naming.check_out_leaf(bad, context="t")


def test_run_dir_may_be_bare_but_build_stem_may_not():
    naming.check_out_leaf("ser-2n", context="t")  # a mirror run directory: fine
    with pytest.raises(SystemExit):
        naming.check_out_leaf("ser-2n", context="t", require_slug=True)
    naming.check_out_leaf("ser-2n_delta4_even_noshrink_v1", context="t", require_slug=True)
