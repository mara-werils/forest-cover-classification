"""PSI / TVD on distributions with a known answer, and the area report."""

import json

import numpy as np
import pytest

from src import config, shift


def test_psi_is_zero_for_identical_samples():
    values = np.random.default_rng(0).normal(size=5_000)
    assert shift.psi(values, values) == pytest.approx(0, abs=1e-12)


def test_psi_grows_with_the_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=20_000)
    small = shift.psi(reference, rng.normal(0.1, 1, 20_000))
    large = shift.psi(reference, rng.normal(1.0, 1, 20_000))
    assert small < 0.10 < 0.25 < large


def test_psi_handles_integer_data_with_ties():
    reference = np.repeat([0, 1, 2], [900, 50, 50])     # most quantiles collapse to 0
    assert np.isfinite(shift.psi(reference, reference[::-1]))
    assert shift.psi(np.zeros(10), np.zeros(10)) == 0.0


def test_categorical_psi_and_tvd():
    a = np.array([1] * 50 + [2] * 50)
    assert shift.categorical_psi(a, a) == pytest.approx(0)
    assert shift.total_variation(a, a) == pytest.approx(0)
    assert shift.total_variation(a, np.array([3] * 100)) == pytest.approx(1)
    assert shift.total_variation(a, np.array([1] * 100)) == pytest.approx(0.5)
    # A category that exists on one side only must not produce inf/nan.
    assert np.isfinite(shift.categorical_psi(a, np.array([1] * 60 + [9] * 40)))


@pytest.mark.parametrize("value, label", [(0.05, "stable"), (0.1, "moderate"),
                                          (0.24, "moderate"), (0.25, "major")])
def test_severity_thresholds(value, label):
    assert shift.severity(value) == label


def test_area_table_covers_every_area_and_feature(sample):
    table = shift.area_shift_table(sample)
    present = sample[config.WILDERNESS].nunique()
    assert len(table) == present * (len(config.NUMERIC_COLS) + 2)
    assert (table["psi"] >= 0).all()
    numeric = table[table["kind"] == "numeric"]
    assert numeric["ks"].between(0, 1).all()
    assert table.loc[table["kind"] == "categorical", "tvd"].between(0, 1).all()


def test_summary_flags_classes_found_in_one_area_only(make_frame):
    df = make_frame(n=280)
    # Move every Cottonwood/Willow cell into area 4, like in the real data.
    cottonwood = df[config.TARGET] == 4
    df.loc[cottonwood, config.WILDERNESS] = 4
    table = shift.area_shift_table(df)
    summary = shift.area_summary(df, table).set_index("area")
    assert summary.loc["Cache la Poudre", "classes_only_in_this_area"] == ["Cottonwood/Willow"]
    assert "Cottonwood/Willow" in summary.loc["Rawah", "classes_missing_here"]


def test_run_shift_writes_outputs(sample, tmp_reports):
    table, summary = shift.run_shift(sample)
    stored = json.loads((tmp_reports / "area_shift.json").read_text())
    assert [a["area"] for a in stored["areas"]] == summary["area"].tolist()
    assert "heldout_f1_macro" not in summary      # no group validation file in tmp
    assert (tmp_reports / "area_shift.csv").exists()
    assert (config.FIGURES_DIR / "20_area_shift.png").exists()


def test_group_validation_is_attached_when_available(sample, tmp_reports):
    records = [{"held_out_area": name, "model": "lightgbm", "f1_macro": 0.5}
               for name in config.WILDERNESS_AREAS.values()]
    (tmp_reports / "group_validation.json").write_text(json.dumps(records))
    _, summary = shift.run_shift(sample, save=False)
    assert (summary["heldout_f1_macro"] == 0.5).all()
