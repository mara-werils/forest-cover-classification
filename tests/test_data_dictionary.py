"""The generated data dictionary covers every column and is in sync with the repo."""

from src import config, data_dictionary, data_loader


def test_every_column_and_code_is_documented(sample):
    text = data_dictionary.build_markdown(sample)
    for col in config.NUMERIC_COLS + config.WILDERNESS_COLS:
        assert f"`{col}`" in text
    for soil_id, (elu, _) in config.SOIL_TYPES.items():
        assert f"| {soil_id} | {elu} |" in text
    for name in config.COVER_TYPES.values():
        assert f"| {name} |" in text


def test_committed_dictionary_is_up_to_date(tmp_path):
    """Fails when config or the Parquet file change without regenerating the doc."""
    fresh = data_dictionary.write_dictionary(data_loader.load_parquet(), tmp_path / "dd.md")
    assert fresh.read_text() == data_dictionary.OUTPUT.read_text(), (
        "Run `python -m src.data_dictionary` and commit docs/DATA_DICTIONARY.md")
