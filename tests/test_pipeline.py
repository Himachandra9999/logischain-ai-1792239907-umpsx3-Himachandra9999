
import pandas as pd

from src.data.pipeline import process_dataset


def test_process_dataset_cleans_and_saves_csv(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    raw_file = raw_dir / "sample.csv"

    pd.DataFrame({
        "amount": [100.0, 100.0, float("inf"), 250.0],
        "default_risk": [0, 0, 1, 0],
    }).to_csv(raw_file, index=False)

    def fake_load_csv(filename):
        return pd.read_csv(raw_file)

    def fake_save_processed_csv(df, filename):
        output_path = processed_dir / filename
        df.to_csv(output_path, index=False)
        return output_path

    monkeypatch.setattr(
        "src.data.pipeline.load_csv", fake_load_csv
    )
    monkeypatch.setattr(
        "src.data.pipeline.save_processed_csv",
        fake_save_processed_csv,
    )

    result = process_dataset("sample.csv")

    assert result["original_rows"] == 4
    assert result["processed_rows"] == 3
    assert result["validation_report"]["infinite_values"] == {}
    assert result["validation_report"]["missing_values"]["amount"] == 1
    assert (processed_dir / "sample_processed.csv").exists()
