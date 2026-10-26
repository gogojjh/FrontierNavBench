"""Figure provenance must use portable keys when maps live outside the checkout."""

from pathlib import Path

from scripts import readme_figures


def test_relocated_map_has_logical_source_name(tmp_path, monkeypatch):
    monkeypatch.setattr(readme_figures, "data_root", lambda: tmp_path, raising=False)
    path = tmp_path / "maps/R05_dense/poses.txt"
    assert readme_figures.source_name(path) == "data/maps/R05_dense/poses.txt"


def test_repository_input_keeps_relative_source_name(tmp_path, monkeypatch):
    monkeypatch.setattr(readme_figures, "data_root", lambda: tmp_path, raising=False)
    path = readme_figures.REPO / "configs/maps.yaml"
    assert readme_figures.source_name(path) == "configs/maps.yaml"


def test_bundled_data_has_the_same_logical_name(monkeypatch):
    data = readme_figures.REPO / "data"
    monkeypatch.setattr(readme_figures, "data_root", lambda: data, raising=False)
    assert readme_figures.source_name(data / Path("maps/R05_dense/poses.txt")) == (
        "data/maps/R05_dense/poses.txt"
    )
