"""Environment tests: UTF-8 handling of Bangla text, the BanglaBERT normalizer, run metadata."""
from pathlib import Path

from normalizer import normalize

from bangla_sentiment.utils import ensure_utf8, git_commit

BANGLA = "ভাই আপনার ক্যামেরা মেনকে দিলেয়া একাই সব সাবার করলেন, হা হা হা"


def test_python_is_in_utf8_mode():
    ensure_utf8()


def test_bangla_round_trip(tmp_path):
    path = tmp_path / "bn.txt"
    with open(path, "w") as f:  # default encoding on purpose: this is what breaks on Windows
        f.write(BANGLA)
    assert path.read_bytes() == BANGLA.encode("utf-8")
    with open(path) as f:
        assert f.read() == BANGLA


def test_git_commit_ignores_results_folder():
    before = git_commit()
    probe = Path(__file__).resolve().parents[1] / "results" / "_probe_test.json"
    probe.write_text("{}", encoding="utf-8")
    try:
        assert git_commit() == before
    finally:
        probe.unlink()


def test_normalizer_collapses_whitespace_and_is_stable():
    out = normalize("আমি   বইটা  পড়ে খুব খুশি!!!")
    assert "  " not in out
    assert out.startswith("আমি বইটা")
    assert normalize(out) == out
