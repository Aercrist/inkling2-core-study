"""Wording rules for the README and the docs."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PUBLIC = ["README.md", "CLAIMS.md"] + [f"docs/{p.name}" for p in sorted((ROOT / "docs").glob("*.md"))]


def test_wording_rules():
    """No 'orders of magnitude' claims and no 'negotiation'; the README states the study's independence."""
    for name in PUBLIC:
        text = (ROOT / name).read_text().lower()
        if name != "CLAIMS.md":  # CLAIMS.md names the phrase in order to rule it out
            assert "orders of magnitude" not in text, name
        assert "negotiation" not in text, name
    assert "not affiliated with" in (ROOT / "README.md").read_text()
