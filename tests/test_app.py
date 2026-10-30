from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_ui_loads_and_rejects_empty_task():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    app.button[0].click().run()
    assert app.warning
    assert not app.exception
