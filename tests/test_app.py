from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_initial_patient_screen_renders_without_exception():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()

    assert not app.exception
    assert app.header[0].value == "Review a retinal image"
    assert "not a medical diagnosis" in app.checkbox[0].label
