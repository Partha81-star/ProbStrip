from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_initial_patient_screen_renders_without_exception():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()

    assert not app.exception
    assert app.header[0].value == "Review a medical image"
    assert "not a medical diagnosis" in app.checkbox[0].label
    assert "Use camera" in app.radio[0].options


def test_camera_screen_is_lazy_and_mobile_capture_is_available():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()
    navigation = next(radio for radio in app.radio if "Camera and live" in radio.options)

    navigation.set_value("Camera and live")
    app.run()

    assert not app.exception
    assert app.header[0].value == "Camera and live preview"
    assert len(app.get("camera_input")) == 1
    assert app.toggle[0].value is False


def test_review_screen_supports_general_imaging_categories():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()
    category = next(
        selectbox
        for selectbox in app.selectbox
        if "Bone or joint X-ray" in selectbox.options
    )

    category.set_value("Bone or joint X-ray")
    app.run()

    assert not app.exception
    assert app.header[0].value == "Review a medical image"
    assert len(app.get("file_uploader")) == 1
