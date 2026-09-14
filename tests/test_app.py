from pathlib import Path
import ast

from streamlit.testing.v1 import AppTest


def test_initial_patient_screen_renders_without_exception():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()

    assert not app.exception
    assert app.header[0].value == "Review a medical image"
    assert "does not replace" in app.checkbox[0].label
    assert any("AI-assisted retinal assessment" in item.value for item in app.markdown)
    assert "Camera and live" in app.radio[0].options


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


def test_general_imaging_page_does_not_shadow_pandas_import():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    module = ast.parse(app_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "general_imaging_page"
    )

    local_pandas_imports = [
        alias
        for node in ast.walk(function)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "pandas"
    ]
    assert not local_pandas_imports
