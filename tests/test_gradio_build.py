import pytest

pytest.importorskip("gradio")

from fall_detection.app.gradio_app import build_demo


def test_build_demo_contains_named_state_groups():
    demo = build_demo()
    config = demo.get_config_file()
    ids = {
        component.get("props", {}).get("elem_id") for component in config["components"]
    }

    assert {"fd-input", "fd-processing", "fd-result", "fd-error"} <= ids


def test_build_demo_does_not_render_a_blank_dataframe_output():
    demo = build_demo()
    config = demo.get_config_file()
    component_types = {component["type"] for component in config["components"]}

    assert "dataframe" not in component_types
