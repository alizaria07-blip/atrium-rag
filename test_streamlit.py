"""Tests for the Streamlit app and Cloud deployment helpers."""
from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_error():
    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=30)
    # The script should render without raising.
    assert not at.exception, at.exception
    # A title should be present.
    assert any(t.value == "📚 Local RAG" for t in at.title)
    # There must be a chat input so the user can ask questions.
    assert len(at.chat_input) >= 1


def test_streamlit_app_warns_when_no_api_key():
    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=30)
    warnings = [w.value for w in at.warning]
    assert any("No API key set" in w for w in warnings)


def test_streamlit_app_loads_sample_database():
    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=30)
    assert not at.exception
    # Find sample database button and click it
    btn = at.button(key="load_sample_db_btn")
    btn.click().run()
    assert not at.exception
    # Selected db id should now be set in session_state
    assert at.session_state.selected_db_id is not None