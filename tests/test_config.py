from multi_agent_research_lab.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.effective_model
    assert settings.max_iterations >= 1


def test_settings_groq_resolution() -> None:
    settings = Settings(
        llm_provider="groq",
        groq_api_key="gsk_test123",
        groq_model="llama-3.3-70b-versatile",
    )
    assert settings.is_groq is True
    assert settings.effective_api_key == "gsk_test123"
    assert settings.effective_model == "llama-3.3-70b-versatile"
    assert settings.effective_base_url == "https://api.groq.com/openai/v1"

