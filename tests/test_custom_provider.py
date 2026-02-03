import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import os
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.config.schema import Config, ProvidersConfig, CustomConfig

@pytest.fixture
def mock_config():
    config = Config()
    config.providers.custom.api_key = "sk-custom"
    config.providers.custom.api_base = "http://localhost:11434"
    config.providers.custom.provider = "ollama"
    config.agents.defaults.model = "llama3"
    return config

def test_config_getters(mock_config):
    assert mock_config.get_api_key() == "sk-custom"
    assert mock_config.get_api_base() == "http://localhost:11434"
    assert mock_config.get_provider_format() == "ollama"

@pytest.mark.asyncio
async def test_litellm_custom_provider():
    # Mock litellm.acompletion
    with patch("nanobot.providers.litellm_provider.acompletion") as mock_acompletion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from Ollama"
        mock_response.choices[0].finish_reason = "stop"
        mock_acompletion.return_value = mock_response

        provider = LiteLLMProvider(
            api_key="sk-custom",
            api_base="http://localhost:11434",
            default_model="llama3",
            provider_format="ollama"
        )

        await provider.chat(messages=[{"role": "user", "content": "Hi"}])

        # Verify call args
        args, kwargs = mock_acompletion.call_args
        assert kwargs["model"] == "ollama/llama3"
        assert kwargs["api_base"] == "http://localhost:11434"

@pytest.mark.asyncio
async def test_litellm_openai_custom():
    # Test for generic openai compatible endpoint
    with patch("nanobot.providers.litellm_provider.acompletion") as mock_acompletion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from Custom OpenAI"
        mock_acompletion.return_value = mock_response

        provider = LiteLLMProvider(
            api_key="sk-custom-openai",
            api_base="https://my-api.com/v1",
            default_model="gpt-4-custom",
            provider_format="openai"
        )

        await provider.chat(messages=[{"role": "user", "content": "Hi"}])

        args, kwargs = mock_acompletion.call_args
        # Should be openai/gpt-4-custom or just gpt-4-custom depending on how LiteLLM handles it.
        # Our logic prepends provider_format.
        assert kwargs["model"] == "openai/gpt-4-custom"
        assert kwargs["api_base"] == "https://my-api.com/v1"

@pytest.mark.asyncio
async def test_litellm_legacy_vllm():
    # Test backward compatibility for vLLM without provider_format
    with patch("nanobot.providers.litellm_provider.acompletion") as mock_acompletion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from vLLM"
        mock_acompletion.return_value = mock_response

        provider = LiteLLMProvider(
            api_key="dummy",
            api_base="http://localhost:8000/v1",
            default_model="llama3",
            # provider_format is None
        )

        assert provider.is_vllm is True

        await provider.chat(messages=[{"role": "user", "content": "Hi"}])

        args, kwargs = mock_acompletion.call_args
        # Should default to hosted_vllm/ prefix
        assert kwargs["model"] == "hosted_vllm/llama3"
