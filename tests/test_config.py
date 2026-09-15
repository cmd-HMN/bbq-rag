import os
import pytest
from bbq.src.config import Config
from bbq.src.client import GeminiClient
from bbq.src.utils.futils import get_system_cache_dir
from bbq.src.common.errors import ConfigFNFWarning, ConfigParseError


def test_system_cache_dir():
    """Verify system cache path computation and subfolder appending."""
    base_dir = get_system_cache_dir()
    assert "bbq" in base_dir
    assert not base_dir.startswith("./.cache")

    sub_dir = get_system_cache_dir("test_folder")
    assert sub_dir == os.path.join(base_dir, "test_folder")


def test_config_defaults():
    """Verify Config defaults and dynamic post-init paths."""
    config = Config()
    assert config.base_model_id == "HuggingFaceTB/SmolVLM-256M-Instruct"
    assert config.lora_adapter_id == "vidore/colSmol-256M"
    assert config.embedding_dim == 128
    assert config.device == "auto"
    assert config.torch_dtype == "bfloat16"
    assert config.mask_non_image_embeddings is False
    assert config.pdf_render_dpi == 150
    assert config.embeddings_output_path == get_system_cache_dir("embeddings")
    assert config.sqlite_db_path == get_system_cache_dir("tracker.db")
    assert config.gemini_model == "gemini-3.6-flash"
    assert config.rag_top_k == 3
    assert config.quantization == "f32"


def test_config_custom_initialization():
    """Verify custom values and type casting in Config."""
    config = Config(
        base_model_id="custom/model",
        embedding_dim=256,
        pdf_render_dpi=200,
        rag_top_k=5,
        gemini_api_key="TEST_API_KEY_XYZ",
        gemini_model="gemini-2.0-flash",
    )
    assert config.base_model_id == "custom/model"
    assert config.embedding_dim == 256
    assert config.pdf_render_dpi == 200
    assert config.rag_top_k == 5
    assert config.gemini_api_key == "TEST_API_KEY_XYZ"
    assert config.gemini_model == "gemini-2.0-flash"


def test_config_empty_base_model_raises():
    """Verify that an empty base_model_id raises a ValueError."""
    with pytest.raises(ValueError, match="base_model_id cannot be empty"):
        Config(base_model_id="")


def test_config_visual_prompt_prefix():
    """Verify visual prompt formatting."""
    config = Config(visual_prompt_command="Describe this document.")
    prefix = config.format_visual_prompt_prefix()
    assert (
        prefix
        == "<|im_start|>User:<image>Describe this document.<end_of_utterance>\nAssistant:"
    )

    # If already formatted, keep as is
    already_formatted = "<|im_start|>custom<end_of_utterance>"
    config_custom = Config(visual_prompt_command=already_formatted)
    assert config_custom.format_visual_prompt_prefix() == already_formatted


def test_config_dictionary_format():
    """Verify dictionary serialization masks API key and includes helper keys."""
    config = Config(
        gemini_api_key="SECRET_KEY_123",
        gemini_model="gemini-2.0-flash",
        rag_top_k=5,
    )
    d = config.convert_to_dictionary_format()
    assert "gemini_api_key" not in d
    assert d["gemini_api_key_configured"] is True
    assert d["gemini_model"] == "gemini-2.0-flash"
    assert d["rag_top_k"] == 5
    assert "visual_prompt_prefix" in d


def test_gemini_client_from_config():
    """Verify GeminiClient.from_config factory method."""
    config = Config(
        gemini_api_key="TEST_KEY_FOR_CLIENT",
        gemini_model="gemini-2.0-flash",
    )
    client = GeminiClient.from_config(config)
    assert client.api_key == "TEST_KEY_FOR_CLIENT"
    assert client.model == "gemini-2.0-flash"
    assert client.is_available() is True


def test_config_from_yaml(tmp_path):
    """Verify loading configuration from a YAML file."""
    yaml_file = tmp_path / "custom_config.yaml"
    yaml_file.write_text("""
        base_model_id: "test/model"
        embedding_dim: 64
        rag_top_k: 10
        gemini_model: "gemini-2.5-flash"
    """)

    config = Config.from_yaml(str(yaml_file))
    assert config.base_model_id == "test/model"
    assert config.embedding_dim == 64
    assert config.rag_top_k == 10
    assert config.gemini_model == "gemini-2.5-flash"
    # Unspecified fields use dataclass defaults
    assert config.device == "auto"
    assert config.torch_dtype == "bfloat16"


def test_config_from_yaml_missing_file():
    """Verify that a non-existent YAML file triggers a warning and returns defaults."""
    with pytest.warns(ConfigFNFWarning, match="Configuration file not found"):
        config = Config.from_yaml("non_existent_config_file_xyz.yaml")
    assert isinstance(config, Config)
    assert config.base_model_id == "HuggingFaceTB/SmolVLM-256M-Instruct"


def test_config_from_yaml_invalid_syntax(tmp_path):
    """Verify that invalid YAML syntax raises ConfigParseError."""
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("::: invalid yaml syntax :::")

    with pytest.raises(ConfigParseError):
        Config.from_yaml(str(invalid_yaml))


def test_config_quantization():
    """Verify default and customized quantization handling."""
    config_default = Config()
    assert config_default.quantization == "f32"

    config_qi8 = Config(quantization="QI8")
    assert config_qi8.quantization == "qi8"


def test_dependency_injection_server_and_client():
    """Verify that components accept an injected Config without reloading from YAML."""
    from bbq.src.client import BBQClient
    from bbq.src.server import BBQServer

    custom_config = Config(embedding_dim=64, rag_top_k=9)

    server = BBQServer(config=custom_config)
    assert server.config is custom_config
    assert server.config.embedding_dim == 64
    assert server.config.rag_top_k == 9

    client = BBQClient(config=custom_config)
    assert client.config is custom_config
    assert client.config is not None
    assert client.config.embedding_dim == 64
