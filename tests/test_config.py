from __future__ import annotations

import pytest

from resonance.config import ConfigError, load_config


VALID_CONFIG = """
[location]
name = "Framingham, Massachusetts"
latitude = 42.2793
longitude = -71.4162
timezone = "America/New_York"

[collection]
personal_interval_seconds = 30
weather_interval_seconds = 900
tcp_test_host = "1.1.1.1"
tcp_test_port = 443
dns_test_hostname = "example.com"
router_host = ""
"""


def test_configuration_loading_and_validation(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(VALID_CONFIG, encoding="utf-8")

    config = load_config(path)

    assert config.location.name == "Framingham, Massachusetts"
    assert config.location.latitude == 42.2793
    assert config.collection.tcp_test_port == 443


def test_malformed_configuration_has_helpful_error(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(VALID_CONFIG.replace("tcp_test_port = 443", "tcp_test_port = 70000"), encoding="utf-8")

    with pytest.raises(ConfigError, match="tcp_test_port"):
        load_config(path)

