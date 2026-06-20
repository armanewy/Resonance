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
    assert config.notifications.enabled is False
    assert config.notifications.dry_run_stdout is True
    assert config.notifications.discovery_cooldown_hours == 24
    assert config.public_sources.eia_grid.enabled is False
    assert config.public_sources.eia_grid.poll_interval_seconds == 3600
    assert config.public_sources.eia_grid.initial_backfill_hours == 720
    assert config.public_sources.eia_grid.normal_lookback_hours == 72
    assert config.public_sources.eia_grid.maximum_gap_repair_hours == 2160
    assert config.public_sources.ripe_atlas.enabled is False
    assert config.public_sources.ripe_atlas.poll_interval_seconds == 900
    assert config.public_sources.ripe_atlas.measurement_ids == (1001, 1004, 1009)


def test_notification_configuration_loading(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        VALID_CONFIG
        + """
[notifications]
enabled = true
dry_run_stdout = false
ntfy_endpoint = "https://ntfy.example/resonance"
history_path = "tmp/notification_history.json"
dashboard_url = "http://127.0.0.1:8501"
discovery_cooldown_hours = 12
finding_cooldown_hours = 6
major_strengthening_threshold = 0.3
request_timeout_seconds = 2
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.notifications.enabled is True
    assert config.notifications.dry_run_stdout is False
    assert config.notifications.ntfy_endpoint == "https://ntfy.example/resonance"
    assert config.notifications.finding_cooldown_hours == 6
    assert config.notifications.major_strengthening_threshold == 0.3


def test_public_source_configuration_loading(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        VALID_CONFIG
        + """
[public_sources.eia_grid]
enabled = true
poll_interval_seconds = 1800
initial_backfill_hours = 24
normal_lookback_hours = 12
maximum_gap_repair_hours = 168

[public_sources.ripe_atlas]
enabled = true
poll_interval_seconds = 600
initial_backfill_hours = 48
normal_lookback_hours = 4
aggregation_seconds = 900
finalization_delay_seconds = 300
initial_radius_km = 100
maximum_radius_km = 300
desired_probe_count = 12
minimum_probe_count = 6
maximum_probes_per_asn = 1
maximum_anchor_count = 2
cohort_refresh_hours = 12
result_chunk_hours = 3
maximum_probe_batch_size = 25
maximum_requests_per_poll = 50
measurement_ids = [1001, 1004]
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.public_sources.eia_grid.enabled is True
    assert config.public_sources.eia_grid.poll_interval_seconds == 1800
    assert config.public_sources.eia_grid.initial_backfill_hours == 24
    assert config.public_sources.eia_grid.normal_lookback_hours == 12
    assert config.public_sources.eia_grid.maximum_gap_repair_hours == 168
    assert config.public_sources.ripe_atlas.enabled is True
    assert config.public_sources.ripe_atlas.poll_interval_seconds == 600
    assert config.public_sources.ripe_atlas.measurement_ids == (1001, 1004)


def test_malformed_configuration_has_helpful_error(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(VALID_CONFIG.replace("tcp_test_port = 443", "tcp_test_port = 70000"), encoding="utf-8")

    with pytest.raises(ConfigError, match="tcp_test_port"):
        load_config(path)


def test_invalid_notification_endpoint_has_helpful_error(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        VALID_CONFIG
        + """
[notifications]
ntfy_endpoint = "ntfy.example/resonance"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="notifications.ntfy_endpoint"):
        load_config(path)


def test_invalid_public_source_configuration_has_helpful_error(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        VALID_CONFIG
        + """
[public_sources.eia_grid]
poll_interval_seconds = 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="public_sources.eia_grid.poll_interval_seconds"):
        load_config(path)


def test_invalid_ripe_public_source_configuration_has_helpful_error(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        VALID_CONFIG
        + """
[public_sources.ripe_atlas]
initial_radius_km = 600
maximum_radius_km = 500
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="initial_radius_km"):
        load_config(path)
