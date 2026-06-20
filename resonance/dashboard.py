from __future__ import annotations

import json
import os
from datetime import timedelta
from io import StringIO
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from resonance.analysis import AlignedPair, LagScanResult, PairAnalysis, ValidationResult
from resonance.analysis.lifecycle import LifecycleEvent, fetch_lifecycle_events
from resonance.analysis.service import (
    MetricPairAnalysis,
    ValidationOptions,
    analyze_metric_pair,
    list_analyzable_metrics,
)
from resonance.config import ConfigError, load_config
from resonance.public_health import eia_source_health_rows
from resonance.public_sources.eia_grid import SOURCE_ID
from resonance.storage import (
    DEFAULT_DB_PATH,
    CorrelationFinding,
    EventMarker,
    correlation_finding_from_row,
    ensure_database,
    fetch_event_markers,
    fetch_correlation_findings,
    fetch_measurements,
    insert_event_marker,
    latest_measurement_by_metric,
    latest_timestamp_by_source,
    recent_errors,
    sample_counts_by_metric,
)
from resonance.time_utils import parse_utc, to_utc_iso, utc_now
from resonance.ui import (
    aligned_transformed_timeline,
    lag_profile,
    lagged_scatter,
    render_finding_card,
    stability_chart,
)
from resonance.ui.pair_explorer import (
    PAIR_EXPLORER_INTERVALS,
    PAIR_MAX_LAGS,
    PAIR_TRANSFORMS,
    PairExplorerSelection,
    coverage_rows,
    evidence_metrics,
    evidence_statement,
    max_lag_steps,
    metric_by_name,
    metric_names,
    pair_cadence_seconds,
    selected_interval,
    selected_max_lag,
    selected_transform,
    warning_messages,
)


INTERVALS = {
    "1 hour": timedelta(hours=1),
    "6 hours": timedelta(hours=6),
    "24 hours": timedelta(hours=24),
    "7 days": timedelta(days=7),
}

CARD_METRICS = {
    "TCP latency": "tcp_latency_ms",
    "DNS latency": "dns_latency_ms",
    "CPU": "cpu_percent",
    "Memory": "memory_percent",
    "Battery": "battery_percent",
    "Temperature": "weather_temperature_c",
    "Precipitation": "weather_precipitation_mm",
}

VISIBLE_FINDING_STATUSES = ("verified", "strengthened", "weakened", "broken")
ARCHIVED_FINDING_STATUSES = {"broken"}


st.set_page_config(page_title="Resonance", layout="wide")


def main() -> None:
    _auto_refresh(30)
    try:
        config = load_config()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    local_tz = ZoneInfo(config.location.timezone)
    now_utc = utc_now()
    selected = st.selectbox("Window", list(INTERVALS.keys()), index=1)
    start_utc = now_utc - INTERVALS[selected]

    conn = ensure_database(DEFAULT_DB_PATH)
    try:
        st.title("Resonance")
        header_cols = st.columns(4)
        header_cols[0].metric("Location", config.location.name)
        header_cols[1].metric("Local time", now_utc.astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S"))
        newest_personal = latest_timestamp_by_source(conn, "personal")
        newest_weather = latest_timestamp_by_source(conn, "open-meteo")
        header_cols[2].metric("Newest personal", _display_time(newest_personal, local_tz))
        header_cols[3].metric("Newest weather", _display_time(newest_weather, local_tz))

        stale_messages = _stale_messages(newest_personal, newest_weather, now_utc, config)
        for message in stale_messages:
            st.warning(message)

        rows = fetch_measurements(conn, start_utc, now_utc)
        df = _rows_to_dataframe(rows, local_tz)

        _current_cards(conn)
        _render_findings(conn)
        _render_event_markers(conn, local_tz)
        _render_connectivity(df)
        _render_utilization(df)
        _render_network(df)
        _render_weather(df)
        _render_public_sources(conn, config, now_utc, local_tz, start_utc)
        _render_tables(conn, start_utc, now_utc, df)
        _render_pair_explorer(DEFAULT_DB_PATH, now_utc)
    finally:
        conn.close()


def _auto_refresh(seconds: int) -> None:
    html = f"<script>setTimeout(function(){{window.location.reload();}}, {seconds * 1000});</script>"
    st.components.v1.html(html, height=0)


def _current_cards(conn) -> None:
    st.subheader("Current")
    metrics = {}
    for label, metric in CARD_METRICS.items():
        row = latest_measurement_by_metric(conn, metric)
        if row is not None:
            metrics[label] = row

    columns = st.columns(len(metrics) if metrics else 1)
    if not metrics:
        columns[0].metric("No measurements", "Waiting")
        return

    for column, (label, row) in zip(columns, metrics.items()):
        column.metric(label, _format_value(float(row["value"]), row["unit"]), help=f"source: {row['source']}")


def _render_findings(conn) -> None:
    st.subheader("Findings")
    show_archived = st.checkbox("Show archived findings", value=False, key="show_archived_findings")
    findings = dashboard_findings(conn, show_archived=show_archived)
    if not findings:
        st.info("No verified findings yet.")
        return

    for index, finding in enumerate(findings):
        if index:
            st.divider()
        st.caption(f"Lifecycle status: {finding.status}")
        render_finding_card(finding, analysis_from_finding(finding), streamlit=st)


def dashboard_findings(conn, *, show_archived: bool) -> tuple[CorrelationFinding, ...]:
    """Return lifecycle-backed findings for display, sorted by evidence quality."""

    latest_events = _latest_lifecycle_events(conn)
    stored_findings = {
        _finding_identity(finding): finding
        for finding in (correlation_finding_from_row(row) for row in fetch_correlation_findings(conn))
    }
    findings = []
    for event in latest_events:
        if event.status not in VISIBLE_FINDING_STATUSES:
            continue
        if event.status in ARCHIVED_FINDING_STATUSES and not show_archived:
            continue
        stored = stored_findings.get(_event_identity(event))
        finding = _finding_from_lifecycle_event(event, stored)
        if finding is not None:
            findings.append(finding)
    findings.sort(key=_evidence_quality_key)
    return tuple(findings)


def analysis_from_finding(finding: CorrelationFinding) -> PairAnalysis:
    evidence = _evidence(finding)
    cadence_seconds = max(1, _evidence_int(evidence, "cadence_seconds", 300))
    best_lag_steps = int(round(finding.lag_seconds / cadence_seconds))
    start_utc = _evidence_timestamp(evidence, "aligned_start_utc", finding.first_seen_utc)
    end_utc = _evidence_timestamp(evidence, "aligned_end_utc", finding.last_verified_utc)
    window_scores = tuple(score for score in evidence.get("window_scores", ()) if isinstance(score, Mapping))

    return PairAnalysis(
        aligned_pair=AlignedPair(
            x_metric=finding.x_metric,
            y_metric=finding.y_metric,
            cadence_seconds=cadence_seconds,
            frame=(),
            x_coverage=_evidence_float(evidence, "x_coverage", 0.0),
            y_coverage=_evidence_float(evidence, "y_coverage", 0.0),
            start_utc=start_utc,
            end_utc=end_utc,
        ),
        transform_name=finding.transform,
        lag_result=LagScanResult(
            scores=(
                {
                    "lag_steps": best_lag_steps,
                    "lag_seconds": finding.lag_seconds,
                    "rho": finding.discovery_rho,
                    "overlap_count": _evidence_int(evidence, "discovery_overlap", finding.overlap_count),
                },
            ),
            best_lag_steps=best_lag_steps,
            best_lag_seconds=finding.lag_seconds,
            best_rho=finding.discovery_rho,
        ),
        validation_result=ValidationResult(
            permutation_p_value=_evidence_optional_float(evidence, "permutation_p_value"),
            holdout_rho=finding.holdout_rho,
            holdout_overlap=finding.overlap_count,
            sign_stability=finding.stability,
            window_scores=window_scores,
            warnings=tuple(str(item) for item in evidence.get("warnings", ()) if item),
        ),
    )


def _latest_lifecycle_events(conn) -> tuple[LifecycleEvent, ...]:
    latest: dict[str, LifecycleEvent] = {}
    for event in fetch_lifecycle_events(conn):
        latest[event.relationship_key] = event
    return tuple(latest.values())


def _finding_from_lifecycle_event(
    event: LifecycleEvent,
    stored: CorrelationFinding | None,
) -> CorrelationFinding | None:
    if event.discovery_rho is None or event.holdout_rho is None:
        return None
    if event.corrected_q is None or event.stability is None or event.overlap_count is None:
        return None

    evidence = dict(stored.evidence) if stored is not None else {}
    return CorrelationFinding(
        x_metric=event.x_metric,
        y_metric=event.y_metric,
        transform=event.transform,
        lag_seconds=event.lag_seconds,
        discovery_rho=event.discovery_rho,
        holdout_rho=event.holdout_rho,
        corrected_q=event.corrected_q,
        stability=event.stability,
        overlap_count=event.overlap_count,
        first_seen_utc=stored.first_seen_utc if stored is not None else event.event_utc,
        last_verified_utc=event.event_utc,
        status=event.status,
        evidence=evidence,
    )


def _evidence_quality_key(finding: CorrelationFinding) -> tuple[float, float, int, float, str, str]:
    return (
        finding.corrected_q,
        -finding.stability,
        -finding.overlap_count,
        -abs(finding.holdout_rho),
        finding.x_metric,
        finding.y_metric,
    )


def _finding_identity(finding: CorrelationFinding) -> tuple[str, str, str]:
    return finding.x_metric, finding.y_metric, finding.transform


def _event_identity(event: LifecycleEvent) -> tuple[str, str, str]:
    return event.x_metric, event.y_metric, event.transform


def _render_event_markers(conn, local_tz: ZoneInfo) -> None:
    st.subheader("Event markers")
    with st.form("mark_event", clear_on_submit=True):
        label_col, note_col, button_col = st.columns([2, 4, 1])
        label = label_col.text_input("Label")
        note = note_col.text_input("Note")
        submitted = button_col.form_submit_button("Mark event now")
    if submitted:
        marked_at_utc = utc_now()
        try:
            insert_event_marker(conn, EventMarker(marked_at_utc, label, note, marked_at_utc))
        except ValueError as exc:
            st.error(str(exc))
        else:
            marked_at_local = marked_at_utc.astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"Event marked at {marked_at_local}.")

    recent_events = _event_rows_to_dataframe(fetch_event_markers(conn, 20), local_tz)
    st.dataframe(recent_events, use_container_width=True, hide_index=True)

    event_csv = _event_rows_csv(fetch_event_markers(conn, None), local_tz)
    st.download_button("Download events CSV", event_csv, "resonance_events.csv", "text/csv")


def _render_connectivity(df: pd.DataFrame) -> None:
    st.subheader("Connectivity")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Latency", "Failed samples"),
    )
    _add_trace(fig, df, "tcp_latency_ms", "TCP latency", row=1, col=1)
    _add_trace(fig, df, "dns_latency_ms", "DNS latency", row=1, col=1)
    failures = []
    for metric, label in [("tcp_success", "TCP failure"), ("dns_success", "DNS failure")]:
        part = df[(df["metric"] == metric) & (df["value"] == 0)] if not df.empty else df
        if not part.empty:
            for source, group in part.groupby("source"):
                failures.append((group, f"{label} ({source})"))
    for index, (group, label) in enumerate(failures):
        fig.add_trace(
            go.Scatter(
                x=group["local_time"],
                y=[index + 1] * len(group),
                mode="markers",
                name=label,
                marker={"size": 10, "symbol": "x"},
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text="ms", row=1, col=1)
    fig.update_yaxes(title_text="fail", row=2, col=1, tickmode="array", tickvals=[])
    _show_chart(fig)


def _render_utilization(df: pd.DataFrame) -> None:
    st.subheader("Computer utilization")
    fig = go.Figure()
    _add_trace(fig, df, "cpu_percent", "CPU")
    _add_trace(fig, df, "memory_percent", "Memory")
    fig.update_yaxes(title_text="percent")
    _show_chart(fig)


def _render_network(df: pd.DataFrame) -> None:
    st.subheader("Network throughput")
    scale, unit = _network_scale(df)
    fig = go.Figure()
    _add_trace(fig, df, "network_recv_bytes_per_second", "Receive", scale=scale)
    _add_trace(fig, df, "network_sent_bytes_per_second", "Send", scale=scale)
    fig.update_yaxes(title_text=unit)
    _show_chart(fig)


def _render_weather(df: pd.DataFrame) -> None:
    st.subheader("Local weather")
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Temperature", "Precipitation", "Wind speed"),
    )
    _add_trace(fig, df, "weather_temperature_c", "Temperature", row=1, col=1)
    _add_trace(fig, df, "weather_precipitation_mm", "Precipitation", row=2, col=1)
    _add_trace(fig, df, "weather_wind_speed_kmh", "Wind speed", row=3, col=1)
    fig.update_yaxes(title_text="C", row=1, col=1)
    fig.update_yaxes(title_text="mm", row=2, col=1)
    fig.update_yaxes(title_text="km/h", row=3, col=1)
    _show_chart(fig, height=620)


def _render_public_sources(conn, config, now_utc, local_tz: ZoneInfo, start_utc) -> None:
    st.subheader("Public data sources")
    rows = eia_source_health_rows(
        conn,
        config=config.public_sources.eia_grid,
        now_utc=now_utc,
        credential_available=bool(os.environ.get("EIA_API_KEY")),
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    chart_rows = _eia_chart_rows(conn, start_utc, now_utc, local_tz)
    if not chart_rows:
        return
    chart_df = pd.DataFrame(chart_rows)
    fig = go.Figure()
    for label, group in chart_df.groupby("series"):
        fig.add_trace(
            go.Scatter(
                x=group["local_time"],
                y=group["value"],
                mode="lines+markers",
                name=label,
            )
        )
    fig.update_yaxes(title_text="MWh")
    _show_chart(fig)


def _eia_chart_rows(conn, start_utc, end_utc, local_tz: ZoneInfo) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT s.display_name, o.valid_start_utc, o.value
        FROM public_observations o
        JOIN series_registry s ON s.series_id = o.series_id
        WHERE s.source_id = ?
          AND o.valid_start_utc >= ?
          AND o.valid_start_utc <= ?
          AND NOT EXISTS (
              SELECT 1
              FROM public_observations newer
              WHERE newer.series_id = o.series_id
                AND newer.source_observation_key = o.source_observation_key
                AND (
                    newer.ingested_at_utc > o.ingested_at_utc
                    OR (
                        newer.ingested_at_utc = o.ingested_at_utc
                        AND newer.observation_id > o.observation_id
                    )
                )
          )
        ORDER BY o.valid_start_utc ASC, s.display_name ASC
        """,
        (SOURCE_ID, to_utc_iso(start_utc), to_utc_iso(end_utc)),
    ).fetchall()
    return [
        {
            "series": row["display_name"],
            "local_time": parse_utc(row["valid_start_utc"]).astimezone(local_tz),
            "value": float(row["value"]),
        }
        for row in rows
    ]


def _render_tables(conn, start_utc, now_utc, df: pd.DataFrame) -> None:
    st.subheader("Sample counts by metric")
    counts = sample_counts_by_metric(conn, start_utc, now_utc)
    st.dataframe(pd.DataFrame([dict(row) for row in counts]), use_container_width=True, hide_index=True)

    st.subheader("Data coverage")
    coverage_rows = []
    if not df.empty:
        for (metric, source), group in df.groupby(["metric", "source"]):
            coverage_rows.append(
                {
                    "metric": metric,
                    "source": source,
                    "first": group["local_time"].min(),
                    "last": group["local_time"].max(),
                    "samples": len(group),
                }
            )
    st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)

    st.subheader("Recent collector errors")
    st.dataframe(pd.DataFrame([dict(row) for row in recent_errors(conn, 10)]), use_container_width=True, hide_index=True)

    csv_bytes = _dataframe_csv(df)
    st.download_button("Download CSV", csv_bytes, "resonance_measurements.csv", "text/csv")


def _render_pair_explorer(database_path, now_utc) -> None:
    st.divider()
    st.header("Pair Explorer")
    st.caption("Manual association check for one selected metric pair.")

    interval_label = st.selectbox(
        "Pair interval",
        list(PAIR_EXPLORER_INTERVALS.keys()),
        index=1,
        key="pair_explorer_interval",
    )
    start_utc = now_utc - selected_interval(interval_label)
    try:
        metric_summaries = list_analyzable_metrics(database_path, start_utc, now_utc)
    except (OSError, ValueError) as exc:
        st.warning(f"Pair Explorer is unavailable: {exc}")
        return

    names = metric_names(metric_summaries)
    if len(names) < 2:
        st.info("Pair Explorer needs at least two metrics in the selected interval.")
        return

    summaries_by_name = metric_by_name(metric_summaries)
    with st.form("pair_explorer_form"):
        columns = st.columns(4)
        x_metric_label = columns[0].selectbox("X metric", names, index=0)
        y_metric_label = columns[1].selectbox("Y metric", names, index=1)
        transform_label = columns[2].selectbox("Transform", list(PAIR_TRANSFORMS.keys()), index=0)
        max_lag_label = columns[3].selectbox("Maximum lag", list(PAIR_MAX_LAGS.keys()), index=3)
        submitted = st.form_submit_button("Analyze")

    x_metric = summaries_by_name[x_metric_label].metric
    y_metric = summaries_by_name[y_metric_label].metric
    summaries_by_metric = {summary.metric: summary for summary in metric_summaries}
    selection = PairExplorerSelection(
        x_metric=x_metric,
        y_metric=y_metric,
        interval_label=interval_label,
        transform_label=transform_label,
        max_lag_label=max_lag_label,
    )
    if submitted:
        _run_pair_explorer_analysis(database_path, start_utc, now_utc, summaries_by_metric, selection)

    result = st.session_state.get("pair_explorer_result")
    if not result or result.get("selection") != selection:
        st.info("Choose metrics and press Analyze to run the Pair Explorer.")
        return

    error = result.get("error")
    if error:
        st.warning(error)
        return

    analysis = result.get("analysis")
    if analysis is not None:
        _render_pair_explorer_result(analysis)


def _run_pair_explorer_analysis(
    database_path,
    start_utc,
    end_utc,
    summaries_by_name,
    selection: PairExplorerSelection,
) -> None:
    if selection.x_metric == selection.y_metric:
        st.session_state["pair_explorer_result"] = {
            "selection": selection,
            "analysis": None,
            "error": "Choose different X and Y metrics.",
        }
        return

    cadence_seconds = pair_cadence_seconds(
        summaries_by_name[selection.x_metric],
        summaries_by_name[selection.y_metric],
    )
    if cadence_seconds is None:
        st.session_state["pair_explorer_result"] = {
            "selection": selection,
            "analysis": None,
            "error": "Insufficient evidence: cadence could not be inferred for both selected metrics.",
        }
        return

    try:
        analysis = analyze_metric_pair(
            database_path,
            selection.x_metric,
            selection.y_metric,
            start_utc,
            end_utc,
            selected_transform(selection.transform_label),
            max_lag_steps=max_lag_steps(selected_max_lag(selection.max_lag_label), cadence_seconds),
            validation_options=ValidationOptions(cadence_seconds=cadence_seconds),
        )
    except ValueError as exc:
        st.session_state["pair_explorer_result"] = {
            "selection": selection,
            "analysis": None,
            "error": f"Insufficient evidence: {exc}",
        }
        return

    st.session_state["pair_explorer_result"] = {
        "selection": selection,
        "analysis": analysis,
        "error": None,
    }


def _render_pair_explorer_result(analysis: MetricPairAnalysis) -> None:
    for message in warning_messages(analysis):
        st.warning(message)

    statement = evidence_statement(analysis)
    if statement.startswith("Insufficient evidence"):
        st.warning(statement)
    else:
        st.info(statement)

    st.subheader("Coverage and samples")
    st.dataframe(pd.DataFrame(coverage_rows(analysis)), use_container_width=True, hide_index=True)

    columns = st.columns(5)
    for column, (label, value) in zip(columns, evidence_metrics(analysis).items()):
        column.metric(label, value)

    st.subheader("Evidence charts")
    _show_chart(aligned_transformed_timeline(analysis))
    _show_chart(lag_profile(analysis))
    _show_chart(lagged_scatter(analysis))
    _show_chart(stability_chart(analysis))


def _rows_to_dataframe(rows, local_tz: ZoneInfo) -> pd.DataFrame:
    records = []
    for row in rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        timestamp = parse_utc(row["timestamp_utc"])
        records.append(
            {
                "timestamp_utc": row["timestamp_utc"],
                "local_time": timestamp.astimezone(local_tz),
                "metric": row["metric"],
                "value": float(row["value"]),
                "unit": row["unit"],
                "source": row["source"],
                "metadata_json": row["metadata_json"],
                "metadata": metadata,
            }
        )
    return pd.DataFrame.from_records(records)


def _event_rows_to_dataframe(rows, local_tz: ZoneInfo) -> pd.DataFrame:
    records = []
    for row in rows:
        timestamp = parse_utc(row["timestamp_utc"]).astimezone(local_tz)
        created_at = parse_utc(row["created_at_utc"]).astimezone(local_tz)
        records.append(
            {
                "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "label": row["label"],
                "note": row["note"],
                "created": created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return pd.DataFrame.from_records(records, columns=["time", "label", "note", "created"])


def _add_trace(
    fig,
    df: pd.DataFrame,
    metric: str,
    label: str,
    row: int | None = None,
    col: int | None = None,
    scale: float = 1.0,
) -> None:
    if df.empty:
        return
    part = df[df["metric"] == metric].copy()
    if part.empty:
        return
    part["scaled_value"] = part["value"] / scale
    for source, group in part.groupby("source"):
        trace = go.Scatter(
            x=group["local_time"],
            y=group["scaled_value"],
            mode="lines+markers",
            name=f"{label} ({source})",
        )
        if row is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=col)


def _show_chart(fig, height: int = 360) -> None:
    if not fig.data:
        fig.add_annotation(text="No data in selected window", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(height=height, margin={"l": 30, "r": 20, "t": 45, "b": 30}, legend={"orientation": "h"})
    st.plotly_chart(fig, use_container_width=True)


def _network_scale(df: pd.DataFrame) -> tuple[float, str]:
    if df.empty:
        return 1024.0, "KB/s"
    part = df[df["metric"].isin(["network_recv_bytes_per_second", "network_sent_bytes_per_second"])]
    if part.empty:
        return 1024.0, "KB/s"
    maximum = part["value"].max()
    if maximum >= 1024 * 1024:
        return 1024.0 * 1024.0, "MB/s"
    return 1024.0, "KB/s"


def _format_value(value: float, unit: str) -> str:
    if unit == "boolean":
        return "yes" if value else "no"
    if unit == "percent":
        return f"{value:.0f}%"
    if unit == "ms":
        return f"{value:.1f} ms"
    if unit == "bytes/second":
        if value >= 1024 * 1024:
            return f"{value / (1024 * 1024):.2f} MB/s"
        return f"{value / 1024:.1f} KB/s"
    return f"{value:.1f} {unit}"


def _display_time(value: str | None, local_tz: ZoneInfo) -> str:
    if not value:
        return "none"
    return parse_utc(value).astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S")


def _stale_messages(newest_personal, newest_weather, now_utc, config) -> list[str]:
    messages = []
    personal_stale_after = config.collection.personal_interval_seconds * 3
    weather_stale_after = config.collection.weather_interval_seconds * 2
    if _is_stale(newest_personal, now_utc, personal_stale_after):
        messages.append("Personal collector is stale or has not produced samples.")
    if _is_stale(newest_weather, now_utc, weather_stale_after):
        messages.append("Weather collector is stale or has not produced samples.")
    return messages


def _is_stale(timestamp_utc: str | None, now_utc, stale_after_seconds: int) -> bool:
    if timestamp_utc is None:
        return True
    return (now_utc - parse_utc(timestamp_utc)).total_seconds() > stale_after_seconds


def _dataframe_csv(df: pd.DataFrame) -> bytes:
    if df.empty:
        return b""
    output = StringIO()
    df.drop(columns=["metadata"], errors="ignore").to_csv(output, index=False)
    return output.getvalue().encode("utf-8")


def _event_rows_csv(rows, local_tz: ZoneInfo) -> bytes:
    records = []
    for row in rows:
        timestamp = parse_utc(row["timestamp_utc"]).astimezone(local_tz)
        created_at = parse_utc(row["created_at_utc"]).astimezone(local_tz)
        records.append(
            {
                "id": row["id"],
                "timestamp_utc": row["timestamp_utc"],
                "timestamp_local": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "label": row["label"],
                "note": row["note"],
                "created_at_utc": row["created_at_utc"],
                "created_at_local": created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    output = StringIO()
    pd.DataFrame.from_records(
        records,
        columns=[
            "id",
            "timestamp_utc",
            "timestamp_local",
            "label",
            "note",
            "created_at_utc",
            "created_at_local",
        ],
    ).to_csv(output, index=False)
    return output.getvalue().encode("utf-8")


def _evidence(finding: CorrelationFinding) -> Mapping[str, Any]:
    return finding.evidence if isinstance(finding.evidence, Mapping) else {}


def _evidence_timestamp(evidence: Mapping[str, Any], key: str, default) -> Any:
    value = evidence.get(key)
    if isinstance(value, str) and value:
        try:
            return parse_utc(value)
        except ValueError:
            return default
    return default


def _evidence_int(evidence: Mapping[str, Any], key: str, default: int) -> int:
    value = evidence.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _evidence_float(evidence: Mapping[str, Any], key: str, default: float) -> float:
    value = _evidence_optional_float(evidence, key)
    return default if value is None else value


def _evidence_optional_float(evidence: Mapping[str, Any], key: str) -> float | None:
    value = evidence.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
