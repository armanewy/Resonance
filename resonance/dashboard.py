from __future__ import annotations

import json
from datetime import timedelta
from io import StringIO
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from resonance.config import ConfigError, load_config
from resonance.storage import (
    DEFAULT_DB_PATH,
    EventMarker,
    ensure_database,
    fetch_event_markers,
    fetch_measurements,
    insert_event_marker,
    latest_measurement_by_metric,
    latest_timestamp_by_source,
    recent_errors,
    sample_counts_by_metric,
)
from resonance.time_utils import parse_utc, utc_now


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
        _render_event_markers(conn, local_tz)
        _render_connectivity(df)
        _render_utilization(df)
        _render_network(df)
        _render_weather(df)
        _render_tables(conn, start_utc, now_utc, df)
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


if __name__ == "__main__":
    main()

