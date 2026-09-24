"""Portfolio insights: observed default rates across segments of the training portfolio."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .common import BLUE, MUTED, NEUTRAL, figure, num, page_header, pct, require_service, show


def render() -> None:
    service = require_service()
    data = service.insights()
    if data is None:
        st.error("No insights found; run `python -m credscore.pipeline insights`.")
        return

    kpis, segments = data["kpis"], data["segments"]
    base_rate = kpis["default_rate"]

    page_header("Portfolio insights",
                f"Observed outcomes for the {num(kpis['applicants'])} labelled loan applications the model learned from.")

    cols = st.columns(6)
    cols[0].metric("Applications", num(kpis["applicants"]))
    cols[1].metric("Default rate", pct(base_rate, 2), f"{num(kpis['defaults'])} with payment difficulties",
                   delta_color="off", delta_arrow="off")
    cols[2].metric("Median income", num(kpis["median_income"]))
    cols[3].metric("Median credit", num(kpis["median_credit"]))
    cols[4].metric("Median age", f"{kpis['median_age']} yrs")
    cols[5].metric("Bureau history", pct(kpis["share_with_bureau_history"], 0), "share with a bureau record",
                   delta_color="off", delta_arrow="off")

    by_key = {s["key"]: s for s in segments}
    pick, note = st.columns([1, 2], vertical_alignment="bottom")
    key = pick.selectbox("Break down by", list(by_key), format_func=lambda k: by_key[k]["title"])
    segment = by_key[key]
    note.caption(segment.get("description", ""))

    rows = segment["rows"] if segment.get("ordered") else sorted(segment["rows"], key=lambda r: -r["default_rate"])
    height = max(320, len(rows) * 30 + 70)
    labels = [r["segment"] for r in rows][::-1]

    left, right = st.columns(2, gap="large")
    with left:
        rates = [r["default_rate"] for r in rows]
        extremes = {max(rates), min(rates)}
        fig = figure(height=height, showlegend=False, title=f"Default rate by {segment['title'].lower()}")
        fig.add_trace(go.Bar(
            x=rates[::-1], y=labels, orientation="h", marker_color=BLUE,
            text=[pct(v) if v in extremes else "" for v in rates][::-1], textposition="outside",
            customdata=[[r["applicants"], r["share"]] for r in rows][::-1],
            hovertemplate="%{y}<br>Default rate %{x:.2%}<br>%{customdata[0]:,} applicants "
                          "(%{customdata[1]:.1%})<extra></extra>",
        ))
        fig.add_vline(x=base_rate, line_dash="dot", line_color=MUTED)
        fig.update_xaxes(tickformat=".0%", rangemode="tozero")
        show(fig)
        st.caption(f"Dotted line: portfolio average of {pct(base_rate)}. Extremes are labelled; hover for the rest.")
    with right:
        fig = figure(height=height, showlegend=False, title="Applicant mix")
        fig.add_trace(go.Bar(
            x=[r["share"] for r in rows][::-1], y=labels, orientation="h", marker_color=NEUTRAL,
            customdata=[r["applicants"] for r in rows][::-1],
            hovertemplate="%{y}<br>%{customdata:,} applicants (%{x:.1%})<extra></extra>",
        ))
        fig.update_xaxes(tickformat=".0%", rangemode="tozero")
        show(fig)

    with st.expander("View as table"):
        st.dataframe(
            pd.DataFrame(segment["rows"]).rename(columns={
                "segment": segment["title"], "applicants": "Applicants", "share": "Share", "default_rate": "Default rate"}),
            hide_index=True,
            column_config={"Share": st.column_config.NumberColumn(format="percent"),
                           "Default rate": st.column_config.NumberColumn(format="percent")},
        )

    _extremes(segments, base_rate)


def _extremes(segments: list[dict], base_rate: float) -> None:
    st.subheader("Where the risk concentrates")
    st.caption("Segments holding at least 2% of applicants, across every breakdown above.")
    frame = pd.DataFrame([
        {"Segment": f"{s['title']}: {r['segment']}", "Share": r["share"], "Default rate": r["default_rate"],
         "vs portfolio": r["default_rate"] / base_rate}
        for s in segments for r in s["rows"] if r["share"] >= 0.02
    ]).sort_values("Default rate", ascending=False)
    config = {
        "Share": st.column_config.NumberColumn(format="percent"),
        "Default rate": st.column_config.NumberColumn(format="percent"),
        "vs portfolio": st.column_config.NumberColumn(format="%.1f×"),
    }
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Highest default rates**")
        st.dataframe(frame.head(8), hide_index=True, column_config=config)
    with right:
        st.markdown("**Lowest default rates**")
        st.dataframe(frame.tail(8).iloc[::-1], hide_index=True, column_config=config)
