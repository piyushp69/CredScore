"""Batch scoring: score a CSV of applicants and review the distribution of outcomes."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .common import BLUE, DECISION, MUTED, RISKY, STRONG, demo_batch, figure, num, page_header, pct, require_service, show

BIN = 20


def _read_csv(data: bytes) -> list[dict]:
    # applicant_id stays text ("007" must not become 7); "utf-8-sig" drops Excel's BOM.
    frame = pd.read_csv(io.BytesIO(data), dtype={"applicant_id": str}, encoding="utf-8-sig", skipinitialspace=True)
    if frame.empty:
        raise ValueError("the file has no applicant rows")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame.replace({np.nan: None}).to_dict(orient="records")


def _load(rows: list[dict], source: str) -> None:
    st.session_state["batch_rows"] = rows
    st.session_state["batch_source"] = source
    st.session_state.pop("batch_response", None)


def render() -> None:
    service = require_service()
    schema, info = service.schema(), service.model_info()
    profile_schema = schema["profile"]
    fields = list(profile_schema["properties"])

    page_header("Batch scoring",
                "Score a whole file of applicants in one go. Invalid rows are flagged individually, not fatal.")

    examples = [
        {"applicant_id": "EXAMPLE-TYPICAL", **schema["defaults"]},
        {"applicant_id": "EXAMPLE-STRONG", **STRONG},
        {"applicant_id": "EXAMPLE-RISKY", **RISKY},
    ]
    template = pd.DataFrame(examples, columns=["applicant_id", *fields]).to_csv(index=False)
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("⤓ Download CSV template", template, file_name="credscore_template.csv",
                       mime="text/csv", width="stretch")
    if c2.button("✨ Load 250 demo applicants", width="stretch"):
        _load(demo_batch(schema["defaults"]), "demo applicants")

    upload = st.file_uploader("Upload applicants (CSV)", type=["csv"],
                              help="One row per applicant, columns as in the template. "
                                   "Optional columns may be left out or empty.")
    if upload is not None and st.session_state.get("batch_upload_id") != upload.file_id:
        st.session_state["batch_upload_id"] = upload.file_id
        try:
            _load(_read_csv(upload.getvalue()), upload.name)
        except Exception as exc:
            st.error(f"Could not read {upload.name}: {exc}")

    rows = st.session_state.get("batch_rows")
    if not rows:
        return

    # Column checks look at the header (first row) only; unknown columns are dropped.
    header = list(rows[0])
    unknown = [c for c in header if c != "applicant_id" and c not in fields]
    missing = [f for f in profile_schema.get("required", []) if f not in header]
    clean = [{k: v for k, v in row.items() if k not in unknown} for row in rows] if unknown else rows

    with st.container(border=True):
        left, right = st.columns([3, 1], vertical_alignment="center")
        left.markdown(f"**{num(len(rows))} applicants** loaded from {st.session_state['batch_source']}")
        go_score = right.button(f"⚡ Score {num(len(rows))} applicants", type="primary",
                                disabled=bool(missing), width="stretch")
        if unknown:
            st.warning(f"Ignoring unrecognised columns: {', '.join(unknown)}")
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")

    if go_score:
        with st.spinner("Scoring…"):
            st.session_state["batch_response"] = service.score_batch(clean, n_reasons=3)

    response = st.session_state.get("batch_response")
    if response:
        _results(response, clean, info["scorecard"]["bands"])


def _results(response: dict, rows: list[dict], bands: list[dict]) -> None:
    summary, results = response["summary"], response["results"]
    table = pd.DataFrame([
        {
            "Applicant": item["applicant_id"] or f"row {item['index'] + 1}",
            "Score": (item["result"] or {}).get("credit_score"),
            "PD": (item["result"] or {}).get("probability_of_default"),
            "Band": ((item["result"] or {}).get("risk_band") or {}).get("code", ""),
            "Decision": DECISION[item["result"]["decision"]]["label"] if item["result"] else "",
            "Main risk factors": " · ".join((item["result"] or {}).get("reasons") or []),
            "Error": item["error"] or "",
        }
        for item in results
    ])
    scored = summary["scored"] or 1

    cols = st.columns(6)
    cols[0].metric("Scored", num(summary["scored"]), f"{num(summary['failed'])} failed validation",
                   delta_color="off" if not summary["failed"] else "inverse", delta_arrow="off")
    for col, key in zip(cols[1:4], DECISION):
        count = summary["decisions"][key]
        col.metric(DECISION[key]["label"], pct(count / scored, 0), f"{num(count)} applicants",
                   delta_color="off", delta_arrow="off")
    cols[4].metric("Average PD", pct(summary["mean_probability_of_default"]))
    cols[5].metric("Average score", num(summary["mean_credit_score"]))

    left, right = st.columns(2, gap="large")
    with left:
        scores = table["Score"].dropna()
        edges = np.arange(300, 850 + BIN, BIN)
        counts, _ = np.histogram(scores.clip(300, 850 - 1e-9), bins=edges)
        fig = figure(height=340, showlegend=False, title="Credit score distribution")
        fig.add_trace(go.Bar(x=edges[:-1] + BIN / 2, y=counts, width=BIN * 0.9, marker_color=BLUE,
                             customdata=np.stack([edges[:-1], edges[1:]], axis=1),
                             hovertemplate="Score %{customdata[0]}–%{customdata[1]}<br>%{y:,} applicants<extra></extra>"))
        for band in bands:
            if band["min_score"] > 300:
                fig.add_vline(x=band["min_score"], line_color=MUTED, line_width=1,
                              annotation_text=band["code"], annotation_position="top")
        fig.update_xaxes(title="Credit score", range=[300, 850])
        show(fig)
        st.caption("Vertical rules mark the risk-band boundaries.")
    with right:
        keys = list(DECISION)
        fig = figure(height=340, showlegend=False, title="Decisions")
        fig.add_trace(go.Bar(
            x=[summary["decisions"][k] for k in keys][::-1],
            y=[f"{DECISION[k]['icon']} {DECISION[k]['label']}" for k in keys][::-1],
            orientation="h", marker_color=[DECISION[k]["color"] for k in keys][::-1],
            text=[num(summary["decisions"][k]) for k in keys][::-1], textposition="outside",
            hovertemplate="%{y}: %{x:,} applicants<extra></extra>",
        ))
        # Headroom so the count printed past the longest bar is not clipped.
        fig.update_xaxes(range=[0, (max(summary["decisions"].values()) or 1) * 1.15])
        show(fig)

    head, button = st.columns([3, 1], vertical_alignment="bottom")
    head.subheader("Results")
    export = pd.concat([pd.DataFrame(rows).reset_index(drop=True), table.drop(columns=["Applicant"])], axis=1)
    button.download_button("⤓ Download results", export.to_csv(index=False), file_name="credscore_results.csv",
                           mime="text/csv", width="stretch")
    st.dataframe(table, hide_index=True, height=460, column_config={
        "PD": st.column_config.NumberColumn(format="percent"),
        "Main risk factors": st.column_config.TextColumn(width="large"),
    })
