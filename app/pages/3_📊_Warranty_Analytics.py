"""PartSense: Warranty & Aftersales Analytics Platform (DuckDB Engine).

Executes versioned SQL queries on the synthetic aftersales layer to track
failure rates, MTBF (operating hours = tool wear min / 60), claims per 1k sales (IPTV),
warranty cost per unit, 80/20 Pareto root causes, monthly trends, and regional heatmaps.
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.utils import (
    get_kpi_runner,
    render_header,
    render_kpi_card,
)

st.set_page_config(
    page_title="Warranty Analytics – PartSense",
    page_icon="📊",
    layout="wide",
)

render_header(
    title="📊 Warranty & Aftersales Analytics",
    subtitle="DuckDB SQL queries on aftersales ledger, claims rate (IPTV), MTBF, and 80/20 Pareto spend",
    badge="SYNTHETIC BENCHMARK LAYER",
)

# Prominent Synthetic Layer Disclaimer
st.markdown(
    """
    <div style="background-color: #1E293B; border-left: 4px solid #38BDF8; padding: 0.85rem 1.25rem; border-radius: 6px; margin-bottom: 1.5rem;">
        <span style="font-weight: 700; color: #38BDF8; font-size: 0.95rem;">⚙️ SYNTHETIC BENCHMARK LAYER DISCLOSURE:</span>
        <span style="color: #CBD5E1; font-size: 0.9rem;">
            The warranty claims, parts catalog, monthly sales, and regional deployments shown below are generated
            deterministically with a fixed seed (<code>seed=42</code>) mapped to actual telemetry failure events.
            <b>Domain Assumption:</b> Component operating hours are defined as <code>operating_hours = tool_wear_min / 60.0</code>.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Run KPI queries through KPIRunner
runner = get_kpi_runner()

try:
    df_mtbf = runner.run_query("mtbf_by_part_family")
    df_claims_rate = runner.run_query("warranty_claims_per_1000_sales")
    df_cost_per_unit = runner.run_query("warranty_cost_per_unit_sold")
    df_monthly = runner.run_query("monthly_claim_trends")
    df_pareto = runner.run_query("pareto_failure_modes")
    df_dimension = runner.run_query("failure_rate_by_dimension")
    df_top_risk = runner.run_query("top_parts_at_risk")
except Exception as e:
    st.error(
        f"Failed to execute DuckDB SQL queries: {e}. Ensure data/synthetic/aftersales.duckdb exists."
    )
    st.stop()

# Top KPI Summary Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_warranty_spend = df_pareto["total_claim_cost"].sum()
overall_mtbf = df_mtbf["total_operating_hours"].sum() / max(1, df_mtbf["failure_count"].sum())
overall_claims_per_1k = (
    df_claims_rate["total_claims"].sum() / max(1, df_claims_rate["total_units_sold"].sum()) * 1000.0
)
overall_cost_per_unit = df_cost_per_unit["total_warranty_cost"].sum() / max(
    1, df_cost_per_unit["total_units_sold"].sum()
)

with kpi1:
    render_kpi_card(
        label="Total Warranty Spend",
        value=f"${total_warranty_spend:,.2f}",
        delta=f"{df_pareto['claim_count'].sum()} Filed Claims",
        delta_color="normal",
        help_text="Cumulative cost of all settled warranty claims across 24-month horizon",
    )

with kpi2:
    render_kpi_card(
        label="Fleet MTBF",
        value=f"{overall_mtbf:.1f} hrs",
        delta="Operating Hours Basis",
        delta_color="positive",
        help_text="Mean Time Between Failures: (Sum of tool_wear_min / 60) / Total Failures",
    )

with kpi3:
    render_kpi_card(
        label="Warranty Claims / 1k Units",
        value=f"{overall_claims_per_1k:.2f}",
        delta="IPTV Industry Metric",
        delta_color="normal",
        help_text="Incident per Thousand Vehicles / Sold Units",
    )

with kpi4:
    render_kpi_card(
        label="Warranty Cost / Unit Sold",
        value=f"${overall_cost_per_unit:.2f}",
        delta="Per Part Manufactured",
        delta_color="normal",
        help_text="Total warranty expense divided by total sales volume",
    )

st.markdown("---")

# Row 1: Pareto 80/20 Analysis & Monthly Trend
col_pareto, col_trend = st.columns([1, 1])

with col_pareto:
    st.subheader("🎯 Pareto Analysis of Failure Modes (80/20 Rule)")
    st.caption("Cumulative warranty cost distribution across root cause failure mechanisms")

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pareto.add_trace(
        go.Bar(
            x=df_pareto["primary_failure_mode"],
            y=df_pareto["total_claim_cost"],
            name="Claim Cost ($)",
            marker_color="#38BDF8",
            text=[f"${c:,.0f}" for c in df_pareto["total_claim_cost"]],
            textposition="auto",
        ),
        secondary_y=False,
    )
    fig_pareto.add_trace(
        go.Scatter(
            x=df_pareto["primary_failure_mode"],
            y=df_pareto["cumulative_pct"],
            name="Cumulative %",
            line={"color": "#EF4444", "width": 3},
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    # Add 80% reference line
    fig_pareto.add_hline(
        y=80,
        line_dash="dot",
        line_color="#F59E0B",
        secondary_y=True,
        annotation_text="80% Threshold",
        annotation_position="bottom right",
    )
    fig_pareto.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"},
        height=320,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig_pareto.update_yaxes(title_text="Total Cost ($)", secondary_y=False)
    fig_pareto.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
    st.plotly_chart(fig_pareto, use_container_width=True)

with col_trend:
    st.subheader("📈 Monthly Warranty Claims & Spend Trend")
    st.caption("24-month in-service timeline tracking volume and financial exposure")

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Bar(
            x=df_monthly["claim_month"],
            y=df_monthly["total_claims"],
            name="Claims Count",
            marker_color="#1E3A8A",
            opacity=0.8,
        ),
        secondary_y=False,
    )
    fig_trend.add_trace(
        go.Scatter(
            x=df_monthly["claim_month"],
            y=df_monthly["total_claim_amount"],
            name="Spend ($)",
            line={"color": "#10B981", "width": 3},
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"},
        height=320,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig_trend.update_yaxes(title_text="Claims Filed", secondary_y=False)
    fig_trend.update_yaxes(title_text="Claim Spend ($)", secondary_y=True)
    st.plotly_chart(fig_trend, use_container_width=True)

# Row 2: Regional Heatmap & MTBF Breakdown
st.markdown("---")
col_dim1, col_dim2 = st.columns([1, 1])

with col_dim1:
    st.subheader("🗺️ Regional Failure Rate Heatmap")
    st.caption("Failure frequency (%) by Spare Part Family across operating geographic regions")

    heat_pivot = df_dimension.pivot_table(
        index="part_family",
        columns="region",
        values="failure_rate_pct",
        aggfunc="mean",
    ).round(2)

    fig_heat = px.imshow(
        heat_pivot,
        text_auto=True,
        color_continuous_scale="Viridis",
        labels={"x": "Region", "y": "Part Family", "color": "Failure %"},
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"},
        height=320,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_dim2:
    st.subheader("⏱️ MTBF Benchmark by Spare Part Family")
    st.caption("Operating hours before breakdown per component family")

    fig_mtbf = px.bar(
        df_mtbf,
        x="part_family",
        y="mtbf_hours",
        color="failure_count",
        color_continuous_scale="Tealgrn",
        text="mtbf_hours",
        labels={"mtbf_hours": "MTBF (Hours)", "part_family": "Spare Part Family"},
    )
    fig_mtbf.update_traces(texttemplate="%{text:.1f} hrs", textposition="outside")
    fig_mtbf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"},
        height=320,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    st.plotly_chart(fig_mtbf, use_container_width=True)

# Row 3: Top Components at Risk
st.markdown("---")
st.subheader("🚨 Top Components at High Risk (DuckDB Live Query)")
st.caption(
    "Query `top_parts_at_risk.sql`: high wear minutes, high strain, and severe thermal divergence"
)

st.dataframe(
    df_top_risk,
    use_container_width=True,
    height=280,
)
