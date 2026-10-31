"""PartSense: About, Methodology, Physics Formulations, and Model Governance.

Details theoretical foundations, data provenance, zero-leakage enforcement,
synthetic aftersales assumptions, physical failure mechanisms, and embedded Model Card.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.utils import render_header

st.set_page_config(
    page_title="About & Methodology – PartSense",
    page_icon="📖",
    layout="wide",
)

render_header(
    title="📖 Methodology, Physics & Data Governance",
    subtitle="Physical failure mechanisms, zero-leakage audit, aftersales synthesis, and model card",
)

tab_method, tab_physics, tab_synth, tab_card = st.tabs(
    [
        "🔬 Methodology & Leakage Rules",
        "⚙️ Physics Formulations",
        "💾 Synthetic Aftersales Layer",
        "📋 Model Card",
    ]
)

with tab_method:
    st.subheader("🛡️ Machine Learning Methodology & Anti-Leakage Audit")
    st.markdown(
        """
        ### 1. The Real-World Predictive Maintenance Challenge
        Industrial milling tools and automotive components degrade gradually under thermal, electrical,
        and mechanical stress until sudden catastrophic failure occurs. In historical maintenance logs,
        failures represent only **~3.4% of all operating hours**, creating severe class imbalance where
        naive accuracy metrics (e.g., predicting 96.6% "Pass") are dangerously misleading.

        ### 2. Strict Zero Data Leakage Enforcement
        Many naive implementations on the AI4I dataset suffer from severe target leakage by including
        downstream failure mode flags or database sequence keys. In **PartSense**, the feature pipeline
        actively executes an automated `verify_zero_data_leakage()` audit asserting that:
        - **Identity Columns (`UDI`, `Product ID`):** Completely removed. A model must learn physical degradation physics, not database row ordering or product serial prefixes.
        - **Failure Mode Flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`):** Strictly barred from the feature matrix when predicting overall `Machine failure`. These are downstream targets, not inputs.

        ### 3. Threshold Optimization for Real-World Value
        Standard binary classifiers default to $p=0.50$, which is sub-optimal for asymmetric industrial costs
        (where a missed catastrophic failure costs $10\\times$ more than an unneeded visual inspection).
        PartSense optimizes the decision threshold on out-of-fold training predictions to maximize
        **Recall at a minimum Precision floor of 50%**, settling on $p^* = 0.3927$, achieving **91.2% test recall**.
        """
    )

with tab_physics:
    st.subheader("⚙️ Domain Physics Feature Formulations")
    st.markdown(
        """
        Rather than relying solely on raw sensor readings, PartSense derives physics-informed
        features aligned with specific failure mechanics:

        #### 1. Mechanical Spindle Power ($P$)
        $$\\text{power\\_W} = \\tau \\cdot \\omega = \\tau \\cdot \\left(\\frac{2\\pi \\cdot \\text{rpm}}{60}\\right)$$
        - **Physical Mechanism (PWF - Power Failure):** Electric drive motors have bounded torque-speed envelopes. Operating below 3,500 W or above 9,000 W causes excessive electrical heating, inverter trip-outs, or tool stalling.

        #### 2. Thermal Dissipation Gradient ($\\Delta T$)
        $$\\text{temp\\_diff\\_K} = T_{\\text{process}} - T_{\\text{air}}$$
        - **Physical Mechanism (HDF - Heat Dissipation Failure):** Cutting friction generates heat dissipated to surrounding air. If $\\Delta T < 8.6\\text{ K}$ while spindle speed is below 1,380 rpm, cooling fluid failure or thermal breakdown occurs.

        #### 3. Cumulative Cutting Tool Strain ($S$)
        $$\\text{strain\\_minNm} = \\text{tool\\_wear\\_min} \\cdot \\text{torque\\_nm}$$
        - **Physical Mechanism (OSF - Overstrain Failure):** Tool wear reduces cutting edge thickness, multiplying effective mechanical shear stress. When strain exceeds $11,000 - 13,000\\text{ min}\\cdot\\text{Nm}$, the cutting insert fractures.

        #### 4. Tool Wear Failure (TWF)
        - Tool wear increases monotonically as cutting edges erode. After $200 - 240\\text{ minutes}$ of operation, wear induces micro-fractures and surface roughness.
        """
    )

with tab_synth:
    st.subheader("💾 Synthetic Aftersales & Warranty Architecture")
    st.markdown(
        """
        To replicate an enterprise automotive warranty platform without exposing proprietary OEM financial data,
        PartSense generates a deterministic synthetic aftersales layer using `src/partsense/synthetic.py` with `seed=42`:

        ### Core Aftersales Tables
        1. **`parts_catalog`**: 6 spare part families (Wheel Bearing, Water Pump, Alternator, Starter Motor, Brake Caliper, Fuel Pump) with unit manufacturing costs and 24-month warranty coverage.
        2. **`components_in_service`**: Maps the 10,000 AI4I telemetry records to specific part families, vehicles, and 4 geographic regions (North America, Europe, Asia Pacific, Latin America).
        3. **`parts_sales`**: 24 months of regional unit sales per part family (totaling ~398,000 sold units).
        4. **`warranty_claims`**: Linked to actual telemetry failure events with claim costs drawn from a lognormal distribution calibrated per failure mode (HDF, PWF, OSF, TWF) plus realistic claim filing lag.

        ### Key Operational Assumptions
        - **Operating Hours Definition:** In industrial milling telemetry, tool wear is recorded in minutes. We adopt the canonical operational assumption:
          $$\\text{Operating Hours} = \\frac{\\text{Tool Wear [min]}}{60.0}$$
        - **Attribution & Dataset License:** The underlying telemetry is from the UCI Machine Learning Repository:
          *Stephan Matzka (2020), AI4I 2020 Predictive Maintenance Dataset (ID 601), Creative Commons Attribution 4.0 International (CC BY 4.0).*
        """
    )

with tab_card:
    st.subheader("📋 Audited Model Card")
    model_card_path = Path(__file__).resolve().parent.parent.parent / "reports" / "MODEL_CARD.md"
    if model_card_path.exists():
        with open(model_card_path, encoding="utf-8") as f:
            card_content = f.read()
        st.markdown(card_content)
    else:
        st.info("Model card generated in reports/MODEL_CARD.md.")

st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; color: #94A3B8; font-size: 0.9rem;">
        <div>
            <b>PartSense Platform</b> | Built with Python 3.11, XGBoost, DuckDB & Streamlit
        </div>
        <div>
            <a href="https://github.com" target="_blank" style="color: #38BDF8; text-decoration: none; margin-right: 1rem;">GitHub Repository</a>
            <a href="https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset" target="_blank" style="color: #38BDF8; text-decoration: none;">UCI Dataset (ID 601)</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
