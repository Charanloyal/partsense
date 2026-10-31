# PartSense Data Architecture & Provenance

This directory stores the primary telemetric data and seeded synthetic aftersales tables powering **PartSense**.

---

## 1. Real Component Telemetry Dataset

- **Source**: [UCI Machine Learning Repository: AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Dataset ID**: `601`
- **Citation**: Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. Third International Conference on Artificial Intelligence for Industries (AI4I), 2020.
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **File Location**: [`data/raw/ai4i2020.csv`](raw/ai4i2020.csv) (10,000 rows, 14 columns)
- **Ingestion**: Ingested primarily via the Python `ucimlrepo` package (`fetch_ucirepo(id=601)`), with automated fallback to the official UCI static database mirror.

### Schema & Physical Definitions

| Column | Type | Unit | Description | Leakage Status |
|---|---|---|---|---|
| `UDI` | Integer | - | Unique record index [1..10000] | **FORBIDDEN (Leakage Rule)** |
| `Product ID` | String | - | Variant letter (L/M/H) + serial number | **FORBIDDEN (Leakage Rule)** |
| `Type` | Categorical | - | Quality variant: L (50%), M (30%), H (20%) | Used via One-Hot (`Type_L`, `Type_M`, `Type_H`) |
| `Air temperature [K]` | Float | Kelvin | Ambient shop / vehicle bay temperature | Used as raw feature |
| `Process temperature [K]` | Float | Kelvin | Cutting / contact interface temperature | Used as raw feature |
| `Rotational speed [rpm]` | Float | RPM | Spindle rotational velocity | Used as raw feature |
| `Torque [Nm]` | Float | Nm | Torque resistance on component | Used as raw feature |
| `Tool wear [min]` | Float | Minutes | Operating tool wear duration | Used as raw feature |
| `Machine failure` | Binary | 0 or 1 | **Primary Prediction Target** (3.39% rate) | **TARGET** |
| `TWF` | Binary | 0 or 1 | Tool Wear Failure mode | **FORBIDDEN in main model** |
| `HDF` | Binary | 0 or 1 | Heat Dissipation Failure mode | **FORBIDDEN in main model** |
| `PWF` | Binary | 0 or 1 | Power Failure mode | **FORBIDDEN in main model** |
| `OSF` | Binary | 0 or 1 | Overstrain Failure mode | **FORBIDDEN in main model** |
| `RNF` | Binary | 0 or 1 | Random Failure mode | **FORBIDDEN in main model** |

---

## 2. Seeded Synthetic Aftersales Layer

- **Generator Module**: [`src/partsense/synthetic.py`](../src/partsense/synthetic.py)
- **Random Seed**: `42` (Fixed, fully deterministic)
- **Storage**: [`data/synthetic/aftersales.duckdb`](synthetic/aftersales.duckdb) and Parquet tables in [`data/synthetic/`](synthetic/)
- **Labeling**: **100% SYNTHETIC DATA**

### Table Catalog

1. **`parts_catalog`**:
   - 6 automotive spare-part families:
     * **Wheel Bearing**: Unit cost $85, Warranty 24 months / 24,000 miles
     * **Water Pump**: Unit cost $140, Warranty 36 months / 36,000 miles
     * **Alternator**: Unit cost $280, Warranty 24 months / 24,000 miles
     * **Starter Motor**: Unit cost $210, Warranty 24 months / 24,000 miles
     * **Brake Caliper**: Unit cost $175, Warranty 12 months / 12,000 miles
     * **Fuel Pump**: Unit cost $320, Warranty 36 months / 36,000 miles

2. **`components_in_service`**:
   - Maps each of the 10,000 AI4I telemetric records to a part family, installation date (24-month timeline: Jan 2024 – Dec 2025), and 1 of 4 geographic regions:
     * `North America`
     * `Europe`
     * `Asia-Pacific`
     * `Latin America`
   - **Operating Hours Assumption**: As specified, component operating hours are computed as:
     $$\text{Operating Hours} = \frac{\text{Tool Wear [min]}}{60.0}$$

3. **`parts_sales`**:
   - Monthly replacement and OEM units sold per part family per region over 24 months.

4. **`warranty_claims` & `service_records`**:
   - Generated strictly from true component failures (`Machine failure == 1`).
   - Claim cost is drawn from a seeded lognormal distribution parameterized by the physical failure mode (e.g. electrical PWF alternator failures carry higher claim costs than basic TWF mechanical adjustments).
   - Includes realistic claim lag (7 to 45 days from failure event to claim reimbursement).
