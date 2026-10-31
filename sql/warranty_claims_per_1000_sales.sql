-- Warranty claims per 1,000 units sold (Automotive IPTV style: Incidents Per Thousand Vehicles/Units)
WITH sales_summary AS (
    SELECT
        part_family,
        SUM(units_sold) AS total_units_sold,
        ROUND(SUM(revenue), 2) AS total_sales_revenue
    FROM parts_sales
    GROUP BY part_family
),
claims_summary AS (
    SELECT
        part_family,
        COUNT(*) AS total_claims,
        ROUND(SUM(claim_amount), 2) AS total_claim_cost
    FROM warranty_claims
    GROUP BY part_family
)
SELECT
    s.part_family,
    s.total_units_sold,
    COALESCE(c.total_claims, 0) AS total_claims,
    ROUND((COALESCE(c.total_claims, 0) * 1000.0) / s.total_units_sold, 2) AS claims_per_1000_units,
    COALESCE(c.total_claim_cost, 0.0) AS total_claim_cost
FROM sales_summary s
LEFT JOIN claims_summary c ON s.part_family = c.part_family
ORDER BY claims_per_1000_units DESC;
