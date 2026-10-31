-- Warranty cost per unit sold (financial warranty exposure ratio)
WITH sales_agg AS (
    SELECT
        part_family,
        SUM(units_sold) AS total_units_sold,
        SUM(revenue) AS total_revenue
    FROM parts_sales
    GROUP BY part_family
),
claims_agg AS (
    SELECT
        part_family,
        SUM(claim_amount) AS total_warranty_cost
    FROM warranty_claims
    GROUP BY part_family
)
SELECT
    s.part_family,
    s.total_units_sold,
    ROUND(COALESCE(c.total_warranty_cost, 0.0), 2) AS total_warranty_cost,
    ROUND(COALESCE(c.total_warranty_cost, 0.0) / s.total_units_sold, 2) AS warranty_cost_per_unit_sold,
    ROUND((COALESCE(c.total_warranty_cost, 0.0) / s.total_revenue) * 100.0, 2) AS warranty_pct_of_revenue
FROM sales_agg s
LEFT JOIN claims_agg c ON s.part_family = c.part_family
ORDER BY warranty_cost_per_unit_sold DESC;
