-- Pareto analysis (80/20 rule) of failure modes ranked by cumulative claim cost
WITH mode_summary AS (
    SELECT
        primary_failure_mode,
        COUNT(*) AS claim_count,
        ROUND(SUM(claim_amount), 2) AS total_claim_cost,
        ROUND(AVG(claim_amount), 2) AS avg_claim_cost
    FROM warranty_claims
    GROUP BY primary_failure_mode
),
total_cost_calc AS (
    SELECT SUM(total_claim_cost) AS grand_total_cost FROM mode_summary
)
SELECT
    m.primary_failure_mode,
    m.claim_count,
    m.total_claim_cost,
    m.avg_claim_cost,
    ROUND((m.total_claim_cost / t.grand_total_cost) * 100.0, 2) AS pct_of_total_cost,
    ROUND(
        SUM(m.total_claim_cost) OVER (
            ORDER BY m.total_claim_cost DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) * 100.0 / t.grand_total_cost,
        2
    ) AS cumulative_pct
FROM mode_summary m
CROSS JOIN total_cost_calc t
ORDER BY m.total_claim_cost DESC;
