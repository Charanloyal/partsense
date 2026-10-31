-- Monthly claim trends: claim volume, total claim cost, and average claim reimbursement
SELECT
    claim_month,
    COUNT(*) AS total_claims,
    ROUND(SUM(claim_amount), 2) AS total_claim_amount,
    ROUND(AVG(claim_amount), 2) AS avg_claim_amount,
    ROUND(AVG(claim_lag_days), 1) AS avg_claim_lag_days,
    SUM(is_approved_warranty) AS approved_warranty_claims
FROM warranty_claims
GROUP BY claim_month
ORDER BY claim_month ASC;
