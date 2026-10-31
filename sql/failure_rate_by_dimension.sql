-- Failure rate percentage by part family, quality type (L/M/H), and geographic region
SELECT
    part_family,
    quality_type,
    region,
    COUNT(*) AS total_components,
    SUM(machine_failure) AS failure_count,
    ROUND(SUM(machine_failure) * 100.0 / COUNT(*), 2) AS failure_rate_pct
FROM components_in_service
GROUP BY part_family, quality_type, region
ORDER BY failure_rate_pct DESC, total_components DESC;
