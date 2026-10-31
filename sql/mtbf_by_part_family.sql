-- Mean Time Between Failures (MTBF in operating hours) by part family
-- MTBF = Total Operating Hours / Total Failures
SELECT
    part_family,
    COUNT(*) AS total_deployed,
    SUM(machine_failure) AS failure_count,
    ROUND(SUM(operating_hours), 1) AS total_operating_hours,
    CASE
        WHEN SUM(machine_failure) > 0 THEN
            ROUND(SUM(operating_hours) / SUM(machine_failure), 1)
        ELSE NULL
    END AS mtbf_hours
FROM components_in_service
GROUP BY part_family
ORDER BY mtbf_hours ASC;
