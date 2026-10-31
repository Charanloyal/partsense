-- Top high-risk deployed components ordered by risk factors (operating hours, torque stress, temperature delta)
SELECT
    component_id,
    product_id,
    part_family,
    quality_type,
    region,
    operating_hours,
    torque_nm,
    rotational_speed_rpm,
    ROUND(process_temperature_k - air_temperature_k, 2) AS temp_diff_k,
    ROUND(torque_nm * (rotational_speed_rpm * 2.0 * 3.14159265 / 60.0), 1) AS power_w,
    machine_failure
FROM components_in_service
ORDER BY
    operating_hours DESC,
    torque_nm DESC
LIMIT 50;
