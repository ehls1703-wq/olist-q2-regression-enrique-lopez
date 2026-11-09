-- Q2: One row per order
SELECT
    i.order_id,
    SUM(i.freight_value) AS freight_value,
    SUM(p.product_weight_g) / 1000.0 AS weight_kg,
    MAX(p.product_length_cm) AS product_length_cm,
    MAX(p.product_height_cm) AS product_height_cm,
    MAX(p.product_width_cm) AS product_width_cm,
    c.customer_state
FROM items i
JOIN products p ON i.product_id = p.product_id
JOIN orders o ON i.order_id = o.order_id
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY i.order_id, c.customer_state;
