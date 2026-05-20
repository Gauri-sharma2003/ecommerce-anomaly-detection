-- ============================================================
-- MANUAL SQL QUERIES — E-Commerce Anomaly Detection Database
-- Author: Gauri Sharma
-- Purpose: Demonstrate SQL proficiency with analytical queries
-- ============================================================

-- ============================================================
-- 1. REVENUE ANALYSIS
-- ============================================================

-- Q1: Monthly revenue with month-over-month growth rate
SELECT 
    strftime('%Y-%m', order_date) AS month,
    ROUND(SUM(order_value), 2) AS revenue,
    COUNT(*) AS total_orders,
    ROUND(AVG(order_value), 2) AS avg_order_value,
    ROUND(
        (SUM(order_value) - LAG(SUM(order_value)) OVER (ORDER BY strftime('%Y-%m', order_date)))
        / LAG(SUM(order_value)) OVER (ORDER BY strftime('%Y-%m', order_date)) * 100, 1
    ) AS mom_growth_pct
FROM fact_orders
WHERE order_status NOT IN ('cancelled', 'unavailable')
GROUP BY month
ORDER BY month;


-- Q2: Revenue concentration — Top 20% of customers driving what % of revenue (Pareto)
WITH customer_revenue AS (
    SELECT 
        customer_id,
        SUM(order_value) AS total_spent,
        NTILE(5) OVER (ORDER BY SUM(order_value) DESC) AS quintile
    FROM fact_orders
    WHERE order_status = 'delivered'
    GROUP BY customer_id
)
SELECT 
    quintile,
    COUNT(*) AS customers,
    ROUND(SUM(total_spent), 0) AS segment_revenue,
    ROUND(SUM(total_spent) * 100.0 / (SELECT SUM(total_spent) FROM customer_revenue), 1) AS pct_of_total
FROM customer_revenue
GROUP BY quintile
ORDER BY quintile;


-- ============================================================
-- 2. ANOMALY DETECTION QUERIES
-- ============================================================

-- Q3: Detect anomaly days using Z-score > 2.5 standard deviations
SELECT 
    order_date,
    daily_revenue,
    order_count,
    z_score,
    CASE 
        WHEN z_score > 2.5 THEN 'SPIKE'
        WHEN z_score < -2.5 THEN 'DROP'
        ELSE 'NORMAL'
    END AS anomaly_type
FROM daily_kpis
WHERE ABS(z_score) > 2.5
ORDER BY ABS(z_score) DESC;


-- Q4: Rolling 7-day average vs actual — identify deviations
SELECT 
    order_date,
    daily_revenue,
    ROUND(AVG(daily_revenue) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2) AS rolling_7d_avg,
    ROUND(daily_revenue - AVG(daily_revenue) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2) AS deviation_from_avg
FROM daily_kpis
ORDER BY ABS(daily_revenue - AVG(daily_revenue) OVER (
    ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
)) DESC
LIMIT 15;


-- ============================================================
-- 3. MODEL PERFORMANCE ANALYSIS
-- ============================================================

-- Q5: Best model configuration per method (lowest deviation from 5% target)
WITH ranked AS (
    SELECT 
        model,
        method_type,
        feature_set,
        scaler,
        anomaly_pct,
        ABS(anomaly_pct - 5.0) AS deviation_from_target,
        ROW_NUMBER() OVER (PARTITION BY method_type ORDER BY ABS(anomaly_pct - 5.0)) AS rn
    FROM model_results
)
SELECT 
    method_type,
    model,
    feature_set,
    scaler,
    anomaly_pct,
    ROUND(deviation_from_target, 2) AS deviation_from_5pct_target
FROM ranked
WHERE rn = 1
ORDER BY deviation_from_target;


-- Q6: Feature set effectiveness — success rate by method and features
SELECT 
    method_type,
    feature_set,
    COUNT(*) AS configs_tested,
    SUM(CASE WHEN anomaly_pct BETWEEN 3 AND 10 THEN 1 ELSE 0 END) AS reasonable_configs,
    ROUND(
        SUM(CASE WHEN anomaly_pct BETWEEN 3 AND 10 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 1
    ) AS success_rate_pct
FROM model_results
GROUP BY method_type, feature_set
HAVING configs_tested > 5
ORDER BY success_rate_pct DESC
LIMIT 20;


-- ============================================================
-- 4. BUSINESS INTELLIGENCE QUERIES
-- ============================================================

-- Q7: Customer segmentation by order frequency and value (RFM-style)
SELECT 
    c.state,
    COUNT(DISTINCT f.customer_id) AS unique_customers,
    COUNT(*) AS total_orders,
    ROUND(AVG(f.order_value), 2) AS avg_order_value,
    ROUND(SUM(f.order_value), 0) AS total_revenue,
    ROUND(1.0 * COUNT(*) / COUNT(DISTINCT f.customer_id), 1) AS orders_per_customer
FROM fact_orders f
JOIN dim_customers c ON f.customer_id = c.customer_id
WHERE f.order_status = 'delivered'
GROUP BY c.state
ORDER BY total_revenue DESC;


-- Q8: Day-of-week ordering patterns
SELECT 
    CASE CAST(strftime('%w', order_date) AS INTEGER)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END AS day_of_week,
    COUNT(*) AS orders,
    ROUND(AVG(order_value), 2) AS avg_value,
    ROUND(SUM(order_value), 0) AS total_revenue
FROM fact_orders
WHERE order_status NOT IN ('cancelled', 'unavailable')
GROUP BY strftime('%w', order_date)
ORDER BY CAST(strftime('%w', order_date) AS INTEGER);


-- Q9: Cancellation analysis — which categories have highest cancel rates?
SELECT 
    p.category,
    COUNT(*) AS total_orders,
    SUM(CASE WHEN f.order_status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled,
    ROUND(
        SUM(CASE WHEN f.order_status = 'cancelled' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 1
    ) AS cancel_rate_pct
FROM fact_orders f
JOIN dim_products p ON f.product_id = p.product_id
GROUP BY p.category
HAVING total_orders > 10
ORDER BY cancel_rate_pct DESC;


-- Q10: Freight cost analysis — heavy products vs delivery cost correlation
SELECT 
    CASE 
        WHEN p.weight_g < 1000 THEN 'Light (<1kg)'
        WHEN p.weight_g < 5000 THEN 'Medium (1-5kg)'
        ELSE 'Heavy (>5kg)'
    END AS weight_category,
    COUNT(*) AS orders,
    ROUND(AVG(f.freight_value), 2) AS avg_freight,
    ROUND(AVG(f.order_value), 2) AS avg_order_value,
    ROUND(AVG(f.freight_value) / AVG(f.order_value) * 100, 1) AS freight_pct_of_order
FROM fact_orders f
JOIN dim_products p ON f.product_id = p.product_id
WHERE f.order_status = 'delivered'
GROUP BY weight_category
ORDER BY avg_freight DESC;
