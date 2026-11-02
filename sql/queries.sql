-- Q1: WHERE then HAVING
-- Keep only completed orders with order_amount > 1000, group by customer_id, and keep customers whose total order value exceeds INR 100,000.
SELECT customer_id, SUM(order_amount) AS total_large_spend
FROM orders
WHERE status = 'completed' AND order_amount > 1000
GROUP BY customer_id
HAVING SUM(order_amount) > 100000;

-- Q2: Three-table JOIN
-- Join customers, orders, and products to find city-category total sales.
SELECT c.city, p.category, SUM(o.order_amount) AS total_sales
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category;

-- Q3: Scalar Subquery
-- Find products priced strictly above the average catalogue price.
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: LEFT JOIN
-- Find customers who have never placed any completed order.
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5: Common Table Expressions (CTEs)
-- Find customers whose total completed spend is above the overall average customer spend.
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total FROM customer_spend
)
SELECT cs.customer_id, cs.total_spend
FROM customer_spend cs, avg_spend as
WHERE cs.total_spend > as.avg_total;

-- Q6: Ranking Window Functions
-- For each customer with completed spend, compute running totals and row numbers.
SELECT 
    customer_id,
    order_id,
    order_ts,
    order_amount,
    SUM(order_amount) OVER (PARTITION BY customer_id ORDER BY order_ts DESC) AS running_sum,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_ts DESC) AS row_num
FROM orders
WHERE status = 'completed';