-- Savings by project view
SELECT
  project_id,
  COUNT(recommendation_id) AS recommendation_count,
  SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_recommendations,
  SUM(CASE WHEN status = 'implemented' THEN 1 ELSE 0 END) AS implemented_recommendations,
  SUM(estimated_savings.monthly) AS total_estimated_monthly_savings,
  SUM(CASE WHEN status = 'implemented' THEN estimated_savings.monthly ELSE 0 END) AS implemented_monthly_savings,
  SUM(CASE WHEN status = 'active' THEN estimated_savings.monthly ELSE 0 END) AS potential_monthly_savings,
  ARRAY_AGG(STRUCT(recommendation_type, COUNT(recommendation_id) AS count, SUM(estimated_savings.monthly) AS monthly_savings) 
    GROUP BY recommendation_type ORDER BY SUM(estimated_savings.monthly) DESC) AS savings_by_type
FROM
  `${project_id}.bqcostopt.recommendations`
GROUP BY
  project_id
ORDER BY
  total_estimated_monthly_savings DESC