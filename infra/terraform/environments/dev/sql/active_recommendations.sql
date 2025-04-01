-- Active recommendations view
SELECT 
  r.*,
  DATE_DIFF(CURRENT_TIMESTAMP(), r.creation_date, DAY) AS days_since_creation
FROM 
  `${project_id}.bqcostopt.recommendations` r
LEFT JOIN 
  `${project_id}.bqcostopt.implementation_history` i
ON 
  r.recommendation_id = i.recommendation_id
WHERE 
  r.status = 'active'
  AND i.implementation_id IS NULL
ORDER BY 
  r.priority_score DESC, r.estimated_savings.monthly DESC