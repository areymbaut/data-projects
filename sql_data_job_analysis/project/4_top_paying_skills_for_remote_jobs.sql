/*
What the the top skills based on salary for remote data-analyst jobs?
*/
SELECT
    skills_dim.skill_id,
    skills_dim.skills AS skill,
    ROUND(AVG(job_postings_fact.salary_year_avg), 0) AS average_salary
FROM 
    job_postings_fact
INNER JOIN skills_job_dim ON skills_job_dim.job_id = job_postings_fact.job_id
INNER JOIN skills_dim ON skills_dim.skill_id = skills_job_dim.skill_id
WHERE
    job_postings_fact.job_title_short = 'Data Analyst'
    AND job_postings_fact.salary_year_avg IS NOT NULL
    AND job_postings_fact.job_work_from_home = true
GROUP BY
    skills_dim.skill_id
ORDER BY
    average_salary DESC
LIMIT 25;

/*
Top 3 skills based on salary:
1. PySpark (big data technology).
2. Bitbucket (Git solution compatible with Jira).
3. Watson (Cloud app for AI deployment) ex-aequo with Couchbase (NoSQL server).

Other top skills notably include famous Python modules
(e.g., numpy, pandas, scikit-learn).
*/