/*
What are the most optimal skills to learn (i.e., high paying
while being in relatively high demand too) for remote data-analyst jobs?
*/
SELECT
    skills_dim.skill_id,
    skills_dim.skills AS skill,
    COUNT(skills_job_dim.job_id) AS demand_count,
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
HAVING
    COUNT(skills_job_dim.job_id) >= 10  -- To take out niche skills
ORDER BY
    average_salary DESC,
    demand_count DESC
LIMIT 20;

/*
Skills related to cloud development and AI deployment are quite sought for,
with more classical skills (e.g., C++, Python, R, SQL, Tableau) following them.
*/