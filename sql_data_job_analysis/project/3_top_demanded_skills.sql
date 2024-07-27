/*
What are the 5 most demanded skills across all data-analyst jobs?
*/
SELECT
    skills_dim.skills AS most_demanded_skills,
    COUNT(skills_job_dim.job_id) AS skill_count
FROM 
    job_postings_fact
INNER JOIN skills_job_dim ON skills_job_dim.job_id = job_postings_fact.job_id
INNER JOIN skills_dim ON skills_dim.skill_id = skills_job_dim.skill_id
WHERE
    job_postings_fact.job_title_short = 'Data Analyst'
GROUP BY
    skills_dim.skills
ORDER BY
    skill_count DESC
LIMIT 5;

/*
SQL, Excel, Python, Tableau and Power BI (in that order)
are the most demanded skills across data-analyst jobs.
*/