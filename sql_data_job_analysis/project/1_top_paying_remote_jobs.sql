/*
What are the 10 top-paying remote data-analyst jobs?
*/
SELECT 
    job_postings_fact.job_id,
    job_postings_fact.job_title,
    company_dim.name AS company_name,
    job_postings_fact.salary_year_avg
FROM 
    job_postings_fact
LEFT JOIN company_dim ON job_postings_fact.company_id = company_dim.company_id
WHERE
    job_postings_fact.job_title_short = 'Data Analyst'
    AND job_postings_fact.job_work_from_home = true
    AND job_postings_fact.salary_year_avg IS NOT NULL
ORDER BY
    job_postings_fact.salary_year_avg DESC
LIMIT 10;


/*
Categorization of data-analyst job postings, with average yearly salaries being
considered low under 100k, standard between 100k and 200k, high otherwise,
ordered from highest salary to lowest salary.
*/
SELECT
    job_postings_fact.job_id,
    job_postings_fact.job_title_short,
    company_dim.name AS company_name,
    job_postings_fact.salary_year_avg,
    CASE 
        WHEN job_postings_fact.salary_year_avg < 100000 THEN 'Low'
        WHEN job_postings_fact.salary_year_avg BETWEEN 100000 AND 200000 THEN 'Standard'
        ELSE 'High'
    END AS salary_bucket
FROM
    job_postings_fact
LEFT JOIN company_dim ON company_dim.company_id = job_postings_fact.company_id
WHERE
    job_postings_fact.job_title_short = 'Data Analyst'
    AND job_postings_fact.job_work_from_home = true
    AND salary_year_avg IS NOT NULL
ORDER BY
    job_postings_fact.salary_year_avg DESC;