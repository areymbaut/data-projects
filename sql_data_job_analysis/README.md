# SQL project - Job market analysis

## Introduction
This project focuses on (remote) data-analyst jobs, identifying top-paying jobs and in-demand skills. Associated SQL queries are located here: [project folder](/sql_data_job_analysis/project/).

The dataset was retrieved from [Luke Barousse's SQL course](https://lukebarousse.com/sql). It contains insights on job titles, salaries, locations and essential skills for various data-analytics job postings from 2023.

## Answered questions
1. What are the top-paying data-analyst jobs?
2. What skills are required for these top-paying jobs?
3. What skills are most in demand for data analysts?
4. Which skills are associated with higher salaries?
5. What are the most optimal skills to learn (_i.e._, top-paying and in-demand)?

## Tools I used
This project was carried out using the following tools:
- **SQL** - backbone of the analysis, allowing for database queries.
- **PostgreSQL** - chosen database management system.
- **Visual Studio Code** - my go-to for database management and executing SQL queries.
- **Git/Github** - essential for version control and code sharing.

## Analysis 

### Top-paying remote data-analyst jobs

To identify the highest-paying roles, I filtered data-analyst positions by average yearly salary and location, focusing on remote jobs.

```sql
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
```

This analysis indicated that the top data-analyst jobs of 2023 were offered by diverse employers (_e.g._, Meta, AT&T, SmartAsset), with a wide salary range ($184,000-$650,000) and job-title variety.

### Skills for top-paying remote jobs

To understand what skills are required for the top-paying jobs, I joined the job postings with the skills data, providing insights into what employers value for high-compensation roles.

```sql
WITH top_paying_jobs AS (
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
    LIMIT 10
)

SELECT 
    top_paying_jobs.*,
    skills_dim.skills
FROM 
    top_paying_jobs
INNER JOIN skills_job_dim ON skills_job_dim.job_id = top_paying_jobs.job_id
INNER JOIN skills_dim ON skills_dim.skill_id = skills_job_dim.skill_id
ORDER BY
    top_paying_jobs.salary_year_avg DESC;
```

A demand count was also computed per skill using SQL, indicating that **SQL**, **Python** and **Tableau** are the most demanded skills for top-paying remote data-analyst jobs.

### In-demand skills for data analysts

This query helped identify the skills most frequently requested in job postings, directing focus to areas with high demand.

```sql
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
```

The analysis shows that these five skills are most frequently requested:
1. SQL. 
2. Excel.
3. Python.
4. Tableau.
5. Power BI.

On the one hand, this emphasizes the need for strong foundational skills in **data processing and spreadsheet manipulation**. On the other hand, this denotes the increasing importance of technical skills in **data storytelling and decision support**.

### Top-paying skills for remote jobs

Exploring the average salaries associated with different skills revealed which skills are the highest paying.

```sql
SELECT
    skills_dim.skill_id,
    skills_dim.skills AS skills,
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
```

Here are the top 3 skills based on salary:
1. PySpark (big data technology).
2. Bitbucket (Git solution compatible with Jira).
3. Watson (Cloud app for AI deployment) ex-aequo with Couchbase (NoSQL server).

Other top-paying skills notably include famous Python modules
such as numpy, pandas, scikit-learn.

### Most optimal skills to learn

Combining insights from demand and salary data, this query aimed to pinpoint skills that are both in high demand and have high salaries, offering a strategic focus for skill development.

```sql
SELECT
    skills_dim.skill_id,
    skills_dim.skills,
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
LIMIT 25;
```

Here is a breakdown of the most optimal skills:
- skills related to cloud/big-data tools (_e.g._, Snowflake, Azure, AWS, BigQuery) show significant demand with relatively high average salaries, pointing towards the growing importance of cloud platforms and big data technologies in data analysis.
- more classical skills (_e.g._, C++, Python, R, SQL, Tableau) remain in high demand, yet with more average salaries, indicating that proficiency in these skills is highly valued but also widely available.