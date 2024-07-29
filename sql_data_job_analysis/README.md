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

To identify the highest-paying roles, data-analyst positions were filtered by average yearly salary and location, focusing on remote jobs.

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

| Job ID | Job title | Company | Average yearly salary ($) |
|-|-|-|:-:|
| 226942 | Data Analyst | Mantys | 650000.0 |
| 547832 | Director of Analytics | Meta | 336500.0 |
| 552322 | Associate Director- Data Insights | AT&T | 255829.5 |
| 99305 | Data Analyst, Marketing | Pinterest Job Advertisements | 232423.0 |
| 1021647 | Data Analyst (Hybrid/Remote) | Uclahealthcareers | 217000.0 |
| 168310 | Principal Data Analyst (Remote) | SmartAsset | 205000.0 |
| 731368 | Director, Data Analyst - HYBRID | Inclusively | 189309.0 |
| 310660 | Principal Data Analyst, AV Performance Analysis | Motional | 189000.0 |
| 1749593 | Principal Data Analyst | SmartAsset | 189000.0 |
| 387860 | ERM Data Analyst | Get It Recruit - Information Technology | 189000.0 |

*Table output by the query.*

This analysis indicated that the top data-analyst jobs of 2023 were offered by diverse employers (_e.g._, Meta, AT&T, SmartAsset), with a wide salary range (k$184-k$650) and job-title variety.

### Skills for top-paying remote jobs

To understand what skills are required for the top-paying jobs, job postings were joined with the skills data, providing insights into what employers value for high-compensation roles.

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

To summarize the information, a demand count was computed per skill using the following query.

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
    skills_dim.skill_id,
    skills_dim.skills AS skill,
    COUNT(skills_dim.skills) AS demand_count
FROM 
    top_paying_jobs
INNER JOIN skills_job_dim ON skills_job_dim.job_id = top_paying_jobs.job_id
INNER JOIN skills_dim ON skills_dim.skill_id = skills_job_dim.skill_id
GROUP BY
    skills_dim.skill_id
ORDER BY
    demand_count DESC;
```

| Skill ID | Skill | Demand count |
|:-:|:-:|:-:|
| 0 | sql | 8 |
| 1 | python | 7 |
| 182 | tableau | 6 |
| 5 | r | 4 |
| 93 | pandas | 3 |
| ... | ... | ... |
| ... | ... | ... |

*Table showing the first five rows of the table output by the query.*

**SQL**, **Python** and **Tableau** are the most demanded skills for top-paying remote data-analyst jobs.

### In-demand skills for data analysts

This query helped identify the skills most frequently requested in job postings, directing focus to areas with high demand.

```sql
SELECT
    skills_dim.skill_id AS skill_id,
    skills_dim.skills AS skill,
    COUNT(skills_job_dim.job_id) AS demand_count
FROM 
    job_postings_fact
INNER JOIN skills_job_dim ON skills_job_dim.job_id = job_postings_fact.job_id
INNER JOIN skills_dim ON skills_dim.skill_id = skills_job_dim.skill_id
WHERE
    job_postings_fact.job_title_short = 'Data Analyst'
GROUP BY
    skills_dim.skill_id
ORDER BY
    demand_count DESC
LIMIT 5;
```

| Skill ID | Skill | Demand count |
|:-:|:-:|:-:|
| 0 | sql | 92628 |
| 181 | excel | 67031 |
| 1 | python | 57326 |
| 182 | tableau | 46554 |
| 183 | power bi | 39468 |

*Table output by the query.*

The analysis shows that the skills above are most frequently requested. This emphasizes:
- the need for strong foundational skills in **data processing and spreadsheet manipulation**.
- the increasing importance of technical skills in **data storytelling and decision support**.

### Top-paying skills for remote jobs

Exploring the average salaries associated with different skills revealed which skills are the highest paying.

```sql
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
```

| Skill ID | Skill | Average yearly salary ($) |
|:-:|:-:|:-:|
| 95 | pyspark | 208172 |
| 218 | bitbucket | 189155 |
| 85 | watson | 160515 |
| 65 | couchbase | 160515 |
| 206 | datarobot | 155486 |
| ... | ... | ... |
| ... | ... | ... |

*Table showing the first five rows of the table output by the query.*

Here are the top 3 skills based on salary:
1. PySpark (big data technology).
2. Bitbucket (Git solution compatible with Jira).
3. Watson (Cloud app for AI deployment) _ex-aequo_ with Couchbase (NoSQL server).

Other top-paying skills notably include famous Python modules such as numpy, pandas, scikit-learn.

### Most optimal skills to learn

Combining insights from demand and salary data, this query aimed to pinpoint skills that are both in high demand and have high salaries, offering a strategic focus for skill development.

```sql
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
```

| Skill ID | Skill | Demand count | Average yearly salary ($) |
|:-:|:-:|:-:|:-:|
| 75 | databricks | 10 | 141907 |
| 8 | go | 27 | 115320 |
| 234 | confluence | 11 | 114210 |
| 97 | hadoop | 22 | 113193 |
| 80 | snowflake | 37 | 112948 |
| ... | ... | ... | ... |
| ... | ... | ... | ... |

*Table showing the first five rows of the table output by the query.*

Here is a breakdown of the most optimal skills (based on the 20 rows output by the above query):
- skills related to **cloud/big-data tools** (_e.g._, Snowflake, Azure, AWS, BigQuery) show significant demand with relatively high average salaries, pointing towards the growing importance of cloud platforms and big data technologies in data analysis.
- **more classical skills** (_e.g._, C++, Python, R, SQL, Tableau) remain in high demand, yet with more average salaries, indicating that proficiency in these skills is highly valued but also widely available.