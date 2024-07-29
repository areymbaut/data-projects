# SQL project - Worldwide layoff analysis

## Introduction
This project aims to extract insights regarding layoffs around the world between March 2020 and March 2023. Associated SQL queries are located here: [project folder](/sql_world_layoffs/project/). The dataset, retrieved from [Alex Freberg's Github repository](https://github.com/AlexTheAnalyst/MySQL-YouTube-Series/blob/main/layoffs.csv), can be loaded into a PostgreSQL database using the scripts located here: [sql_load folder](/sql_world_layoffs/sql_load/).

## Answered questions
1. When did massive layoffs occur during the covered time period? 
2. Were all industries affected similarly by these layoffs?
3. Did layoffs occur in specific geographical locations?
4. Which companies went through the largest layoffs?
5. Which industries went through the largest layoffs?

## Tools I used
This project was carried out using the following tools:
- **SQL** - backbone of the analysis, allowing for database queries.
- **PostgreSQL** - chosen database management system.
- **Python/Pandas/Matplotlib** - loading and visualization of analysis outputs.
- **Visual Studio Code** - my go-to for database management and executing SQL queries.
- **Git/Github** - essential for version control and code sharing.

## Data cleaning

The original dataset was cleaned using the queries found in [1_data_cleaning.sql](/sql_world_layoffs/project/1_data_cleaning.sql). The data cleaning consisted in:
1. creating a copy of the original dataset.
2. removing duplicates.
3. standardizing entries.
4. converting data types when relevant.
5. imputing certain `NULL` values (when it made sense to do so).
6. removing the entries that do not contain any layoff-related information.

## Exploratory data analysis

Exploratory data analysis was performed using the queries found in [2_exploratory_data_analysis.sql](/sql_world_layoffs/project/2_exploratory_data_analysis.sql).

### Temporal evolution of the layoffs

The total number of employees laid off was computed across the entire dataset and grouped by month/year. This allowed for the calculation of its rolling sum across the covered time period.

```sql
WITH month_cte AS (
    SELECT 
        SUBSTRING(date::text, 1, 7) AS date_month,
        SUM(total_laid_off) AS sum_total_laid_off
    FROM layoffs_staging
    WHERE date IS NOT NULL
    GROUP BY date_month
    ORDER BY 1 ASC
)
SELECT 
    date_month,
    sum_total_laid_off,
    SUM(sum_total_laid_off) OVER(ORDER BY date_month) AS rolling_sum_total_laid_off
FROM month_cte
```

<img src="https://github.com/user-attachments/assets/891af829-3524-419b-b366-88be2154ee34" width="700">

*Total number of employees laid off and its rolling sum over the covered time period (graph made using Matplotlib).*

This analysis indicated that:
- 2020 was marked by rather big layoffs (probably due to the covid pandemic). 
- 2021 was relatively quiet in terms of layoff.
- the second half of 2022 and the first quarter of 2023 were marked by massive layoffs (probably due to financial market instabilities following the Russia/Ukraine conflict that started in the first quarter of 2022).

### Global layoffs per industry

Layoff-related information was grouped by industry across the covered time period:
- the total number of employees laid off was used as an indicator of the **absolute size of the layoff**.
- the average of the percentage of the workforce that was laid off was used as a proxy for the **relative size of the layoff**.

An ad-hoc threshold of 20k employees laid off was used to truncate the query.

```sql
SELECT 
    industry,
    SUM(total_laid_off) AS sum_total_laid_off,
    ROUND(AVG(percentage_laid_off), 2) AS avg_percentage_laid_off
FROM layoffs_staging
WHERE total_laid_off IS NOT NULL
GROUP BY industry
HAVING SUM(total_laid_off) >= 20000
ORDER BY sum_total_laid_off DESC;
```

| Industry | Absolute layoff size | Relative layoff size (%) |
|-|:-:|:-:|
| Consumer | 45182 | 24 |
| Retail | 43613 | 22 |
| Other | 36289 | 21 |
| Transportation | 33748 | 21 |
| Finance | 28344 | 20 |
| Healthcare | 25953 | 25 |
| Food | 22855 | 30 |

*Table output by the query.*

Most layoffs occured in the **consumer** and **retail** industries. While the total number of employees laid off varies across industry sectors, the average percentage of laid-off workforce is rather similar across them (~20%). This may hint at common layoff causes across these industries.

### Geographical localization of the layoffs

#### Globally

Layoff-related information was grouped by country across the covered time period:
- the total number of employees laid off was used as an indicator of the **absolute size of the layoff**.
- the average of the percentage of the workforce that was laid off was used as a proxy for the **relative size of the layoff**.

An ad-hoc threshold of 10k employees laid off was used to truncate the query.

```sql
SELECT
    country,
    SUM(total_laid_off) AS sum_total_laid_off,
    ROUND(AVG(percentage_laid_off), 2) AS avg_percentage_laid_off
FROM layoffs_staging
WHERE total_laid_off IS NOT NULL
GROUP BY country
HAVING SUM(total_laid_off) >= 10000
ORDER BY sum_total_laid_off DESC;
```

| Country | Absolute layoff size | Relative layoff size (%) |
|-|:-:|:-:|
| United States | 256559 | 22 |
| India | 35993 | 25 |
| Netherlands | 17220 | 14 |
| Sweden | 11264 | 13 |
| Brazil | 10391 | 20 |

*Table output by the query.*

Most layoffs occured in the **US**, **India**, **the Netherlands**, **Sweden** and **Brazil**. Layoffs were of the order of ~15% of average laid-off workforce in the Netherlands and Sweden, and ~20-25% of average laid-off workforce in the US, India and Brazil.

#### In Europe

The query below focuses on Europe, to better understand from which companies the layoffs originated in the Netherlands and Sweden. An ad-hoc threshold of 2000 employees laid off was used to truncate the query.

```sql
SELECT
    country,
    company,
    total_laid_off,
    date
FROM layoffs_staging
WHERE 
    country IN ('Sweden', 'Netherlands')
    AND total_laid_off IS NOT NULL
    AND total_laid_off >= 2000
ORDER BY 
    country,
    total_laid_off DESC;
```

| Country | Company | Absolute layoff size | Date |
|-|-|:-:|:-:|
| Netherlands | Philips | 6000 | 2023-01-30 |
| Netherlands | Booking.com | 4375 | 2020-07-30 |
| Netherlands | Philips | 4000 | 2022-10-24 |
| Sweden | Ericsson | 8500 | 2023-02-24 |

*Table output by the query.*

It appears that Dutch layoffs were mainly imputable to **Booking.com** (2020) and **Philips** (2022, 2023), and that Swedish layoffs were mainly imputable to **Ericsson** (2023).

### Companies - Largest layoffs per year

Companies were ranked per year according to absolute layoff size: 
- the total number of employees laid off was used as an indicator of the **absolute size of the layoff**.
- the average of the percentage of the workforce that was laid off was used as a proxy for the **relative size of the layoff**.

The query below shows the top 3 companies per year.

```sql
WITH yearly_layoffs AS 
(
    SELECT 
        company,
        EXTRACT(YEAR FROM date) AS layoff_year,
        SUM(total_laid_off) AS sum_total_laid_off,
        ROUND(AVG(percentage_laid_off), 2) AS avg_percentage_laid_off
    FROM layoffs_staging
    WHERE
        total_laid_off IS NOT NULL
        AND EXTRACT(YEAR FROM date) IS NOT NULL
    GROUP BY 
        company,
        EXTRACT(YEAR FROM date)
),
yearly_layoffs_rank AS (
    SELECT 
        company,
        layoff_year,
        sum_total_laid_off,
        avg_percentage_laid_off,
        DENSE_RANK() OVER (PARTITION BY layoff_year ORDER BY sum_total_laid_off DESC) AS ranking
    FROM yearly_layoffs
)
SELECT 
    company,
    layoff_year,
    ranking,
    sum_total_laid_off,
    avg_percentage_laid_off
FROM yearly_layoffs_rank
WHERE 
    ranking <= 3
ORDER BY 
    layoff_year ASC,
    sum_total_laid_off DESC;
```

| Company | Year | Ranking | Absolute layoff size | Relative layoff size (%) |
|-|:-:|:-:|:-:|:-:|
| Uber | 2020 | 1 | 7525 | 19 |
| Booking.com | 2020 | 2 | 4375 | 25 |
| Groupon | 2020 | 3 | 2800 | 44 |
| Bytedance | 2021 | 1 | 3600 | `NULL` |
| Katerra | 2021 | 2 | 2434 | 100 |
| Zillow | 2021 | 3 | 2000 | 25 |
| Meta | 2022 | 1 | 11000 | 13 |
| Amazon | 2022 | 2 | 10150 | 3 |
| Cisco | 2022 | 3 | 4100 | 5 |
| Google | 2023 | 1 | 12000 | 6 |
| Microsoft | 2023 | 2 | 10000 | 5 |
| Ericsson | 2023 | 3 | 8500 | 8 |

*Table output by the query.*

As aforementioned, layoffs mainly occurred during 2020, 2022 and 2023, with 2020 layoffs probably due to the covid pandemic and 2022-2023 layoffs probably due to financial market instabilities following the recent Russia/Ukraine conflict. Assuming that these interpretations are accurate:
- the covid pandemic affected rather **mid-size companies** (large absolute and relative layoffs).
- financial market instabilities even affected **tech giants** (small relative layoffs despite large absolute layoffs).

Moreover, one can notice Katerra's **100% relative layoff** in 2021, corresponding to its liquidation that year.

### Industries - Largest layoffs per year

Industries were ranked per year according to absolute layoff size: 
- the total number of employees laid off was used as an indicator of the **absolute size of the layoff**.
- the average of the percentage of the workforce that was laid off was used as a proxy for the **relative size of the layoff**.

The query below shows the top 3 industries per year.

```sql
WITH yearly_layoffs AS 
(
    SELECT 
        industry,
        EXTRACT(YEAR FROM date) AS layoff_year,
        SUM(total_laid_off) AS sum_total_laid_off,
        ROUND(AVG(percentage_laid_off), 2) AS avg_percentage_laid_off
    FROM layoffs_staging
    WHERE
        total_laid_off IS NOT NULL
        AND EXTRACT(YEAR FROM date) IS NOT NULL
    GROUP BY 
        industry,
        EXTRACT(YEAR FROM date)
),
yearly_layoffs_rank AS (
    SELECT 
        industry,
        layoff_year,
        sum_total_laid_off,
        avg_percentage_laid_off,
        DENSE_RANK() OVER (PARTITION BY layoff_year ORDER BY sum_total_laid_off DESC) AS ranking
    FROM yearly_layoffs
)
SELECT 
    industry,
    layoff_year,
    ranking,
    sum_total_laid_off,
    avg_percentage_laid_off
FROM yearly_layoffs_rank
WHERE 
    ranking <= 3
ORDER BY 
    layoff_year ASC,
    sum_total_laid_off DESC;
```

| Industry | Year | Ranking | Absolute layoff size | Relative layoff size (%) |
|-|:-:|:-:|:-:|:-:|
| Transportation | 2020 | 1 | 14656 | 23 |
| Travel | 2020 | 2 | 13983 | 27 |
| Finance | 2020 | 3 | 8624 | 24 |
| Consumer | 2021 | 1 | 3600 | `NULL` |
| Real Estate | 2021 | 2 | 2900 | 17 |
| Food | 2021 | 3 | 2644 | 18 |
| Retail | 2022 | 1 | 20914 | 19 |
| Consumer | 2022 | 2 | 19856 | 20 |
| Transportation | 2022 | 3 | 15227 | 16 |
| Other | 2023 | 1 | 28512 | 16 |
| Consumer | 2023 | 2 | 15663 | 18 |
| Retail | 2023 | 3 | 13609 | 10 |

*Table output by the query.*

We had previously identified the retail and consumer industries as having suffered the largest layoffs, followed by `Other` and the transportation industry. While the retail, consumer and transportation industries appear multiple times in the above table, `Other` only appears once in 2023. What contributed to that?

This query focuses on the `Other` industry in 2023, using an ad-hoc threshold of 5000 employees laid off to truncate it.

 ```sql
SELECT *
FROM layoffs_staging
WHERE
    EXTRACT(YEAR FROM date) = 2023
    AND industry = 'Other'
    AND total_laid_off IS NOT NULL
    AND total_laid_off > 5000
ORDER BY
    total_laid_off DESC;
 ```

| Company | Absolute layoff size | Relative layoff size (%) |
|-|:-:|:-:|
| Microsoft | 10000 | 5 |
| Ericsson | 8500 | 8 |

*Table output by the query.*

It appears that tech giants such as Microsoft and Ericsson are categorized as part of the `Other` industry in the original dataset. This shows that it would probably be relevant to enhance the original dataset and re-categorize major companies to extract more refined insights.
