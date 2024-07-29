/*
General information
*/
SELECT 
    MIN(date) AS earliest_date,
    MAX(date) AS latest_date,
    ROUND(AVG(total_laid_off), 2) AS avg_total_laid_off,
    ROUND(STDDEV_SAMP(total_laid_off), 2) AS std_total_laid_off,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_laid_off) AS median_total_laid_off,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_laid_off) - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_laid_off) AS iqr_total_laid_off,
    ROUND(AVG(percentage_laid_off), 2) AS avg_percentage_laid_off,
    ROUND(STDDEV_SAMP(percentage_laid_off), 2) AS std_percentage_laid_off,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY percentage_laid_off) AS median_percentage_laid_off,
    ROUND(CAST(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY percentage_laid_off) - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY percentage_laid_off) AS NUMERIC), 2) AS iqr_percentage_laid_off
FROM layoffs_staging;

/*
Rolling sum of the number of employees laid off across months
*/
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

/*
Industries with largest layoffs
*/
SELECT 
    industry,
    SUM(total_laid_off) AS sum_total_laid_off,
    ROUND(AVG(percentage_laid_off), 2) AS avg_percentage_laid_off
FROM layoffs_staging
WHERE total_laid_off IS NOT NULL
GROUP BY industry
HAVING SUM(total_laid_off) >= 20000
ORDER BY sum_total_laid_off DESC;

/*
Countries with largest layoffs
*/
SELECT
    country,
    SUM(total_laid_off) AS sum_total_laid_off,
    ROUND(AVG(percentage_laid_off), 2) AS avg_percentage_laid_off
FROM layoffs_staging
WHERE total_laid_off IS NOT NULL
GROUP BY country
HAVING SUM(total_laid_off) >= 10000
ORDER BY sum_total_laid_off DESC;

-- Let us take a closer look at the 2 top European countries in the above query
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

/*
Companies sorted by descending order of number of
employees laid off across the entire covered period
(Per year if you uncomment the date-related lines below)
*/
SELECT 
    company,
    SUM(total_laid_off) AS sum_total_laid_off,
    ROUND(AVG(percentage_laid_off), 2) AS avg_percentage_laid_off,
    CASE
        WHEN ROUND(AVG(percentage_laid_off), 2) IS NULL THEN NULL
        WHEN ROUND(AVG(percentage_laid_off), 2) < 0.1 THEN 'Small'
        WHEN ROUND(AVG(percentage_laid_off), 2) BETWEEN 0.1 AND 0.33 THEN 'Intermediate'
        ELSE 'Large'
    END AS bucket_relative_layoff
FROM layoffs_staging
WHERE
    total_laid_off IS NOT NULL
    -- AND date IS NOT NULL
    -- AND EXTRACT(YEAR from date) = 2021
GROUP BY company
ORDER BY sum_total_laid_off DESC;

/*
Top 3 companies with largest layoffs for each year
*/
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

/*
Top 3 industries with largest layoffs for each year
*/
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

/*
Focusing on the top industry in 2023, i.e.,'Other',
what mostly contributed to it? 
*/
SELECT 
    company,
    total_laid_off,
    percentage_laid_off
FROM layoffs_staging
WHERE
    EXTRACT(YEAR FROM date) = 2023
    AND industry = 'Other'
    AND total_laid_off IS NOT NULL
    AND total_laid_off > 5000
ORDER BY
    total_laid_off DESC;