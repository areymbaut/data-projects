/*
Create a duplicate table
(to heavily modify the data without affecting the raw data)
*/
CREATE TABLE layoffs_staging
AS TABLE layoffs;

/*
(Optional) Identify duplicates
*/
-- WITH duplicate_cte AS (
--     SELECT
--         *,
--         ROW_NUMBER() OVER(PARTITION BY company, location, industry, total_laid_off, percentage_laid_off, date, stage, country, funds_raised_millions) AS row_num
--     FROM layoffs_staging
-- )
-- SELECT *
-- FROM duplicate_cte
-- WHERE row_num > 1;

/*
Remove duplicates
*/
DELETE FROM layoffs_staging
WHERE ctid NOT IN (
  SELECT MIN(ctid)
  FROM layoffs_staging
  GROUP BY company, location, industry, total_laid_off, percentage_laid_off, date, stage, country, funds_raised_millions
);

/*
Standardize data (including converting 'NULL' to NULL)
*/
-- In company: some company names start with a space
UPDATE layoffs_staging
SET company = NULL
WHERE company = 'NULL';

UPDATE layoffs_staging
SET company = TRIM(company);

-- In industry: Crypto, Crypto Currency and CryptoCurrency exist
UPDATE layoffs_staging
SET industry = NULL
WHERE industry = 'NULL';

UPDATE layoffs_staging
SET industry = 'Crypto'
WHERE industry LIKE 'Crypto%';

-- In country: 'United States' vs 'United States.'
UPDATE layoffs_staging
SET country = NULL
WHERE country = 'NULL';

UPDATE layoffs_staging
SET country = TRIM(TRAILING '.' FROM country)
WHERE country LIKE 'United States_';

-- In location: 'Dusseldorf' vs 'Düsseldorf'
UPDATE layoffs_staging
SET location = NULL
WHERE location = 'NULL';

UPDATE layoffs_staging
SET location = 'Dusseldorf'
WHERE location = 'Düsseldorf';

-- Convert date
UPDATE layoffs_staging
SET date = NULL
WHERE date = 'NULL';

UPDATE layoffs_staging
SET date = TO_DATE(date, '%MM/%DD/%YYYY');

ALTER TABLE layoffs_staging
ALTER COLUMN date
TYPE DATE USING date::DATE;

-- In stage
UPDATE layoffs_staging
SET stage = NULL
WHERE stage = 'NULL';

-- In total_laid_off
UPDATE layoffs_staging
SET total_laid_off = NULL
WHERE total_laid_off = 'NULL';

ALTER TABLE layoffs_staging
ALTER COLUMN total_laid_off
TYPE INTEGER USING total_laid_off::INTEGER;

-- In percentage_laid_off
UPDATE layoffs_staging
SET percentage_laid_off = NULL
WHERE percentage_laid_off = 'NULL';

ALTER TABLE layoffs_staging
ALTER COLUMN percentage_laid_off
TYPE NUMERIC USING percentage_laid_off::NUMERIC;

-- In funds_raised_millions
UPDATE layoffs_staging
SET funds_raised_millions = NULL
WHERE funds_raised_millions = 'NULL';

ALTER TABLE layoffs_staging
ALTER COLUMN funds_raised_millions
TYPE NUMERIC USING funds_raised_millions::NUMERIC;

/*
NULL/blank values:
- Industries have some NULL values that can be imputed
    from other entries of their associated companies
- There is not much else we can do here without searching
    for information outside our dataset
*/
UPDATE layoffs_staging
SET industry = t2.industry
FROM layoffs_staging t2
WHERE 
    layoffs_staging.company = t2.company
    AND layoffs_staging.industry IS NULL
    AND t2.industry IS NOT NULL;

/*
NULL/blank values:
- Some data entries have NULL total_laid_off and percentage_laid_off
- These entries are useless in the context of analyzing layoffs,
    so we might as well delete them from layoffs_staging
*/
DELETE FROM layoffs_staging
WHERE 
    total_laid_off IS NULL
    AND percentage_laid_off IS NULL;


-- Final look up
SELECT *
FROM layoffs_staging;