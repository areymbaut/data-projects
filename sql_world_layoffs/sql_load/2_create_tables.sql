-- Create company_dim table with primary key
CREATE TABLE public.layoffs
(
    company TEXT,
    location TEXT,
    industry TEXT,
    total_laid_off TEXT,
    percentage_laid_off TEXT,
    date TEXT,
    stage TEXT,
    country TEXT,
    funds_raised_millions TEXT
);

-- Set ownership of the tables to the postgres user
ALTER TABLE public.layoffs OWNER TO postgres;