# Python project - Customer call list cleanup

## Introduction
The goal is to clean up the dataset contained in [customer_call_list.xlsx](customer_call_list.xlsx) to isolate a list of customers that can be called by an imaginary commercial team.

The dataset was obtained [here](https://github.com/AlexTheAnalyst/PandasYouTubeSeries/blob/main/Customer%20Call%20List.xlsx) from Alex Freberg's Github repository.

## Tools I used
This project was carried out using the following tools:
- **Python (Pandas)** - backbone of the data cleanup.
- **Jupyter Notebooks** - facilitating table visualizations during the data cleanup.
- **Visual Studio Code** - my go-to code editor.
- **Git/Github** - essential for version control and code sharing.

## Data cleaning

The data cleaning is described/visualized step by step in [data_cleaning.ipynb](data_cleaning.ipynb) and amounts to the code below.

```python
import pandas as pd

# Load excel file
df = pd.read_excel('customer_call_list.xlsx')

# Remove duplicate and irrelevant column
df = df.drop_duplicates()
df = df.drop(columns='Not_Useful_Column')

# Standardize column names
rename_dict = {'CustomerID': 'Customer_ID',
               'Paying Customer': 'Paying_Customer'}
df = df.rename(columns=rename_dict)

# Strip incorrect characters at the beginning and/or end of some last names
df['Last_Name'] = df['Last_Name'].str.strip('./_')

# Replace everything in 'Phone_Number' except numbers by an empty string
df['Phone_Number'] = df['Phone_Number'].str.replace('[^0-9]', '', regex=True)

# Format phone number as xxx-xxx-xxxx
format_phone_number = lambda x: str(x)[0:3] + '-' + str(x)[3:6] + '-' + str(x)[6:10]
df['Phone_Number'] = df['Phone_Number'].apply(format_phone_number)

# Take out weird nan-like entries in 'Phone_Number'
repl_dict = {'nan--': '',
             '--': ''}
df['Phone_Number'] = df['Phone_Number'].replace(repl_dict)

# Split the `Address` column into the `Street_Address`, `State` and `Zip_Code` columns
df[['Street_Address', 'State', 'Zip_Code']] = df['Address'].str.split(',', n=2, expand=True)

# Remove the `Address` column
df = df.drop(columns='Address')

# Rename some entries in the columns below for consistency
# (regex=False is necessary to ensure exact matching)
repl_dict = {'Y': 'Yes',
             'N': 'No'}
df['Paying_Customer'] = df['Paying_Customer'].replace(repl_dict, regex=False)
df['Do_Not_Contact'] = df['Do_Not_Contact'].replace(repl_dict, regex=False)

# Take out remanining NaN values
df = df.fillna('')
df['Paying_Customer'] = df['Paying_Customer'].replace('N/a', '')
df['Do_Not_Contact'] = df['Do_Not_Contact'].replace('nan', '')

# Isolate customers that can be called
df = df.loc[(df['Phone_Number'] != '')
            & (df['Do_Not_Contact'] != 'Yes')]

# Remove the 'Zip_Code' and 'Do_Not_Contact' columns, which are now irrelevant
df = df.drop(columns=['Zip_Code', 'Do_Not_Contact'])

# Perform final index reset
df = df.reset_index(drop=True)
```