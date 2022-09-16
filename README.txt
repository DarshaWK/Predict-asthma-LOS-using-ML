## Data Pre-processing ##

* Extract Admit Month from Admit Date

* Removing rows having Ethnicity Prioritised Desc "unknown" ("Don't know" or "Not stated") records

* Removing columns:
        - Admit Date
        - Admit CBU Desc
        - Discharge Date
        - DHB Desc
        - Ethnicity Desc
        - Ethnicity Prioritised Desc
        - Diag Desc

- Grouping DHBs (4 main DHBs, Others removed)
        - Auckland 
        - Waitemata
        - Counties Manukau
        - Northland

- Converting categorical to numerical
        - Gender Desc, Smoker Flag (mapping)
        - Ethnicity Group, DHB Groups (Label encoding)
        - Admit Day of Week, Admit Month, Diag Code (onehot encoding)
        - Standardizing Age column
		

# Feature Derivation
Dataset contains the `AdmitDate` of the hospital admission for each patient. 
To include a seasonality factor in our dataset, we extract the `AdmitMonth` from`AdmitDate`

# Removing unnecassary features
Remove rows having Ethnicity Prioritised Desc "unknown" ("Don't know" or "Not stated") records.
Remove a row with Gender Desc having the value "Unspecified"

# Grouping Ethnic groups
Ethnic groups of the patients are grouped into prioritized/major ethnic groups as mentioned in https://www.health.govt.nz/nz-health-statistics/data-references/code-tables/common-code-tables/ethnicity-code-tables

# Extract 4 main DHBs
There ara multiple DHBs available in th edataset. But many of them are having only very few instances. 
Therefore, we focus only on the frequently available DHBs; 
- Auckland            (8583)
- Waitemata           (1224)
- Counties Manukau     (888)
- Northland            (192)

# Grouping minor DHB groups
To make easy to remove minor DHBs, first the less frequent DHBs are grouped into one group `Other`

# Removing unnecessary columns
The dataset set has few other columns which does not add new/more information to the dataset and some are not applicable in machine learning model development. Therefore following columns were removed.

> - Admit Date
> - Admit CBU Desc
> - Discharge Date
> - DHB Desc 
> - Ethnicity Desc
> - Ethnicity Prioritised Desc
> - Diag Desc

# Handle Outliers
LOS variable is having many outliers. Outliers were observed from some graphs and using Z-score we remove them from the dataset.


# Convert binary categorical features to numeric
To convert binary-valued-features, we just map the values to 0 and 1
	Convert "Smoker Flag"
		- N = 0
		- Y = 1

	Convert "Gender"
		- Female = 0
		- Male = 1

# Convert categorical features to numerical
Before input data to the machine learning models, categorical data should be converted to numerical values. The following categorical variables are converted accordingly.
	- Admit Day of Week
	- Gender Desc
	- Ethnicity Group
	- Smoker Flag
	- Diag Code
	- DHB Group
	- Admit Month