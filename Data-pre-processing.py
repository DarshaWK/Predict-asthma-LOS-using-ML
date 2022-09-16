# Import tools
import numpy as np
import pandas as pd
import calendar

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data
admission_data = pd.read_csv("data/2017_to_2021_asthma_admissions_ALL DHBs.csv")


# Feature derivation
# Make a copy of the original dataframe before making any amendments
admission_data_copy = admission_data.copy()

# Convert `AdmitDate` to a datetime object
admission_data_copy['Admit Date'] = pd.to_datetime(admission_data_copy['Admit Date'])

# Generate new variable, "AdmitMonth"
admission_data_copy['Admit Month'] = admission_data_copy['Admit Date'].dt.month

# Change numeric representation of month to string format
admission_data_copy['Admit Month'] = admission_data_copy['Admit Month'].apply(lambda x: calendar.month_abbr[x])


# Remone unncessary features
admission_data_copy.drop(admission_data_copy.index[admission_data_copy["Ethnicity Prioritised Desc"] == "Don't Know"],inplace=True)
admission_data_copy.drop(admission_data_copy.index[admission_data_copy["Ethnicity Prioritised Desc"] == "Not Stated"],inplace=True)
admission_data_copy.drop(admission_data_copy.index[admission_data_copy["Gender Desc"] == "Unspecified"],inplace=True)


# Grouping into prioritized ethnic groups
admission_data_copy.loc[(admission_data_copy["Ethnicity Prioritised Desc"]=="New Zealand European")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="Other European")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="European NFD"),"Ethnicity Group"] = 'European'

admission_data_copy.loc[(admission_data_copy["Ethnicity Prioritised Desc"]=="Maori"),"Ethnicity Group"] = 'Maori'

admission_data_copy.loc[(admission_data_copy["Ethnicity Prioritised Desc"]=="Tongan")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="Samoan")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="Cook Island Maori")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="Niuean")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="Fijian")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="Other Pacific Peoples")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="Tokelauan")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="Pacific Island NFD"),"Ethnicity Group"] = 'Pacific Peoples'

admission_data_copy.loc[(admission_data_copy["Ethnicity Prioritised Desc"]=="Indian")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="Chinese")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="Other Asian")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="Southeast Asian")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="Asian NFD"),"Ethnicity Group"] = 'Asian'

admission_data_copy.loc[(admission_data_copy["Ethnicity Prioritised Desc"]=="Middle Eastern")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="African")|
                 (admission_data_copy["Ethnicity Prioritised Desc"]=="Latin American"),"Ethnicity Group"] = 'Middle Eastern/ Latin American/ African'

admission_data_copy.loc[(admission_data_copy["Ethnicity Prioritised Desc"]=="Other Ethnicity"),"Ethnicity Group"] = 'Other Ethnicity'


# Grouping minor DHBs into "Other" group
admission_data_copy["DHB Group"] = admission_data_copy["DHB Desc"]
admission_data_copy.loc[(admission_data_copy["DHB Desc"]=="Unknown")|
                 (admission_data_copy["DHB Desc"]=="Taranaki")|
                 (admission_data_copy["DHB Desc"]=="Hawke's Bay")|
                 (admission_data_copy["DHB Desc"]=="Lakes")|
                 (admission_data_copy["DHB Desc"]=="MidCentral")|
                 (admission_data_copy["DHB Desc"]=="Southern")|
                 (admission_data_copy["DHB Desc"]=="Capital and Coast")|
                 (admission_data_copy["DHB Desc"]=="Canterbury")|
                 (admission_data_copy["DHB Desc"]=="Whanganui")|
                 (admission_data_copy["DHB Desc"]=="Hutt")|
                 (admission_data_copy["DHB Desc"]=="Tairawhiti")|
                 (admission_data_copy["DHB Desc"]=="Nelson Marlborough")|
                 (admission_data_copy["DHB Desc"]=="Wairarapa")|
                 (admission_data_copy["DHB Desc"]=="South Canterbury")|
                 (admission_data_copy["DHB Desc"]=="Waikato")|
                 (admission_data_copy["DHB Desc"]=="Bay Of Plenty")
                 ,"DHB Group"] = 'Other'
        
# Remove "Other" DHBs
admission_data_copy.drop(admission_data_copy[admission_data_copy['DHB Group']=="Other"].index, axis=0, inplace=True)


# Eliminating unnecessary columns
admission_data_copy.drop(["Admit Date", "Admit CBU Desc","Discharge Date","DHB Desc","Ethnicity Desc","Ethnicity Prioritised Desc","Diag Desc"],axis=1,inplace=True)


# Handle outliers
# Calculating the Z_score (temporary variable created in the dataframe)
admission_data_copy['Z_score'] = (admission_data_copy["LOS"] - admission_data_copy["LOS"].mean())/admission_data_copy["LOS"].std()

# Removing instances with Z_score<3
admission_data_copy = admission_data_copy[admission_data_copy["Z_score"]<3]

# Removing the Z_score temporary variable
admission_data_copy.drop("Z_score",inplace=True,axis=1)


# Conert categorical features to numeric
admission_data_copy["Smoker Flag"] = admission_data_copy["Smoker Flag"].map(dict(N=0,Y=1))
admission_data_copy["Gender Desc"] = admission_data_copy["Gender Desc"].apply(lambda x:0 if x=='Female' else 1)

# Export data
admission_data_copy.to_csv("data/processed_admission_data.csv",index=False)