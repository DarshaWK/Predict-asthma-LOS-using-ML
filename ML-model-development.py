# Import tools and libraries
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost.XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
# Load data
asthma_admission_data = pd.read_csv("data/processed_admission_data.csv")

# Eliminate LOS=0 records
asthma_admission_data.drop(asthma_admission_data[asthma_admission_data['LOS']==0].index, axis=0, inplace=True)

# Eliminate records with LOS>14
asthma_admission_data.drop(asthma_admission_data[asthma_admission_data["LOS"]>14].index, axis=0, inplace=True)

# Rename columns for clear identification
asthma_admission_data.rename(columns={"Gender Desc":"Gender", "Smoker Flag":"Smoker Status", "Diag Code":"Diagnostic Code"}, inplace=True)

# Define pre-processor
categorical_features = ["Admit Day of Week", "Diagnostic Code", "Admit Month", "Ethnicity Group", "DHB Group"]
ohe = OneHotEncoder(sparse=False)
num_feature = ["Age"]
std_scalar = StandardScaler()
preprocessor = ColumnTransformer(
    transformers = [("categorical", ohe, categorical_features),
                   ("numerical", std_scalar, num_feature)],
    remainder="passthrough")

    
# Define train and test sets
np.random.seed(42)

# Shuffle the dataset
asthma_admission_data_shuffled = asthma_admission_data.sample(frac=1).reset_index(drop=True)

# Defining X and y
X = asthma_admission_data_shuffled.drop("LOS",axis=1)
y = asthma_admission_data_shuffled["LOS"]

# Defining boundry limits for train and validation sets
train_split = round(0.7*len(asthma_admission_data_shuffled)) #80:20%

# Splitting into train, val and test sets
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Define functions for model evaluation
scoring = ['neg_mean_squared_error','r2','neg_mean_absolute_error']#,'roc_auc']

# Functions to retrive performance metrics values
def print_cv_metrics_mean(metrics):
    # This function is used to return mean metric values in cross validation 
    print(f"Mean MSE: {metrics['test_neg_mean_squared_error'].mean():.2f}")
    
def evaluate_models (y_true, y_preds):
    # This function is used to return the metrics values of ML models on test data
    MSE = mean_squared_error(y_true,y_preds)
    RMSE = mean_squared_error(y_true,y_preds,squared=False)
    MAE = mean_absolute_error(y_true,y_preds) 

    # Printing the metrics values
    print(f"MSE: {MSE:.2f}")
    print(f"RMSE: {RMSE:.2f}")
    print(f"MAE: {MAE:.2f}") 


