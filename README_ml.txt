## Machine learning model development

# Load data
First pre-processed data is loaded

# Eliminate records with LOS=0 and LOS>14
For predicting LOS, we only considered records with 1<= LOS <=14, as there were very limited number of records with higher LOS values which is not suitable for training machine learning models.

# Rename columns
Some columns in the data set were not very clear for the non-technical people. Therefore, few columns were renamed with meaningful namesa as follows.
		"Gender Desc":"Gender", 
		"Smoker Flag":"Smoker Status", 
		"Diag Code":"Diagnostic Code"

# Define pre-processor
A pre-processor was defines to apply one-hot encoding and standardization. Because this should be applied seperately on training and testing data in each iteration (ex: in cross validation)

# Define train and test sets
First the dataset is shuffled to make the records in a random order and then define X as features variables and y as target variable. In order to seperate the train and test sets, we defined a boundary value by deviding the whole set in the 70"30 proportion. Using that boundary index, split the X and y sets into train and test sets accordingly.


# Functions to retrieve performance metrices
Two functions were written to obtain the performance metrices. The function named "print_cv_metrics_mean" will print mean mentric values during cross validation. Beacause for each iteration in the cross validation, performance is evaluated and to get the mean value of each performance metric at the end of CV, this function used. Thsi function returns mean of Mean Squared Error (MSE) values.
Other function named "evaluate_models" is used to evaluate models on test data after training. This function returns, MSE, Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

# ML model development
Initially baseline models were trained and validated using the training set via 5-fold cross validation.
Then using GridSearchCV techniques, hyperparameters were tuned.
After tuning, best hyperparameter values and best score were retrieved to compare with the baseline models.
Finally, models with best hyperparameter values were developed again using the training set and tested using test set and retrieved the performance metrices.