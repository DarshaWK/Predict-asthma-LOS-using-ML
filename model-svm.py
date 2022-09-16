# Developing SVM models

# Baseline model
np.random.seed(42)
model_svm_baseline = SVR()
pipe_svm_baseline = Pipeline(steps=[("preprocess",preprocessor),
                      ("model",model_svm_baseline)])

cross_val_scores_svm = cross_validate(pipe_svm_baseline,X_train,y_train, cv=5, scoring=scoring, error_score='raise')

# Retrieve evaluation metrices
print_cv_metrics_mean(cross_val_scores_svm)


# Hyperparameter Tuning with GridSearchCV
np.random.seed(42)
model_svm_4 = SVR()
pipe_svm_gs = Pipeline(steps=[("preprocess",preprocessor),
                      ("model",model_svm_3)])

svm_gs_grid = {"model__C":[0.1,1],
            "model__gamma": ['scale','auto'],
            "model__kernel": ["rbf","poly","linear"],
            "model__degree":[2,3]}

gs_model_svm = GridSearchCV(estimator=pipe_svm_gs,
                            param_grid=svm_gs_grid,
                            cv=5,
                            verbose=2)
# Train and validate models
gs_model_svm.fit(X_train,y_train)

# Retrieve best model hyperparameter values and best score
gs_model_svm.best_params_
gs_model_svm.best_score_


# Retrain best model
np.random.seed(42)
final_model_svm = SVR(C = 1,
                     degree = 2,
                     gamma = "scale",
                     kernel = "rbf")

pipe_svm_final = Pipeline(steps=[("preprocess",preprocessor),
                      ("model",final_model_svm)])

pipe_svm_final.fit(X_train,y_train)
final_y_preds_svm = pipe_svm_final.predict(X_test)

# Retrieve performance evaluation metrices
evaluate_models(y_test,final_y_preds_svm)
