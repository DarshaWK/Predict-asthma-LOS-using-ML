# Develop Random Forest models

# Baseline model
np.random.seed(42)
model_rf_baseline = RandomForestRegressor()
pipe_rf_baseline = Pipeline(steps=[("preprocess",preprocessor),
                      ("model",model_rf_baseline)])

cross_val_scores_rf = cross_validate(pipe_rf_baseline,X_train,y_train, cv=5, scoring=scoring, error_score="raise")

# Retrieve evaluation metric
print_cv_metrics_mean(cross_val_scores_rf)

# Hyperparameter Tuning with GridSearchCV
np.random.seed(42)
model_rf_gs = RandomForestRegressor(n_jobs=-1)
pipe_rf_gs = Pipeline(steps=[("preprocess",preprocessor),
                      ("model",model_rf_gs)
                      ]
               )

rf_gs_grid = {"model__n_estimators":[200,500,750],
              "model__max_features": ["log2","sqrt",1],
              "model__min_samples_split":[2,4],
              "model__min_samples_leaf":[1,2,4]}

gs_model_rf = GridSearchCV(estimator=pipe_rf_gs,
                           param_grid=rf_gs_grid,
                           scoring='r2',
                           cv=5,
                           verbose=2)
# Train and validate models
gs_model_rf.fit(X_train,y_train)

# Retrieve best model hyperparameter values and best score
gs_model_rf.best_params_
gs_model_rf.best_score_

# Retrain best model
np.random.seed(42)
final_model_rf = RandomForestRegressor(max_features='sqrt',
                                        min_samples_leaf=4, 
                                        min_samples_split=4, 
                                        n_estimators=200)

pipe_rf_final = Pipeline(steps=[("preprocess",preprocessor),
                      ("model",final_model_rf)
                      ]
               )

pipe_rf_final.fit(X_train,y_train)
final_y_preds_rf = pipe_rf_final.predict(X_test)

# Retrieve performance evaluation metrices
evaluate_models(y_test,final_y_preds_rf)