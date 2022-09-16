# Develop XGBoost models

# Baseline model
np.random.seed(42)
gb_model_baseline = xgboost.XGBRegressor()
pipe_xgb_baseline = Pipeline(steps=[("preprocess",preprocessor),
                      ("model",gb_model_baseline)
                      ]
               )

cross_val_scores_gb = cross_validate(pipe_xgb_baseline,X_train,y_train, cv=5, scoring=scoring)

# Retrieve evaluation metric
print_cv_metrics_mean(cross_val_scores_gb)

# Hyperparameter Tuning with GridSearchCV
np.random.seed(42)
model_gb_gs = xgboost.XGBRegressor()
pipe_xgb_gs = Pipeline(steps=[("preprocess",preprocessor),
                      ("model",model_gb_gs)
                      ]
               )
gb_gs_grid = {"model__eta": [0.01,0.1,0.3],
                    "model__n_estimators":[150,200,300],
                    "model__max_depth":[1,2,5],
                    "model__subsample":[1],
                    "model__colsample_bytree":[0.5]}

gs_model_gb = GridSearchCV(estimator=pipe_xgb_gs,
                           param_grid=gb_gs_grid,
                           cv=5,
                           verbose=2)
# Train and validate models
gs_model_gb.fit(X_train,y_train)

# Retrieve best model hyperparameter values and best score
gs_model_gb.best_params_
gs_model_gb.best_score_

# Retrain best model
np.random.seed(42)
final_model_gb = xgboost.XGBRegressor(ets = 0.1, 
                                      n_estimators = 300,
                                      subsample=1, 
                                      max_depth = 1,
                                      colsample_bytree = 0.5,
                                      random_state=42
                                    )
pipe_xgb_final = Pipeline(steps=[("preprocess",preprocessor),
                      ("model",final_model_gb)
                      ]
               )

pipe_xgb_final.fit(X_train,y_train)
final_y_preds_gb = pipe_xgb_final.predict(X_test)

# Retrieve performance evaluation metrices
evaluate_models(y_test,final_y_preds_gb)