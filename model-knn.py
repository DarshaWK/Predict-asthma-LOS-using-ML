# Develop KNN models

# Baseline model
np.random.seed(42)
knn_model_baseline = KNeighborsRegressor()
pipe_knn_baseline = Pipeline(steps=[("preprocess",preprocessor),
                      ("model",gb_model_baseline)
                      ]
               )

cross_val_scores_knn = cross_validate(pipe_knn_baseline,X_train,y_train, cv=5, scoring=scoring)

# Retrieve evaluation metrices
print_cv_metrics_mean(cross_val_scores_knn)

# Hyperparameter Tuning with GridSearchCV
np.random.seed(42)
model_knn_gs = KNeighborsRegressor()
pipe_knn_gs = Pipeline(steps=[("preprocess",preprocessor),
                      (model_knn_gs)
                      ]
               )
knn_gs_grid = { "model__n_neighbors":[3,5,9],
               "model__p": [1,2],
                "model__weights": ["uniform","distance"],
                "model__algorithm" : ["auto", "ball_tree", "kd_tree", "brute"]
        }

gs_model_knn = GridSearchCV(estimator=pipe_knn_gs,
                           param_grid=knn_gs_grid,
                           cv=5,
                           verbose=2)
# Train and validate models
gs_model_knn.fit(X_train,y_train)

# Retrieve best model hyperparameter values and best score
gs_model_knn.best_params_
gs_model_knn.best_score_

# Retrain best model
np.random.seed(42)
model_knn_final = KNeighborsClassifier(algorithm = "ball_tree",
                                  n_neighbors = 9,
                                  p = 2,
                                  weights = "uniform")

pipe_knn_final = Pipeline(steps=[("preprocess",preprocessor),
                      ("model",model_knn_final)
                      ]
               )

pipe_knn_final.fit(X_train,y_train)
final_y_preds_knn = pipe_knn_final.predict(X_test)

# Retrieve performance evaluation metrices
evaluate_models(y_test,final_y_preds_knn)
