import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# 1. Load and clean data
# -----------------------------
df = pd.read_csv("Salary_Data.csv")

# Fill missing values

object_columns = ['Education Level', 'Gender', 'Job Title']
for col in object_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

numeric_columns = ['Age', 'Years of Experience', 'Salary']
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

df.drop_duplicates(inplace=True)

# Drop unwanted columns
df_cleaned = df.drop(['Gender'], axis=1)


# 2. Remove outliers (IQR method)
# -----------------------------
continuous = ['Age', 'Years of Experience', 'Salary']
mask = pd.Series(True, index=df_cleaned.index)
for col in continuous:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask &= df_cleaned[col].between(lower_bound, upper_bound)

df_cleaned = df_cleaned[mask]


# 3. Features and target
# -----------------------------
X = df_cleaned.drop('Salary', axis=1)
y = df_cleaned['Salary']


# 4. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. Preprocessing
# -----------------------------
numeric_features = ['Age', 'Years of Experience']
categorical_features = ['Education Level', 'Job Title']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


# 6. XGBoost model
# -----------------------------
xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)


# 7. Pipeline
# -----------------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb)
])


# 8. Parameter grid
# -----------------------------
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 5, 7],
    'model__subsample': [0.7, 0.8, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 1.0],
    'model__gamma': [0, 0.1, 0.3],
    'model__reg_lambda': [1, 2, 5],
    'model__reg_alpha': [0, 0.5, 1]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1
)


# 9. Train GridSearchCV on TRAINING data
# -----------------------------
grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_


# 10. Evaluate on TEST data
# -----------------------------
y_pred = best_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(r2)
print(f"✅ Test MSE: ₹{mse:,.2f}")
print(f"✅ Test RMSE: ₹{rmse:,.2f}")
print("Best Parameters:", grid_search.best_params_)


# 11. Save pipeline (preprocessing + model together)
# -----------------------------
joblib.dump(best_pipeline, 'Salary_XGBoost_Pipeline.joblib')
print("🎉 Full pipeline saved successfully as 'Salary_XGBoost_Pipeline.joblib'")
