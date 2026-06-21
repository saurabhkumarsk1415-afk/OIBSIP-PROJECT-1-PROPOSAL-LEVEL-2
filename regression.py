"""
Predicting House Prices with Linear Regression
Project 1 Proposal - Level 2
Dataset: Housing.csv (545 rows, Delhi region real estate)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# 1. LOAD & INSPECT
# ══════════════════════════════════════════════════════════
housing = pd.read_csv('Housing.csv')
print(f"Shape: {housing.shape}")
print(f"Nulls: {housing.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════
# 2. OUTLIER TREATMENT (IQR method on price and area)
# ══════════════════════════════════════════════════════════
Q1 = housing.price.quantile(0.25)
Q3 = housing.price.quantile(0.75)
IQR = Q3 - Q1
housing = housing[(housing.price >= Q1 - 1.5*IQR) & (housing.price <= Q3 + 1.5*IQR)]

Q1 = housing.area.quantile(0.25)
Q3 = housing.area.quantile(0.75)
IQR = Q3 - Q1
housing = housing[(housing.area >= Q1 - 1.5*IQR) & (housing.area <= Q3 + 1.5*IQR)]

print(f"Shape after outlier removal: {housing.shape}")

# ══════════════════════════════════════════════════════════
# 3. DATA PREPARATION
# ══════════════════════════════════════════════════════════
varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
housing[varlist] = housing[varlist].apply(lambda x: x.map({'yes': 1, 'no': 0}))

status = pd.get_dummies(housing['furnishingstatus'], drop_first=True).astype(int)
housing = pd.concat([housing, status], axis=1)
housing.drop(['furnishingstatus'], axis=1, inplace=True)

housing.to_csv('Housing_clean.csv', index=False)

# ══════════════════════════════════════════════════════════
# 4. TRAIN-TEST SPLIT & SCALING
# ══════════════════════════════════════════════════════════
np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size=0.7, test_size=0.3, random_state=100)

scaler = MinMaxScaler()
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_test[num_vars] = scaler.transform(df_test[num_vars])

y_train = df_train.pop('price')
X_train = df_train
y_test = df_test.pop('price')
X_test = df_test

# ══════════════════════════════════════════════════════════
# 5. FEATURE SELECTION (RFE)
# ══════════════════════════════════════════════════════════
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=6)
rfe.fit(X_train, y_train)
selected_cols = X_train.columns[rfe.support_]
print(f"Selected features: {list(selected_cols)}")

# VIF check
X_train_rfe = X_train[selected_cols]
vif = pd.DataFrame()
vif['Feature'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
print(f"\nVIF:\n{vif.sort_values('VIF', ascending=False)}")

# ══════════════════════════════════════════════════════════
# 6. MODEL BUILDING (OLS)
# ══════════════════════════════════════════════════════════
X_train_sm = sm.add_constant(X_train_rfe.astype(float))
model = sm.OLS(y_train, X_train_sm).fit()
print(model.summary())

# ══════════════════════════════════════════════════════════
# 7. MODEL EVALUATION ON TEST SET
# ══════════════════════════════════════════════════════════
X_test_rfe = X_test[selected_cols]
X_test_sm = sm.add_constant(X_test_rfe.astype(float))
X_test_sm = X_test_sm[X_train_sm.columns]
y_pred = model.predict(X_test_sm)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n=== TEST SET PERFORMANCE ===")
print(f"R2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
