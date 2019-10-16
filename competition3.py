import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn import metrics

# Define missing values
missing_values = ["#N/A", "N/A"]

# Import the dataset
folder_path = "D:/Documents/College/FINAL YEAR/CSU44061 MACHINE LEARNING/Individual Competition/Data/"
training_dataset = pd.read_csv(folder_path + "tcd_competition_train.csv", na_values = missing_values) # Shape RxC (111993, 12)
testing_dataset = pd.read_csv(folder_path + "tcd_competition_test.csv", na_values = missing_values)

# Selecting variables and categorise them into numerical and categorical
numerical_data = ['Year of Record', 'Age', 'Size of City', 'Wears Glasses', 'Body Height [cm]']
categorical_data = ['Gender', 'Country', 'Profession', 'University Degree', 'Hair Color']

# Create a train/test split for training_data
X = training_dataset.drop('Income in EUR', axis=1)
y = training_dataset['Income in EUR']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a pipeline, starting with appropriate transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('target', TargetEncoder(handle_missing='value', handle_unknown='value')),
    ('imputer', SimpleImputer(strategy='mean'))])

# Apply transformers to appropriate features
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numerical_transformer, numerical_data),
        ('categorical', categorical_transformer, categorical_data)])
        
# Perform the regression, in this case a boosted Random Forest
regressor = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regression', AdaBoostRegressor(RandomForestRegressor(random_state=0)))])

# Fit the Random Forest to the training data
regressor.fit(X_train, y_train)

# Predict from the data
X_test_new = testing_dataset.drop('Income', axis=1)
y_pred = regressor.predict(X_test_new)
dataframe = pd.DataFrame({'Income':y_pred,'Instance':testing_dataset['Instance']}).to_csv(folder_path + "predictions7.csv", index=None)

# Find root mean squared error (commented out as only used in testing)
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))