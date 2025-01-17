import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle
from lazypredict.Supervised import LazyRegressor


data = pd.read_csv('StudentScore.xls')
# print(data.info())
# profile = ProfileReport(data, title = 'Student_score')
# profile.to_file('Student_score.html')
# numrical_columns = data.select_dtypes(include=['int'])
# print(numrical_columns.corr())
# print(data.columns)


num_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])


one_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy= 'most_frequent')),
    ('encoder', OneHotEncoder())
])


gender_order = ['female', 'male']
edu_order = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree" ]
test_order = ['none', 'completed']
lunch_order = ['standard' ,'free/reduced']

ord_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy= 'most_frequent')),
    ('encoder', OrdinalEncoder(categories = [gender_order, edu_order, lunch_order, test_order]))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transform, ['reading score', 'writing score']),
    ('one', one_transform, ['race/ethnicity']),
    ('ord', ord_transform, ['gender', 'parental level of education', 'lunch', 'test preparation course'])
])

# split data 
target = 'math score'
x = data.drop(target, axis = 1)
y = data[[target]]
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.2, random_state= 5)

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)

reg = LazyRegressor()
models, predictions = reg.fit(x_train, x_test, y_train, y_test)

print(models)