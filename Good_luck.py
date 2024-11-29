import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# load data
data = pd.read_csv('StudentScore.xls')



# profile = ProfileReport(data, title = 'StudentScore')
# profile.to_file('StudentScore')


# Split data
x = data.drop('math score', axis = 1)
y = data[['math score']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 5)

num_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler())
])

education_order = ['some high school', 'high school', 'some college',  "associate's degree",
                    "bachelor's degree", "master's degree" ]
gender_order = data['gender'].unique()
lunch_order = data['lunch'].unique()
test_order = data['test preparation course'].unique()


odr_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('encoder', OrdinalEncoder(categories=[education_order, gender_order,lunch_order, test_order ]))
])

one_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('encoder', OneHotEncoder())
])

preprocessor = ColumnTransformer(transformers=[

    ('num', num_transform, ['reading score', 'writing score']),
    ('ord', odr_transform, ['parental level of education', 'gender', 'lunch', 'test preparation course']),
    ('one', one_transform, ['race/ethnicity'])
])

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)


LR = LinearRegression()
LR.fit(x_train, y_train)

y_pred = LR.predict(x_test)

print(f'r2_score: {r2_score(y_test, y_pred)}')
print(f'mean_squared_error: {mean_squared_error(y_test, y_pred)}')
print(f'mean_absolute_error: {mean_absolute_error(y_test, y_pred)}')


y_pred = y_pred.ravel()  # Or use y_pred = y_pred.flatten()
y_test = y_pred.ravel()
# Create the DataFrame
data_output = {'True Value': y_test, 'Predicted Value': y_pred}
df = pd.DataFrame(data_output)

# Display the DataFrame
print(df)
