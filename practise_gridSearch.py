import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from ydata_profiling import ProfileReport
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor



data = pd.read_csv('StudentScore.xls')
print(data.columns)
x = data.drop('math score', axis = 1)
y = data[['math score']]
x_train,  x_test, y_train,  y_test = train_test_split(x, y, test_size=0.2, random_state=5)


num_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy= 'median')),
    ('scaler', StandardScaler())
])

one_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])
# for feature in ['gender', 'parental level of education', 'lunch' , 'test preparation course']:
#     print(f'{feature}_order = {data[feature].unique()}')

gender_order = ['female', 'male']
education_order = ['some high school', 'high school', 'some college',"associate's degree",
                    "bachelor's degree", "master's degree" ]
lunch_order = ['standard', 'free/reduced']
course_order = ['none', 'completed']
 
ord_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[gender_order, education_order, lunch_order, course_order]))
])



preprocessor = ColumnTransformer(transformers=[
    ('num', num_transform, ['reading score', 'writing score' ]),
    ('one', one_transform, ['race/ethnicity']),
    ('ord', ord_transform, ['gender', 'parental level of education', 'lunch' , 'test preparation course']),
])


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('Regressor', RandomForestRegressor())
    
])

params = {
    'Regressor__n_estimators': [50, 100, 150],
    "Regressor__criterion" : ['squared_error', 'absolute_error'],
    'Regressor__max_depth' : [10, 20],
    'Regressor__min_samples_split' : [2, 3,  5],
    "preprocessor__num__imputer__strategy" : ['mean', 'median']
}





# model = GridSearchCV(model, param_grid= params, scoring= 'r2', n_jobs = 6, cv = 4, verbose=2)
model = RandomizedSearchCV(model, param_distributions= params, scoring= 'r2', n_jobs = 6, cv = 4, verbose=2, n_iter= 10)

model.fit(x_train, y_train)


print(f'the best score : {model.best_score_}')
print(f'the best pramas : {model.best_params_}')