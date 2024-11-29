import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle



data = pd.read_csv('StudentScore.xls')




# split data 
target = 'math score'
x = data.drop(target, axis = 1)
y = data[[target]]
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.2, random_state= 5)



with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

model.fit(x_train, y_train)

print(f'Best parameters: {model.best_params_}')   
print(f'Best Score: {model.best_score_}')