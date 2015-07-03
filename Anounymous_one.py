%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
## Fetch the data and load it in pandas
data = pd.read_csv("data_amadeus.csv")
data.head()
data.hist(column='log_PAX', bins=20);
data.hist(column='PAX', bins=15);
data.hist('std_wtd', bins=15);
data.hist('WeeksToDeparture', bins=15);
print data['log_PAX'].mean()
print data['log_PAX'].std()
## Preprocessing for prediction
data_encoded = data

data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Departure'], prefix='d'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Arrival'], prefix='a'))
data_encoded = data_encoded.drop('Departure', axis=1)
data_encoded = data_encoded.drop('Arrival', axis=1)

# following http://stackoverflow.com/questions/16453644/regression-with-date-variable-using-scikit-learn
data_encoded['DateOfDeparture'] = pd.to_datetime(data_encoded['DateOfDeparture'])
#data_encoded['year'] = data_encoded['DateOfDeparture'].dt.year
#data_encoded['month'] = data_encoded['DateOfDeparture'].dt.month
#data_encoded['day'] = data_encoded['DateOfDeparture'].dt.day
data_encoded['weekday'] = data_encoded['DateOfDeparture'].dt.weekday
data_encoded['week'] = data_encoded['DateOfDeparture'].dt.week
data_encoded['n_days'] = data_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

#data_encoded = data_encoded.join(pd.get_dummies(data_encoded['year'], prefix='y'))
#data_encoded = data_encoded.join(pd.get_dummies(data_encoded['month'], prefix='m'))
#data_encoded = data_encoded.join(pd.get_dummies(data_encoded['day'], prefix='d'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['weekday'], prefix='wd'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['week'], prefix='w'))
data_encoded.tail(5)
features = data_encoded.drop(['PAX', 'log_PAX','DateOfDeparture'], axis=1)
X_columns = data_encoded.columns.drop(['PAX', 'log_PAX','DateOfDeparture'])
X = features.values
y = data_encoded['log_PAX'].values
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score

reg = LinearRegression()

scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='mean_squared_error')
print("log RMSE: {:.4f} +/-{:.4f}".format(
    np.mean(np.sqrt(-scores)), np.std(np.sqrt(-scores))))
reg.fit(X_train, y_train);
dict(zip(X_columns, reg.coef_))
Exercise: Visualize the coefficients, try to make sense of them.
## Random Forests
def objective_function(x):
    objective_function.n_iterations += 1
    lg_n_estimators, max_features, lg_max_depth = x
    n_estimators = int(10 ** lg_n_estimators)
    max_features = int(max_features)
    max_depth = int(10 ** lg_max_depth)
    print objective_function.n_iterations, \
        ": n_estimators = ", n_estimators, \
        ": max_features = ", max_features, \
        ": max_depth = ", max_depth
    reg = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth)
    scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='mean_squared_error', n_jobs=3)
    print "log RMSE = ", np.mean(np.sqrt(-scores))
    return np.mean(np.sqrt(-scores))
This may run for a while.
from hyperopt import fmin as hyperopt_fmin
from hyperopt import tpe, hp, STATUS_OK, space_eval

objective_function.n_iterations = 0
best = hyperopt_fmin(objective_function,
    space=(hp.quniform('lg_n_estimators', 0., 3., 0.1),
           hp.quniform('max_features', 1., 10., 1.),
           hp.quniform('lg_max_depth', 0., 2., 0.1),
          ),
    algo=tpe.suggest,
    max_evals=100)
best
# hard-saved the best in case you don't want to re-execute the tuning
best = {'lg_max_depth': 1.9,
        'lg_n_estimators': 2.7,
        'max_features': 10.0}
%%time
from sklearn.ensemble import RandomForestRegressor

n_estimators = int(10**best['lg_n_estimators'])
max_depth = int(10**best['lg_max_depth'])
max_features = int(best['max_features'])
print n_estimators, max_depth, max_features
reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)

scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='mean_squared_error',n_jobs=3)
print("log RMSE: {:.4f} +/-{:.4f}".format(
    np.mean(np.sqrt(-scores)), np.std(np.sqrt(-scores))))
## Variable importances
reg.fit(X_train, y_train)
len(X_columns)
plt.figure(figsize=(15, 5))

ordering = np.argsort(reg.feature_importances_)[::-1][:50]

importances = reg.feature_importances_[ordering]
feature_names = X_columns[ordering]

x = np.arange(len(feature_names))
plt.bar(x, importances)
plt.xticks(x + 0.5, feature_names, rotation=90, fontsize=15);
### The feature extractor

The feature extractor implements a single <code>transform</code> function. It receives the full pandas object X_df (without the labels). It should produce a numpy array representing the features extracted. If you want to use the (training) labels to save soem state of the feature extrctor, you can do it in the fit function. 


You can choose one of the example feature extractors and copy-paste it into your feature_extractor.py file.
class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
        X_array = X_encoded.values
        return X_array
class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        
        #uncomment the line below in the submission
        #path = os.path.dirname(__file__)
        data_weather = pd.read_csv("data_weather.csv")
        
        X_weather = data_weather[['Date', 'AirPort', 'Max TemperatureC']]
        X_weather = X_weather.rename(columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})
        X_encoded = X_encoded.set_index(['DateOfDeparture', 'Arrival'])
        X_weather = X_weather.set_index(['DateOfDeparture', 'Arrival'])
        X_encoded = X_encoded.join(X_weather).reset_index()
        
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)

        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
        X_array = X_encoded.values
        return X_array
### The regressor

The regressor should implement an sklearn-like regressor with fit and predict functions. You can copy paste either of these into your first regressor.py file.
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = RandomForestRegressor(n_estimators=10, max_depth=10, max_features=10)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = LinearRegression()

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
### Unit test

This cells unit-tests your selected regressor and feature extractor and print the RMSE.
def train_model(X_df, y_array, skf_is):
    fe = FeatureExtractor()
    fe.fit(X_df, y_array)
    X_array = fe.transform(X_df)
    # Regression
    train_is, _ = skf_is
    X_train_array = np.array([X_array[i] for i in train_is])
    y_train_array = np.array([y_array[i] for i in train_is])
    reg = Regressor()
    reg.fit(X_train_array, y_train_array)
    return fe, reg

def test_model(trained_model, X_df, skf_is):
    fe, reg = trained_model
    # Feature extraction
    X_array = fe.transform(X_df)
    # Regression
    _, test_is = skf_is
    X_test_array = np.array([X_array[i] for i in test_is])
    y_pred_array = reg.predict(X_test_array)
    return y_pred_array
from sklearn.cross_validation import ShuffleSplit

data = pd.read_csv("data_amadeus.csv")
X_df = data.drop(['PAX', 'log_PAX'], axis=1)
y_array = data['log_PAX'].values

skf = ShuffleSplit( y_array.shape[0], n_iter=2, test_size=0.2, random_state=61)
skf_is = list(skf)[0]

trained_model = train_model(X_df, y_array, skf_is)
y_pred_array = test_model(trained_model, X_df, skf_is)
_, test_is = skf_is
ground_truth_array = y_array[test_is]

score = np.sqrt(np.mean(np.square(ground_truth_array - y_pred_array)))
print score
