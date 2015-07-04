import pandas as pd
import numpy as np
import os

class FeatureExtractor(object):
    def __init__(self):
        pass
 
    def fit(self, X_df, y_array):
        pass
 
    def transform(self, data_encoded):
 
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Departure'], prefix='d'))
        data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Arrival'], prefix='a'))
        data_encoded = data_encoded.drop('Departure', axis=1)
        data_encoded = data_encoded.drop('Arrival', axis=1)
 
        
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
        path = os.path.dirname(__file__)
        data_oil = pd.read_csv(os.path.join(path, "oil.csv"),sep=';', decimal=',')
        path = os.path.dirname(__file__)
        data_holidays = pd.read_csv(os.path.join(path,"data_holidays_2.csv"))
        
        X_Oil = data_oil[['DateOfDeparture','Price']]
        X_holidays = data_holidays[['DateOfDeparture','Xmas','Xmas-1','NYD','NYD-1','Ind','Thg','Thg+1','Lab','Mem']]
        
        X_Oil = X_Oil.set_index(['DateOfDeparture'])
        X_holidays = X_holidays.set_index(['DateOfDeparture'])
        X_Oil = X_Oil.join(X_holidays).reset_index()   
        
        X_Oil['DateOfDeparture'] = pd.to_datetime(X_Temporary['DateOfDeparture'])
        
        data_encoded = data_encoded.merge(X_Oil, how='left', left_on=['DateOfDeparture'], right_on=['DateOfDeparture'], sort=False)
        data_encoded = data_encoded.drop(['index'], axis=1)
        
        X_array = np.array(data_encoded)
        
        return X_array
