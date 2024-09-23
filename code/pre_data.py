import pandas as pd
import numpy as np
from numpy import nan
from pandas import read_csv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from matplotlib import pyplot


#load the dataset
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
dataset = read_csv('../data_tsel.csv')
dataset.describe().round(2)
print(dataset.describe().round(2))

#replace '0' values with 'nan'
dataset[['Mo']] = dataset[['Mo']].replace(0, nan)
data = dataset.values
ix = [i for i in range(data.shape[1]) if i!=12 and 13]
X, y = data[:, ix], data[:, 12]
#print('Missing: %d' % sum(np.isnan(X).flatten()))


#imputer
imputer = SimpleImputer(strategy='mean')
#fit and transform the dataset
imputer.fit_transform(X)

#summarize total missing
#print('Missing: %d' % sum(np.isnan(Xtrans).flatten()))

#evalate each strategy on the dataset
results = list()
#strategies = ['mean', 'median', 'most_frequent', 'constant']
strategies = [str(i) for i in range(1,21)]
for s in strategies:
    ##create the modeling pipeline
    pipeline = Pipeline(steps=[('i', IterativeImputer(max_iter=int(s))), ('m', RandomForestRegressor())])
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    scores = np.absolute(scores)
    #store results
    results.append(scores)
    print('>%s %.3f (%.3f)' % (s, np.mean(scores), np.std(scores)))

#plot model performance for comparison
pyplot.boxplot(results, labels=strategies, showmeans=True)
pyplot.show()


#descriptor
import pandas as pd
from xenonpy.datatools import preset
from xenonpy.datatools import Dataset
from xenonpy.descriptor import Compositions


#df1 = pd.read_csv('data_composition.csv')
data = pd.read_csv()
X = data.iloc[:, ] #elements
y = data.iloc[:, ] #target

preset.sync('elements')
preset.sync('elements_completed')

preset.elements
preset.elements_completed

data = []
for i in range(0, 367, 1):
    comps = X.loc[i]
    a = dict(comps)
    data.append(a)
print(data)

cal = Compositions()
descriptor = cal.transform(data)
re = pd.DataFrame(descriptor)
re.to_csv('../data_descriptor_ts.csv', index=False)

#samples = preset.sync('elements')
'''
cal = Compositions()
descriptor = cal.transform(samples)
descriptor.head(5)
'''
