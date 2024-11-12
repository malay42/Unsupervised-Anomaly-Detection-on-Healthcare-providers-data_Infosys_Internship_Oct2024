
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# %matplotlib inline

df=pd.read_csv('data_encoded.csv')


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

data_scaled.shape

from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 150],
    'max_samples': [0.6, 0.8, 1.0],
    'contamination': [0.04, 0.05, 0.06],
    'max_features': [0.5, 0.7, 1.0]
}

isolation_forest = IsolationForest(contamination=0.05, random_state=42)
labels = isolation_forest.fit_predict(data_scaled)
df['if_anomaly'] = labels
df['ifb_anomaly'] = np.where(df['if_anomaly'] == -1, 1, 0)

# pca = PCA(n_components=2)
# pca_df = pca.fit_transform(df.drop(['ifb_anomaly','if_anomaly'], axis=1))

# score = calinski_harabasz_score(pca_df, df['ifb_anomaly'])

pca = PCA(n_components=2)
pca_df = pca.fit_transform(df.drop(['ifb_anomaly','if_anomaly'], axis=1))

score = calinski_harabasz_score(pca_df, df['ifb_anomaly'])
score



from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import numpy as np

param_dist = {
    'n_estimators': [50, 100, 150],
    'max_samples': [0.6, 0.8, 1.0],
    'contamination': [0.04, 0.05, 0.06],
    'max_features': [0.5, 0.7, 1.0]
}

def custom_score(estimator, X):
    labels = estimator.fit_predict(X)
    pca = PCA(n_components=2)
    pca_df = pca.fit_transform(X)
    score = calinski_harabasz_score(pca_df, labels)
    return score

isolation_forest = IsolationForest(random_state=42)

random_search = RandomizedSearchCV(
    estimator=isolation_forest,
    param_distributions=param_dist,
    scoring=custom_score,  
    cv=3,
    n_iter=10,
    random_state=42
)

random_search.fit(data_scaled)

best_params = random_search.best_params_
print("Best Parameters:", best_params)



from skopt import BayesSearchCV

bayes_search = BayesSearchCV(
    estimator=isolation_forest,
    search_spaces=param_dist,
    scoring=custom_score,  
    n_iter=10,  
    cv=3,
    random_state=42
)

bayes_search.fit(data_scaled)

best_params = bayes_search.best_params_
print("Best Parameters:", best_params)



from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
import numpy as np

def custom_score2(estimator, X, y=None):
    labels = estimator.fit_predict(X)
    normal_data = X[labels == 1]
    centroid = normal_data.mean(axis=0)
    distances = pairwise_distances(normal_data, [centroid])
    return -distances.mean()  

isolation_forest = IsolationForest(random_state=42)

random_search1 = RandomizedSearchCV(
    estimator=isolation_forest,
    param_distributions=param_dist,
    scoring=custom_score2, 
    cv=3,
    n_iter=10,
    random_state=42
)

random_search1.fit(data_scaled)

best_params1 = random_search1.best_params_
print("Best Parameters:", best_params1)



bayes_search2 = BayesSearchCV(
    estimator=isolation_forest,
    search_spaces=param_dist,
    scoring=custom_score2,  
    n_iter=10,  
    cv=3,
    random_state=42
)

bayes_search2.fit(data_scaled)

best_params2 = bayes_search2.best_params_
print("Best Parameters:", best_params2)



[('contamination', 0.06),
 ('max_features', 0.7),
  ('max_samples', 0.8),
  ('n_estimators', 100)]