#after installing scikit optimize


#BAYESIAN SEARCH CODE

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from skopt import BayesSearchCV


numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])

def custom_scorer(estimator, X):

    y_pred = estimator.predict(X)
   
    return np.mean(y_pred == 1)

# Define the parameter space
param_space = {
    'n_estimators': (50, 200),
    'contamination': (0.01, 0.1),  
}


iso_forest = IsolationForest(random_state=42)


bayes_search = BayesSearchCV(
    estimator=iso_forest,
    search_spaces=param_space,
    n_iter=30,
    scoring=custom_scorer,
    n_jobs=-1,
    cv=3
)


bayes_search.fit(numeric_data)

best_iso_forest = bayes_search.best_estimator_

print("Best parameters found: ", bayes_search.best_params_)
#testing isolation forest with the results produced using bayesian search

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])


iso_forest = IsolationForest(n_estimators=50, contamination=0.01, random_state=42)


iso_forest.fit(numeric_data)


data_dropped['Anomaly'] = iso_forest.predict(numeric_data)


anomalies = data_dropped[data_dropped['Anomaly'] == -1]


plt.figure(figsize=(12, 6))


plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Services'],
            data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Payment Amount'],
            label='Normal', alpha=0.5)

plt.scatter(anomalies['Number of Services'],
            anomalies['Average Medicare Payment Amount'],
            color='red', label='Anomaly', alpha=0.7)
plt.title('Anomaly Detection: Number of Services vs Average Medicare Payment Amount')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))

plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Medicare Beneficiaries'],
            data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Allowed Amount'],
            label='Normal', alpha=0.5)
plt.scatter(anomalies['Number of Medicare Beneficiaries'],
            anomalies['Average Medicare Allowed Amount'],
            color='red', label='Anomaly', alpha=0.7)
plt.title('Anomaly Detection: Number of Medicare Beneficiaries vs Average Medicare Allowed Amount')
plt.xlabel('Number of Medicare Beneficiaries')
plt.ylabel('Average Medicare Allowed Amount')
plt.legend()
plt.grid()
plt.show()
print("Anomalies detected:")
print(anomalies)
print(f"Total number of anomalies: {anomalies.shape[0]}")
plt.savefig('Bayesian Search.png')  


#RANDOMISED SEARCH CODE

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])


iso_forest = IsolationForest(random_state=42)

param_dist = {
    'n_estimators': [10, 50, 100, 150, 200],  
    'max_samples': [0.5, 0.75, 1.0],         
    'contamination': [0.01, 0.05, 0.1, 0.2],  
    'max_features': [0.5, 0.75, 1.0]        
}


random_search = RandomizedSearchCV(
    estimator=iso_forest,
    param_distributions=param_dist,
    n_iter=20,                  
    scoring='accuracy',      
    cv=5,                     
    random_state=42,
    n_jobs=-1                  
)


random_search.fit(numeric_data)

best_iso_forest = random_search.best_estimator_
best_params = random_search.best_params_

print("Best parameters found:", best_params)
print("Best estimator:", best_iso_forest)

data_dropped['Anomaly'] = best_iso_forest.fit_predict(numeric_data)
anomalies = data_dropped[data_dropped['Anomaly'] == -1]

print(f"Total number of anomalies detected with best parameters: {anomalies.shape[0]}")

#implementation with the produced parameters

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])


iso_forest = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)


iso_forest.fit(numeric_data)


data_dropped['Anomaly'] = iso_forest.predict(numeric_data)


anomalies = data_dropped[data_dropped['Anomaly'] == -1]


plt.figure(figsize=(12, 6))


plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Services'],
            data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Payment Amount'],
            label='Normal', alpha=0.5)

plt.scatter(anomalies['Number of Services'],
            anomalies['Average Medicare Payment Amount'],
            color='red', label='Anomaly', alpha=0.7)
plt.title('Anomaly Detection: Number of Services vs Average Medicare Payment Amount')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))

plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Medicare Beneficiaries'],
            data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Allowed Amount'],
            label='Normal', alpha=0.5)
plt.scatter(anomalies['Number of Medicare Beneficiaries'],
            anomalies['Average Medicare Allowed Amount'],
            color='red', label='Anomaly', alpha=0.7)
plt.title('Anomaly Detection: Number of Medicare Beneficiaries vs Average Medicare Allowed Amount')
plt.xlabel('Number of Medicare Beneficiaries')
plt.ylabel('Average Medicare Allowed Amount')
plt.legend()
plt.grid()

plt.show()
print("Anomalies detected:")
print(anomalies)
print(f"Total number of anomalies: {anomalies.shape[0]}")
plt.savefig('Randomised Search.png')  

#OPTUNA OPTIMISATION CODE 

import optuna
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(numeric_data, test_size=0.2, random_state=42)


def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_samples = trial.suggest_float("max_samples", 0.1, 1.0)
    max_features = trial.suggest_float("max_features", 0.5, 1.0)
    contamination = trial.suggest_float("contamination", 0.001, 0.1)


    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        contamination=contamination,
        random_state=42
    )

    model.fit(X_train)
    y_pred = model.predict(X_train)
    y_pred = np.where(y_pred == -1, 1, 0) 


    outlier_count = np.sum(y_pred == 1)  
    
    return -outlier_count  


study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best parameters found:", best_params)


#Checking with isolation forest


best_model = IsolationForest(
    n_estimators=best_params['n_estimators'],
    max_samples=best_params['max_samples'],
    max_features=best_params['max_features'],
    contamination=best_params['contamination'],
    random_state=42
)


best_model.fit(numeric_data)


outliers = best_model.predict(numeric_data)
outlier_count = np.sum(outliers == -1)  
print("Number of outliers detected:", outlier_count)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])


iso_forest = IsolationForest(n_estimators=259, contamination=0.09, random_state=42)


iso_forest.fit(numeric_data)


data_dropped['Anomaly'] = iso_forest.predict(numeric_data)


anomalies = data_dropped[data_dropped['Anomaly'] == -1]


plt.figure(figsize=(12, 6))


plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Services'],
            data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Payment Amount'],
            label='Normal', alpha=0.5)

plt.scatter(anomalies['Number of Services'],
            anomalies['Average Medicare Payment Amount'],
            color='red', label='Anomaly', alpha=0.7)
plt.title('Anomaly Detection: Number of Services vs Average Medicare Payment Amount')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))

plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Medicare Beneficiaries'],
            data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Allowed Amount'],
            label='Normal', alpha=0.5)
plt.scatter(anomalies['Number of Medicare Beneficiaries'],
            anomalies['Average Medicare Allowed Amount'],
            color='red', label='Anomaly', alpha=0.7)
plt.title('Anomaly Detection: Number of Medicare Beneficiaries vs Average Medicare Allowed Amount')
plt.xlabel('Number of Medicare Beneficiaries')
plt.ylabel('Average Medicare Allowed Amount')
plt.legend()
plt.grid()
plt.savefig('Optuna.png')  
plt.show()
print("Anomalies detected:")
print(anomalies)
print(f"Total number of anomalies: {anomalies.shape[0]}")


#MANUAL SEARCH CODE

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest



n_estimators = [50, 100, 200, 300]
max_samples = [0.1, 0.5, 0.75, 1.0]
max_features = [0.5, 0.75, 1.0]
contamination = [0.001, 0.01, 0.1]


results = []


for n in n_estimators:
    for samples in max_samples:
        for features in max_features:
            for cont in contamination:
                
                model = IsolationForest(n_estimators=n,
                                        max_samples=samples,
                                        max_features=features,
                                        contamination=cont,
                                        random_state=42)

                
                model.fit(numeric_data)

                
                outlier_predictions = model.predict(numeric_data)

                
                num_outliers = np.sum(outlier_predictions == -1)

                
                results.append((n, samples, features, cont, num_outliers))

                
                print(f"Parameters: {n}, {samples}, {features}, {cont} => Number of outliers detected: {num_outliers}")


results_df = pd.DataFrame(results, columns=['n_estimators', 'max_samples', 'max_features', 'contamination', 'num_outliers'])

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])


iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42 , max_samples = 0.5 ,  max_features=0.5)


iso_forest.fit(numeric_data)


data_dropped['Anomaly'] = iso_forest.predict(numeric_data)


anomalies = data_dropped[data_dropped['Anomaly'] == -1]


plt.figure(figsize=(12, 6))


plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Services'],
            data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Payment Amount'],
            label='Normal', alpha=0.5)

plt.scatter(anomalies['Number of Services'],
            anomalies['Average Medicare Payment Amount'],
            color='red', label='Anomaly', alpha=0.7)
plt.title('Anomaly Detection: Number of Services vs Average Medicare Payment Amount')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))

plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Medicare Beneficiaries'],
            data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Allowed Amount'],
            label='Normal', alpha=0.5)
plt.scatter(anomalies['Number of Medicare Beneficiaries'],
            anomalies['Average Medicare Allowed Amount'],
            color='red', label='Anomaly', alpha=0.7)
plt.title('Anomaly Detection: Number of Medicare Beneficiaries vs Average Medicare Allowed Amount')
plt.xlabel('Number of Medicare Beneficiaries')
plt.ylabel('Average Medicare Allowed Amount')
plt.legend()
plt.grid()
plt.savefig('Manual Search.png')
plt.show()
print("Anomalies detected:")
print(anomalies)
print(f"Total number of anomalies: {anomalies.shape[0]}")


#A General OVERVIEW OF ALL THE METHODS USED WTH ISOLATION FOREST


import matplotlib.pyplot as plt


methods = ['Bayesian Search', 'Random Search', 'Optuna Optimization', 'Manual Search']
outliers_detected = [561, 561, 5046, 5603] 


plt.figure(figsize=(10, 6))
plt.bar(methods, outliers_detected, color=['blue', 'orange', 'green', 'red'])


plt.title('Number of Outliers Detected by Different Methods', fontsize=14)
plt.xlabel('Detection Methods', fontsize=12)
plt.ylabel('Number of Outliers Detected', fontsize=12)
plt.xticks(rotation=10)  


plt.grid(axis='y')
plt.tight_layout()
plt.savefig('All Hypertuning methods outlierd detected.png')
plt.show()


#MEAN ANOMALY SCORE COMPARISION

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


mean_anomaly_scores = {}


hyperparameter_sets = {
    "Bayesian Search": {"n_estimators": 50, "contamination": 0.01, "max_samples": 0.5, "max_features": 0.75},
    "Random Search": {"n_estimators": 200, "contamination": 0.01, "max_samples": 0.5, "max_features": 0.75},
    "Optuna Optimization": {"n_estimators": 259, "contamination": 0.0999, "max_samples": 0.4699, "max_features": 0.8071},
    "Manual Search": {"n_estimators": 100, "contamination": 0.01, "max_samples": 0.5, "max_features": 0.5}
}

for method, params in hyperparameter_sets.items():
    
    iso_forest = IsolationForest(
        n_estimators=params["n_estimators"],
        contamination=params["contamination"],
        max_samples=params["max_samples"],
        max_features=params["max_features"],
        random_state=42
    )
    iso_forest.fit(numeric_data)
    

    data_dropped['Anomaly'] = iso_forest.predict(numeric_data)
    data_dropped['Anomaly_Score'] = iso_forest.decision_function(numeric_data)
    outliers_scores = data_dropped[data_dropped['Anomaly'] == -1]['Anomaly_Score']
    
    
    mean_anomaly_scores[method] = outliers_scores.mean()


plt.figure(figsize=(10, 6))
plt.bar(mean_anomaly_scores.keys(), mean_anomaly_scores.values(), color=['blue', 'orange', 'green', 'red'])
plt.title("Mean Anomaly Score Comparison across Hyperparameter Tuning Methods")
plt.xlabel("Hyperparameter Tuning Methods")
plt.ylabel("Mean Anomaly Score for Outliers")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig('Mean anomaly comparision.png')
plt.show()


print("Mean Anomaly Scores for Outliers by Method:")
for method, score in mean_anomaly_scores.items():
    print(f"{method}: {score:.4f}")







