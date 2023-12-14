import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve
from xgboost import XGBClassifier


train_file = np.load('/mnt/data1/users/ariasant/data/train_ds.npz')
test_file = np.load('/mnt/data1/users/ariasant/data/val_ds.npz')

X_train = train_file['data']
Y_train = train_file['labels']

X_test = test_file['data']
Y_test = test_file['labels']

print('Data loaded', flush=True)

def objective(trial):
    
    n_estimators = trial.suggest_int("n_trees",100,1000)
    lr = trial.suggest_float("lr",0.005,0.1)
    max_depth = trial.suggest_int("max_depth",2,8)

    model = XGBClassifier(n_estimators=n_estimators,
                          learning_rate=lr,
                          max_depth=max_depth,
                          n_jobs=-1
    )
                        

    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)

    precision,recall,_ = precision_recall_curve(Y_test, predictions,pos_label=1)

    AUC = auc(recall,precision)

    return AUC


study = optuna.create_study(direction='maximize')
study.optimize(objective,n_trials=50,n_jobs=1, show_progress_bar=True)
print(study.best_params, flush=True)
print('Best AUC: {:.2f}'.format(study.best_value), flush=True)
