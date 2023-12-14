import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import tensorflow as tf


train_file = np.load('/mnt/data1/users/ariasant/data/train_ds_trans_mod_GANN.npz')
val_file = np.load('/mnt/data1/users/ariasant/data/val_ds_trans_mod_GANN.npz')

print('Data Loaded',flush=True)

train_data = train_file['data']
val_data = val_file['data']

train_labels = train_file['labels']
val_labels = val_file['labels']


# Create datasets
batchsize=1024
train_ds = tf.data.Dataset.from_tensor_slices((train_data.astype('float32'),
                                               train_labels.astype('float32'))).shuffle(5000000).batch(batchsize)
val_ds = tf.data.Dataset.from_tensor_slices((val_data.astype('float32'),
                                             val_labels.astype('float32'))).shuffle(5000000).batch(batchsize)


tf.config.run_functions_eagerly(True)


def objective(trial):
    
    n_neurons = trial.suggest_int("neurons",1,100)
    dropout_rate = trial.suggest_float("dropout",0.1,0.7)
    l1_reg = trial.suggest_float("l1_reg",0.001, 0.1)

    model = tf.keras.Sequential([tf.keras.layers.Dense(n_neurons,
                                                       activation='selu',
                                                       kernel_regularizer=tf.keras.regularizers.L1(l1_reg)),
                                 tf.keras.layers.Dropout(dropout_rate),
                                 tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer = tf.keras.optimizers.legacy.Adam(0.01),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    model.fit(train_ds, epochs=3, verbose=1)

    predictions = model.predict(val_data, batch_size=val_data.shape[0])

    precision, recall, thresholds = precision_recall_curve(val_labels, predictions, pos_label=1)
    AUC = auc(recall,precision)

    return AUC

print('Transfomational Machine Learning using GANN as base model.', flush=True)
study = optuna.create_study(direction='maximize')
study.optimize(objective,n_trials=50,n_jobs=1, show_progress_bar=True)
print(study.best_params, flush=True)
print('Best AUC: {:.2f}'.format(study.best_value), flush=True)
