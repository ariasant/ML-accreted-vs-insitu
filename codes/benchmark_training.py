import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def plot_training_loss(training_results, filename=None):
    validation_loss = np.array(training_results['val_loss'])
    training_loss = np.array(training_results['loss'])

    v_max = np.median(validation_loss)+3*np.std(validation_loss)
    l_max = np.median(training_loss)+3*np.std(training_loss)
    v_min = np.min(validation_loss)
    l_min = np.min(training_loss)
    y_max = np.max([v_max, l_max])
    y_min = np.min([v_min,l_min])

    fig,ax = plt.subplots(layout='constrained')

    ax.plot(np.arange(1,len(training_loss)+1), training_loss, color='red', label='Training Loss')
    ax.plot(np.arange(1,len(validation_loss)+1), validation_loss, color='blue', label='Validation Loss')
    fig.legend()
    plt.grid()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Binary Cross-entropy Loss')
    ax.set_ylim(ymin=y_min, ymax=y_max)
    if (filename!=None):
        try: 
            fig.savefig(path_plots+'ml_training/'+filename+'_overfitting_plot.png',dpi=400)
        except:
             fig.savefig(filename+'_overfitting_plot.png',dpi=400)
    plt.close()

def GANN(output_bias=None, input_parameters=4):
    """
    Artificial neual network for classifying halo stars as "in-situ" or accreted" as described in 
    Tronrud et al 2022 (10.1093/mnras/stac2027).
    """
    
    initializer = tf.keras.initializers.lecun_normal()

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    GANN = tf.keras.Sequential([tf.keras.layers.BatchNormalization(input_shape=(input_parameters,)),
                                tf.keras.layers.Dense(64, activation='selu', input_shape=(input_parameters,),
                                                      kernel_initializer=tf.keras.initializers.lecun_normal()),
                                tf.keras.layers.Dense(256, activation='selu', 
                                                      kernel_initializer=tf.keras.initializers.lecun_normal()),
                                tf.keras.layers.Dense(64, activation='selu', 
                                                      kernel_initializer=tf.keras.initializers.lecun_normal()),
                                tf.keras.layers.Dense(32, activation='selu', 
                                                      kernel_initializer=tf.keras.initializers.lecun_normal()),
                                tf.keras.layers.Dense(1, activation='sigmoid', 
                                                      kernel_initializer=tf.keras.initializers.lecun_normal(),
                                                      bias_initializer=output_bias)],
                               name="GANN"
    )


    return GANN

####
# Create datasets
####
# In units of solar masses

test_halos = [6,21]
train_halos = [16, 23, 24, 27]

features = ['age_Gyr','aFe','FeH']

train_ds = np.zeros((1,len(features)))
val_ds = np.zeros((1,len(features)))
test_ds = np.zeros((1,len(features)))


train_labels_ds = np.zeros((1))
val_labels_ds = np.zeros((1))
test_labels_ds = np.zeros((1))

for halo in test_halos:

    # Pre-process data
    df = pd.read_pickle('/mnt/data1/users/ariasant/data/auriga/dataframes/G{:02}_df.pkl'.format(halo))
    df = df[df['r_cylindrical']<=np.sqrt(2)*50]
    df = df[np.abs(df['z_kpc'])<50]
    df = df[(df['FeH']>=-4) & (df['FeH']<1.5)]
    df = df[(df['aFe']<1.5) & (df['aFe']>-0.5)]
    # Ignore stars in exitisting subhalo
    df = df[df['insitu_flag']!=-1]

    print(df['insitu_flag'].unique())

    # Turn abundances into linear form 
    df['FeH'] = np.power(10,df['FeH'])
    df['aFe'] = np.power(10,df['aFe'])
    df.index = range(len(df.index))

    test_data = df[features].values
    test_labels = df['insitu_flag'].values

    test_ds = np.vstack((test_ds,test_data))
    test_labels_ds = np.append(test_labels_ds,test_labels)

    print('Halo G{:02} done'.format(halo))

for halo in train_halos:

    # Pre-process data
    df = pd.read_pickle('/mnt/data1/users/ariasant/data/auriga/dataframes/G{:02}_df.pkl'.format(halo))
    df = df[df['r_cylindrical']<=np.sqrt(2)*50]
    df = df[np.abs(df['z_kpc'])<50]
    df = df[(df['FeH']>=-4) & (df['FeH']<1.5)]
    df = df[(df['aFe']<1.5) & (df['aFe']>-0.5)]

    df = df[df['insitu_flag']!=-1]

    print(df['insitu_flag'].unique())

    # Turn abundances into linear form
    df['FeH'] = np.power(10,df['FeH'])
    df['aFe'] = np.power(10,df['aFe'])
    df.index = range(len(df.index))

    data = df[features].values
    labels = df['insitu_flag'].values 

    # Identify accreted stars
    idx_accreted = np.where(labels==1)[0]
    idx_insitu = np.delete(np.arange(data.shape[0]), idx_accreted)

    # Sample an equal number of in-situ stars to the accreted
    idx_insitu = np.random.choice(idx_insitu, size=len(idx_accreted))

    data = data[np.append(idx_accreted, idx_insitu)]
    labels = labels[np.append(idx_accreted, idx_insitu)]

    print('Accreted fraction in training set: {:.2f}'.format(len(labels[labels==1])/len(labels)))

    # Select 20% of the stars in each galaxy to go to validation set
    train_data, val_data, train_labels, val_labels = train_test_split(data,labels, test_size=0.2, random_state=42)

    print('Accreted fraction in training set: {:.2f}'.format(len(train_labels[train_labels==1])/len(train_labels)))
    print('Accreted fraction in validation set: {:.2f}'.format(len(val_labels[val_labels==1])/len(val_labels)))

    train_ds = np.vstack((train_ds,train_data))
    val_ds = np.vstack((val_ds,val_data))

    train_labels_ds = np.append(train_labels_ds,train_labels)
    val_labels_ds = np.append(val_labels_ds,val_labels)

    print('Halo G{:02} done'.format(halo))

train_data = train_ds[1:]
val_data = val_ds[1:]
test_data = test_ds[1:]

train_labels = train_labels_ds[1:]
val_labels = val_labels_ds[1:]
test_labels =  test_labels_ds[1:]


batchsize=1024
train_ds = tf.data.Dataset.from_tensor_slices((train_data.astype('float32'),
                                               train_labels)).shuffle(5000000).batch(batchsize)
val_ds = tf.data.Dataset.from_tensor_slices((val_data.astype('float32'),
                                            val_labels)).shuffle(5000000).batch(batchsize)

####
# Create model
####
model =  GANN(input_parameters=train_data.shape[1])
model.summary()
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.01),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_delta=0.001, min_lr=1e-5)
checkpoint_filepath = '/mnt/data1/users/ariasant/'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True)

####
# Start training
####
accreted_fraction_train = len(train_labels[train_labels==1])/len(train_labels)
print('Accreted Fraction Train Set: {:.2f}'.format(accreted_fraction_train))
accreted_fraction_val = len(val_labels[val_labels==1])/len(val_labels)
print('Accreted Fraction Val Set: {:.2f}'.format(accreted_fraction_val))

training_history = model.fit(train_ds,
                             validation_data=val_ds,
                             epochs=100,
                             callbacks=[reduce_lr
                             ]
)

# Save model and training outputs
filename = 'benchmark_AURIGA'
model.save_weights(filename+'.h5')

training_results = training_history.history

np.savez(filename, loss=np.array(training_results['loss']), val_loss=np.array(training_results['val_loss']))

plot_training_loss(training_results,filename)

# Make predictions on test and validation data
predictions = model.predict(test_data, batch_size=test_data.shape[0])
np.save('test_pred_'+filename, predictions)

predictions = model.predict(val_data, batch_size=val_data.shape[0])
np.save('val_pred_'+filename, predictions)


