import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def virial_radius(virial_mass):

    H2 = np.power(70,2) #kms /Mpc
    G = 4.3009172706e-9 #Mpc M_sun^-1 kms ^2 
    
    r_virial3 = (virial_mass * G) / (100 * H2)

    return np.power(r_virial3,1/3) * 1000 #kpc

def epsilon(df):
    """
    Returns the circularity parameters of the stars in df.
    """
    r = np.sqrt(df['x_kpc']**2 + df['y_kpc']**2 + df['z_kpc']**2)
    v = np.sqrt(df['vx_kms']**2 + df['vy_kms']**2 + df['vz_kms']**2)
    
    Jz = df['x_kpc']*df['vy_kms'] - df['y_kpc']*df['vx_kms']
    J = r*v
    
    return np.array(Jz/J)
    
def select_accreted_and_insitu_stars(df,halo):

    df['v'] = np.sqrt(df['vx_kms']**2+df['vy_kms']**2+df['vz_kms']**2)
    r = np.sqrt(df['x_kpc']**2 + df['y_kpc']**2 + df['z_kpc']**2).values

    # Select accreted stars
    idx_accreted = np.where(r>25)[0]

    # Select in-situ stars based on kinematics
    r_bulge = 2 # assumption
    r_200 = virial_radius(virial_mass[halo])
    eps = epsilon(df)
    idx_insitu = np.where( (r>r_bulge) & (r<0.15*r_200) &
                           ((df['z_kpc']<10) & (df['z_kpc']>-10)) &
                           (eps>0.4) )[0]
    # Select an equal number of in-situ and accreted stars
    idx = np.random.randint(low=0, high=len(idx_insitu), size=len(idx_accreted))
    idx_insitu = idx_insitu[idx]

    # See what fraction are actually in-situ/accreted stars
    check = df.loc[idx_insitu,'insitu_flag'].values
    print('Purity of in-situ sample: {:.2f}'.format(len(check[check==0])/len(check)))

    check = df.loc[idx_accreted,'insitu_flag'].values
    print('Purity of accreted sample: {:.2f}'.format(len(check[check==0])/len(check)))

    # Define data and labels
    data = df.loc[np.append(idx_insitu,idx_accreted),features].values
    labels = np.append(np.zeros(len(idx_insitu)),np.ones(len(idx_accreted)))

    return data,labels

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
virial_mass = {1: 11.9e11,
               3: 17.01e11,
               8: 16.31e11,
               13: 11.69e11,
               16: 12.69e11,
               17: 11.69e11,
               20: 10.58e11,
               22: 10.08e11,
               24: 10.29e11,
               25: 9.12e11,
               26: 8.96e11,
               27: 7.96e11,
               28: 7.67e11,
               30: 8.08e11,
               31: 8.32e11,
               33: 7.80e11,
               37: 6.66e11,
               39: 7.48e11,
               15: 11.22e11,
               18: 9.68e11,
               23: 9.95e11,
               29: 8.82e11,
               34: 7.89e11,
               38: 7.14e11,
               40: 7.57e11,
               42: 7.18e11,
               19: 9.62e11
}

test_halos = [29,30,34,42]
train_halos = [1,15,17,18,19,23,24,25,27,38,40]

features = ['age_Gyr','aFe','FeH']

train_ds = np.zeros((1,len(features)))
val_ds = np.zeros((1,len(features)))
test_ds = np.zeros((1,len(features)))


train_labels_ds = np.zeros((1))
val_labels_ds = np.zeros((1))
test_labels_ds = np.zeros((1))

for halo in test_halos:

    # Pre-process data
    df = pd.read_pickle('/mnt/data1/users/ariasant/data/dataframes/G{:02}_df.pkl'.format(halo))
    df = df[df['r_cylindrical']<=np.sqrt(2)*50]
    df = df[np.abs(df['z_kpc'])<50]
    df = df[(df['FeH']>=-4) & (df['FeH']<1.5)]
    df = df[(df['aFe']<1.5) & (df['aFe']>-0.5)]

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
    df = pd.read_pickle('/mnt/data1/users/ariasant/data/dataframes/G{:02}_df.pkl'.format(halo))
    df = df[df['r_cylindrical']<=np.sqrt(2)*50]
    df = df[np.abs(df['z_kpc'])<50]
    df = df[(df['FeH']>=-4) & (df['FeH']<1.5)]
    df = df[(df['aFe']<1.5) & (df['aFe']>-0.5)]

    # Turn abundances into linear form
    df['FeH'] = np.power(10,df['FeH'])
    df['aFe'] = np.power(10,df['aFe'])
    df.index = range(len(df.index))

    data,labels = select_accreted_and_insitu_stars(df,halo)

    # Select 20% of the stars in each galaxy to go to validation set
    train_data, val_data, train_labels, val_labels = train_test_split(data,labels, test_size=0.2, random_state=42)

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
                             callbacks=[early_stopping,
                                        reduce_lr,
                                        checkpoint_callback
                             ]
)

# Save model and training outputs
filename = 'GANN_ARTEMIS'
model.save_weights(filename+'.h5')

training_results = training_history.history

np.savez(filename, loss=np.array(training_results['loss']), val_loss=np.array(training_results['val_loss']))

plot_training_loss(training_results,filename)

# Make predictions on test and validation data
predictions = model.predict(test_data, batch_size=test_data.shape[0])
np.save('test_pred_'+filename, predictions)

predictions = model.predict(val_data, batch_size=val_data.shape[0])
np.save('val_pred_'+filename, predictions)


