import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf


def get_predictive_description(data,
                               domains=None,
                               mask=False
):

    prediction_list = [model.predict(data, batch_size=data.shape[0])[:,0] for model in model_ensemble]
        
    predictions = np.vstack([prediction_list]).T

    if mask:

        # Create a mask for the predictions of the base models trained on the star domains
        mask = np.ones((len(domains),len(prediction_list)))
        mask[range(len(domains)),domains.astype('int')] = False 
        # Apply mask so that the prediction from the base model trained on the domain where
        # the star was taken is equal to 0
        predictions = np.where(mask, predictions, 0.0)

    return predictions

#===================================================

class update_lr_on_plateau(tf.keras.callbacks.Callback):
    """
    Everytime the learning rate is updated by multiplying the current learning
    rate by the value specified by "factor"
    The training ends if there is no significant improvement of the loss function
    for the numbe of epochs defined by "training_patience".
    -------------------------------------------------------
    Training patience updates everytime there is a significant (>min_delta)
    decrease of the loss function. 
    Learning rate patience updates everytime the learning rate is changed.
    
    """

    def __init__(self, lr_patience=2, min_delta=0.1, factor=0.5, min_lr=1e-6, monitor='loss',
                training_patience=5):
        
        self.previous_loss = np.inf
        self.delta = 0
        self.reset_lr_patience = lr_patience
        self.reset_training_patience = training_patience
        #Percentage of the previous loss function 
        self.min_delta = min_delta
        self.factor = factor
        self.min_lr = min_lr
        self.monitor = monitor
        #Number of epochs to wait before ending the training
        self.training_patience = training_patience
        #Number of epochs to wait before updating learning rate
        self.lr_patience = lr_patience
    
        
    def on_epoch_end(self,epoch,logs=None):   
        #Define new min delta
        min_delta = self.min_delta*self.previous_loss
        current_loss = float(logs[self.monitor])
        self.delta = current_loss - self.previous_loss
        
        self.previous_loss = current_loss
        
        
        if (self.delta>-min_delta):
            self.lr_patience-=1
            self.training_patience-=1
        
        if (self.delta<-min_delta):
            self.training_patience = self.reset_training_patience

        if (self.training_patience==0):
            self.model.stop_training = True
            
    def on_epoch_begin(self, epoch, logs=None):
        
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print("Learning rate is {:2g}.".format(lr))

        if (self.lr_patience==0):

            # Get the current learning rate from model's optimizer.
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            new_lr = lr*self.factor
            if (new_lr>self.min_lr):
                # Set the value back to the optimizer before this epoch starts
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                self.lr_patience=self.reset_lr_patience
            else:
                tf.keras.backend.set_value(self.model.optimizer.lr, self.min_lr)
                self.lr_patience=self.reset_lr_patience

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

#===================

tf.config.run_functions_eagerly(True)

# Load data
train_file = np.load('/mnt/data1/users/ariasant/data/auriga/train_ds.npz')
val_file = np.load('/mnt/data1/users/ariasant/data/auriga/val_ds.npz')
test_file = np.load('/mnt/data1/users/ariasant/data/auriga/test_ds.npz')

train_data = train_file['data']
train_labels = train_file['labels']
train_domains = train_file['domains']

val_data = val_file['data']
val_labels = val_file['labels']

test_data = test_file['data']
test_labels = test_file['labels']

# Create ensemble of MLP models trained on the training galaxies
train_halos = [1,15,17,18,19,23,24,25,27,38,40,44]

weights_list = ['/mnt/data1/users/ariasant/weights/individual_halos/G{:02}_MLP_nn_model_weights.h5'.format(halo) for halo in train_halos]
model_ensemble = []

for weights in weights_list:

    model = tf.keras.Sequential([tf.keras.layers.Dense(100, activation='selu'),
                                 tf.keras.layers.Dense(1, activation='sigmoid')])
    model.build(input_shape=(None,8))
    model.load_weights(weights)
    model_ensemble.append(model)

# Get a description of each star in the training set based on the predictions of
# the MLP in the ensemble
train_data = get_predictive_description(train_data, domains=train_domains, mask=True)
val_data = get_predictive_description(val_data)
test_data = get_predictive_description(test_data)


# Create datasets
batchsize=1024
train_ds = tf.data.Dataset.from_tensor_slices((train_data.astype('float32'),
                                               train_labels.astype('float32'))).shuffle(5000000).batch(batchsize)
val_ds = tf.data.Dataset.from_tensor_slices((val_data.astype('float32'),
                                             val_labels.astype('float32'))).shuffle(5000000).batch(batchsize)


# Initialise model for MLP base models
model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='selu', kernel_regularizer=tf.keras.regularizers.L2()),
                             tf.keras.layers.Dense(1, activation='sigmoid')
                             ])

epochs=100

model.compile(
    optimizer = tf.keras.optimizers.legacy.Adam(0.01),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = update_lr_on_plateau(lr_patience=5,
                                 min_delta=0.005,
                                 min_lr=0.00001,
                                 training_patience=epochs
                                 )
checkpoint_filepath = '/mnt/data1/users/ariasant/'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='weights',
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True)


# Start training
print('Training started',flush=True)
training_history = model.fit(train_ds,
                             validation_data=val_ds,
                             epochs=epochs,
                             callbacks=[early_stopping,
                                        reduce_lr,
                                        checkpoint_callback
                             ]                             
)

filename = 'TransMod_MLP'

# Plot and save training results
training_results = training_history.history
np.savez(filename, loss=np.array(training_results['loss']), val_loss=np.array(training_results['val_loss']))
plot_training_loss(training_results,filename)

# Get predictions on test and validation data
val_pred = model.predict(val_data, batch_size=val_data.shape[0])
test_pred = model.predict(test_data, batch_size=test_data.shape[0])

# Save predictions
np.save('val_pred_'+filename, val_pred)
np.save('test_pred_'+filename, test_pred)

# Save model
model.save_weights(filename+'.h5')

