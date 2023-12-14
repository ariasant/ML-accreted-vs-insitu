import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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


#Load data
train_file = np.load('/mnt/data1/users/ariasant/data/train_ds.npz')
val_file = np.load('/mnt/data1/users/ariasant/data/val_ds.npz')

print('Data Loaded',flush=True)
train_data = train_file['data']
train_labels = train_file['labels']

val_data = val_file['data']
val_labels = val_file['labels']

#create datasets
batchsize=64

train_ds = tf.data.Dataset.from_tensor_slices((train_data.astype('float32'),
                                               train_labels)).shuffle(5000000).batch(batchsize)
val_ds = tf.data.Dataset.from_tensor_slices((val_data.astype('float32'),
                                            val_labels)).shuffle(5000000).batch(batchsize)

print('Training started',flush=True)
#create model
model =  tf.keras.Sequential([tf.keras.layers.Dense(100, activation='selu', kernel_regularizer=tf.keras.regularizers.L2()),
                              tf.keras.layers.Dense(1, activation='sigmoid')])
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.01),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)
model.build(input_shape=(None,train_data.shape[1]))
model.summary()

#define callbacks
epochs=100
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs)
reduce_lr = update_lr_on_plateau(lr_patience=5,
                                 min_delta=0.005,
                                 training_patience=epochs,
                                 min_lr=0.0001
                                 )

checkpoint_filepath = '/mnt/data1/users/ariasant/'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True)

#Start training
accreted_fraction_train = len(train_labels[train_labels==1])/len(train_labels)
print('Accreted Fraction Train Set: {:.2f}'.format(accreted_fraction_train))
accreted_fraction_val = len(val_labels[val_labels==1])/len(val_labels)
print('Accreted Fraction Val Set: {:.2f}'.format(accreted_fraction_val))


training_history = model.fit(train_ds,
                             validation_data=val_ds,
                             epochs=epochs,
                             callbacks=[early_stopping,
                                        reduce_lr,
                                        checkpoint_callback
                             ]
)

#Save model and training outputs
filename = 'MLP'
model.save_weights(filename+'.h5')

training_results = training_history.history

np.savez(filename, loss=np.array(training_results['loss']), val_loss=np.array(training_results['val_loss']))

plot_training_loss(training_results,filename)

#Load test data
test_file = np.load('/mnt/data1/users/ariasant/data/test_ds.npz')
test_data = test_file['data']

predictions = model.predict(test_data, batch_size=test_data.shape[0])
np.save('test_pred_'+filename, predictions)

predictions = model.predict(val_data, batch_size=val_data.shape[0])
np.save('val_pred_'+filename, predictions)


