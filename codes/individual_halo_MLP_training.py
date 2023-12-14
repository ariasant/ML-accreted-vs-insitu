import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
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

def plot_pr(training_results, filename=None):

    keys = list(training_results.keys())
    tr_prec = training_results[keys[1]]
    tr_rec =  training_results[keys[2]]
    val_prec = training_results[keys[4]]
    val_rec = training_results[keys[5]]

    x = np.arange(1,len(tr_prec)+1)
    fig,ax = plt.subplots(layout='constrained')
    ax.plot(x,tr_prec, color='red', label='Training Precision')
    ax.plot(x, tr_rec, color='red', ls=':', label='Training Recall')
    ax.plot(x, val_prec, color='blue', label='Validation Precision')
    ax.plot(x, val_rec, color='blue', ls =':', label='Validation Recall')

    ax.set_xlabel('Epochs')

    ax.set_ylim([0.5,1])
    fig.legend()

    if (filename!=None):
        try:
            fig.savefig(path_plots+'ml_training/'+filename+'_overfitting_plot.png',dpi=400)
        except:
             fig.savefig(filename+'_overfitting_plot.png',dpi=400)
    plt.close('all')
    
def training(train_data, train_labels, val_data, val_labels, halo_number):

    
    model = tf.keras.Sequential([tf.keras.layers.Dense(100, activation='selu'),
                                 tf.keras.layers.Dense(1, activation='sigmoid')])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(lr),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()]
    )

    print('Training started for G{:02}'.format(halo_number))                                            
    training_history = model.fit(
        x = train_data, y = train_labels,
        batch_size = batch_size,
        validation_data = (val_data,val_labels),
        epochs = epochs,
        callbacks=[reduce_learning_rate,early_stopping],
        verbose=1
    )
    print('End of training for G{:02}'.format(halo_number))

    model.save_weights('/mnt/data1/users/ariasant/weights/individual_halos/G{:02}_MLP_nn_model_weights.h5'.format(halo_number))
    print('Model saved for the NN trained on G{:02} data'.format(halo_number))

    training_results = training_history.history
    np.save('/mnt/data1/users/ariasant/training_results/individual_halos/G{:02}_MLP_training_results'.format(halo_number), training_results)

    #Check if the model is overfitting
    plot_training_loss(training_results=training_results,
                       filename='/mnt/data1/users/ariasant/training_results/individual_halos/G{:02}_MLP_nn_overfitting_plot'.format(halo_number)
    )

    plot_pr(training_results=training_results,
                       filename='/mnt/data1/users/ariasant/training_results/individual_halos/G{:02}_MLP_nn_pr_plot'.format(halo_number)
    )

    return model




#################################################################################

#Training parameters
lr = 0.01
epochs = 100
batch_size = 128

train_halos = [1,15,17,18,19,23,24,25,27,38,40,44]

# Create customised callback for training
reduce_learning_rate = update_lr_on_plateau(lr_patience=5,
                                            min_delta=0.001,
                                            training_patience=epochs,
                                            min_lr=0.00001
                                            )

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


# Load training data
train_file = np.load('/mnt/data1/users/ariasant/data/train_ds.npz')
train_domains =  train_file['domains']
train_data = train_file['data']
train_labels = train_file['labels']

# Find how many stars are in the train halos
train_halo_sizes = [len(train_domains[train_domains==domain]) for domain in np.arange(train_domains.max()+1)]

# Split train dataset into train halos
indices = [np.sum(train_halo_sizes[:i]) for i in range(1,len(train_halo_sizes))]
train_data_ls = np.split(train_data, indices)
train_labels_ls =  np.split(train_labels, indices)

#-----
#Repeat for validation data
#-----
# Load training data
val_file = np.load('/mnt/data1/users/ariasant/data/val_ds.npz')
val_domains = val_file['domains']
val_data = val_file['data']
val_labels = val_file['labels']

# Find how many stars are in the train halos
val_halo_sizes = [len(val_domains[val_domains==domain]) for domain in np.arange(val_domains.max()+1)]

# Split train dataset into train halos
indices = [np.sum(val_halo_sizes[:i]) for i in range(1,len(val_halo_sizes))]
val_data_ls = np.split(val_data, indices)
val_labels_ls =  np.split(val_labels, indices)


output_file = open('MLP_table.csv', 'w')
header = ",".join(['Train halo'] + ['G{:02}'.format(halo) for halo in train_halos] + ['\n'])
output_file.write(header)

for i,halo_number in enumerate(train_halos):

    #if rank==i:
    trained_model = training(train_data_ls[i],
                             train_labels_ls[i],
                             val_data_ls[i],
                             val_labels_ls[i],
                             halo_number)
    
    # Define evaluation dataset such that all the stars are considered for the halos
    # the model was not trained on and only validation data is considered fot the halo it was
    data_ls = [np.vstack((val_data_ls[it], train_data_ls[it])) for it in range(len(train_halos)) if it!=i]
    labels_ls = [np.append(val_labels_ls[it], train_labels_ls[it]) for it in range(len(train_halos)) if it!=i]
    data_ls.insert(i, val_data_ls[i])
    labels_ls.insert(i, val_labels_ls[i])

    # Evaluate model on validation data 
    AUC_ls  = []

    for it,data in enumerate(data_ls):

        predictions = trained_model.predict(data, batch_size=data.shape[0])

        precision, recall, _ = precision_recall_curve(labels_ls[it], predictions, pos_label=1) 
        AUC_ls.append( auc(recall,precision) )

    # Write scores into file
    scores = ",".join(['G{:02}'.format(halo_number)] + ['{:.2f}'.format(AUC) for AUC in AUC_ls] + ['\n'])
    output_file.write(scores)

    print('Model trained on G{:02} done.'.format(halo_number))

output_file.close()


    

    
    
