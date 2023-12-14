import joblib
import numpy as np
import pickle
import tensorflow as tf

tf.config.run_functions_eagerly(True)

class MLP_ARTEMIS(tf.keras.Model):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        # Define base architecture
        self.MLP = tf.keras.Sequential([tf.keras.layers.Dense(100, activation='selu', kernel_regularizer=tf.keras.regularizers.L2()),
                                    tf.keras.layers.Dense(1, activation='sigmoid')])

        # Load weights
        self.MLP.load_weights('model_weights/MLP_ARTEMIS.h5')

    def call(self, inputs, training=False):

        # Normalize features in data into the range seen by the models during training
        scaler = joblib.load('model_weights/scaler_ARTEMIS.save')
        inputs = tf.convert_to_tensor(scaler.transform(inputs.numpy()), np.float32)

        label = self.MLP(inputs)


class MLP_Auriga(tf.keras.Model):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        # Define base architecture
        self.MLP = tf.keras.Sequential([tf.keras.layers.Dense(100, activation='selu', kernel_regularizer=tf.keras.regularizers.L2()),
                                    tf.keras.layers.Dense(1, activation='sigmoid')])

        # Load weights
        self.MLP.load_weights('model_weights/MLP_Auriga.h5')

    def call(self, inputs, training=False):

        # Normalize features in data into the range seen by the models during training
        scaler = joblib.load('model_weights/scaler_Auriga.save')
        inputs = tf.convert_to_tensor(scaler.transform(inputs.numpy()), np.float32)

        label = self.MLP(inputs)


class MLP_galaxy_features_ARTEMIS(tf.keras.Model):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        # Define base architecture
        self.MLP = tf.keras.Sequential([tf.keras.layers.Dense(100, activation='selu', kernel_regularizer=tf.keras.regularizers.L2()),
                                    tf.keras.layers.Dense(1, activation='sigmoid')])

        # Load weights
        self.MLP.load_weights('model_weights/MLP_galaxy_features_ARTEMIS.h5')

    def call(self, inputs, training=False):

        # Normalize features in data into the range seen by the models during training
        scaler = joblib.load('model_weights/scaler_ARTEMIS_galaxy_features.save')
        inputs = tf.convert_to_tensor(scaler.transform(inputs.numpy()), np.float32)

        label = self.MLP(inputs)


class MLP_galaxy_fetures_Auriga(tf.keras.Model):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        # Define base architecture
        self.MLP = tf.keras.Sequential([tf.keras.layers.Dense(100, activation='selu', kernel_regularizer=tf.keras.regularizers.L2()),
                                    tf.keras.layers.Dense(1, activation='sigmoid')])

        # Load weights
        self.MLP.load_weights('model_weights/MLP_galaxy_features_Auriga.h5')

    def call(self, inputs, training=False):

        # Normalize features in data into the range seen by the models during training
        scaler = joblib.load('model_weights/scaler_AURIGA_galaxy_features.save')
        inputs = tf.convert_to_tensor(scaler.transform(inputs.numpy()), np.float32)

        label = self.MLP(inputs)


class TML_ARTEMIS(tf.keras.Model):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)


        train_halos = [1,15,17,18,19,23,24,25,27,38,40,44]
        weights_list = ['model_weights/individual_galaxies/ARTEMIS/G{:02}_MLP_weights.h5'.format(halo) for halo in train_halos]
        self.model_ensemble = []

        for weights in weights_list:

            model = tf.keras.Sequential([tf.keras.layers.Dense(100, activation='selu'),
                                        tf.keras.layers.Dense(1, activation='sigmoid')])
            model.build(input_shape=(None,8))
            model.load_weights(weights)
            self.model_ensemble.append(model)

        self.TML = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='selu', kernel_regularizer=tf.keras.regularizers.L2()),
                                tf.keras.layers.Dense(1, activation='sigmoid')
                                ])
        
        self.TML.build(input_shape=(None, 12))
        self.TML.load_weights('model_weights/TML_ARTEMIS.h5')


    def call(self, inputs, training=False):

        # Normalize features in data into the range seen by the models during training
        scaler = joblib.load('model_weights/scaler_ARTEMIS.save')
        inputs = tf.convert_to_tensor(scaler.transform(inputs.numpy()), np.float32)

        # Get predictive descriptions from single models in the ensemble
        prediction_list = [model(inputs) for model in self.model_ensemble]
        x = tf.concat(prediction_list, 1)

        # Pass predictions to neural network
        label = self.TML(x)

        return label


class TML_Auriga(tf.keras.Model):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)


        train_halos = [1,15,17,18,19,23,24,25,27,38,40,44]
        weights_list = ['model_weights/individual_galaxies/Auriga/G{:02}_MLP_weights.h5'.format(halo) for halo in train_halos]
        self.model_ensemble = []

        for weights in weights_list:

            model = tf.keras.Sequential([tf.keras.layers.Dense(100, activation='selu'),
                                        tf.keras.layers.Dense(1, activation='sigmoid')])
            model.build(input_shape=(None,8))
            model.load_weights(weights)
            self.model_ensemble.append(model)

        self.TML = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='selu', kernel_regularizer=tf.keras.regularizers.L2()),
                                tf.keras.layers.Dense(1, activation='sigmoid')
                                ])
        
        self.TML.build(input_shape=(None, 12))
        self.TML.load_weights('model_weights/TML_Auriga.h5')


    def call(self, inputs, training=False):

        # Normalize features in data into the range seen by the models during training
        scaler = joblib.load('model_weights/scaler_AURIGA.save')
        inputs = tf.convert_to_tensor(scaler.transform(inputs.numpy()), np.float32)

        # Get predictive descriptions from single models in the ensemble
        prediction_list = [model(inputs) for model in self.model_ensemble]
        x = tf.concat(prediction_list, 1)

        # Pass predictions to neural network
        label = self.TML(x)

        return label


class xgb_ARTEMIS():

    def __init__(self):
        
        self.xgb_model = pickle.load((open('model_weights/xgb_ARTEMIS.pkl','rb')))

    def predict(self, inputs):

        # Normalize features in data into the range seen by the models during training
        scaler = joblib.load('model_weights/scaler_ARTEMIS.save')
        inputs = scaler.transform(inputs)

        labels = self.xgb_model.predict_proba(inputs)[:,1]

        return labels


class xgb_Auriga():

    def __init__(self):
        
        self.xgb_model = pickle.load((open('model_weights/xgb_Auriga.pkl','rb')))

    def predict(self, inputs):

        # Normalize features in data into the range seen by the models during training
        scaler = joblib.load('model_weights/scaler_AURIGA.save')
        inputs = scaler.transform(inputs)

        labels = self.xgb_model.predict_proba(inputs)[:,1]

        return labels
        

class UMAP_ARTEMIS():

    def __init__(self):
        
        self.umap_model = pickle.load((open('model_weights/UMAP_ARTEMIS.sav', 'rb')))
        print('UMAP model trained on ARTEMIS loaded.')

    def predict(self, inputs):

        # Normalize features in data into the range seen by the models during training
        scaler = joblib.load('model_weights/scaler_ARTEMIS.save')
        inputs = scaler.transform(inputs)

        # Project data into the UMAP embedding
        X = self.umap_model.transform(inputs)
        dim_1 = X[:,0]
        dim_2 = X[:,1]

        # Infer label predictions based on the region of the embedding
        labels = np.zeros(inputs.shape[0])
        idx_accreted = np.where( ((dim_1<9) & (dim_1>4) & (dim_2<6) & (dim_2>1)) )[0]

        labels[idx_accreted] = 1.0

        return labels, X

