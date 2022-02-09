import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import pydelfi22.priors as priors
import pydelfi22.ndes as ndes
import pydelfi22.delfi as delfi
tf.logging.set_verbosity(tf.logging.ERROR)
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler




class Compress():

    def __init__(self, train_data, train_input, validation_fraction  = 0.5, n_ensemble = 3, max_hidden = 3, hidden_size = 25, n_epochs = 200, patience = 20,  activation='relu', loss = 'mean_squared_error', optimizer = 'adam', save_dir  = 'models/', data_standardization = None,):

        self.train_data = train_data
        self.train_input = train_input
        self.validation_fraction = validation_fraction
        self.n_ensemble = n_ensemble
        self.max_hidden = max_hidden
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.n_ensemble = n_ensemble
        self.save_dir = save_dir
        self.data_standardization =  data_standardization

        #get the number of data points
        self.n_features = np.shape(self.train_data)[1]
        self.param_size = np.shape(train_input)[1]

        #split the data into a training and training set
        n_data = np.shape(self.train_data)[0]
        n_train = int(n_data * self.validation_fraction)
        self.trainX, self.testX = self.train_data[:n_train, :], self.train_data[n_train:, :]
        self.trainy, self.testy = self.train_input[:n_train], self.train_input[n_train:]


    def standardize_input(self):
        if self.data_standardization == None:
            pass
        elif self.data_standardization == 'MinMaxScaler':
            print ('Standardizing Input')
            scalar = MinMaxScaler()
            scalar.fit(np.vstack((self.trainX,self.testX)))
            self.trainX = scalar.transform(self.trainX)
            self.testX = scalar.transform(self.testX)
        elif self.data_standardization =='StandardScaler':
            print ('Standardizing Input')
            scalar = StandardScaler()
            scalar.fit(np.vstack((self.trainX,self.testX)))
            self.trainX = scalar.transform(self.trainX)
            self.testX = scalar.transform(self.testX)
        else:
            print ('Error:data_standardization must be None, "MinMaxScaler", or "StandardScaler."')
            


    def train_network(self, n_hidden):
        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim = self.n_features, activation = self.activation))
        for i in range(n_hidden):
            model.add(Dense(self.hidden_size, input_dim= self.hidden_size, activation = self.activation))
        model.add(Dense(self.param_size, activation='linear'))
        model.compile(loss = self.loss, optimizer = self.optimizer)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience= self.patience)
        
        #you are now here in the code
        history = model.fit(self.trainX, self.trainy, validation_data=(self.testX, self.testy), epochs=self.n_epochs, callbacks=[es])
        # evaluate the model
        train_mse = model.evaluate(self.trainX, self.trainy)
        test_mse = model.evaluate(self.testX, self.testy)
        print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
        return model, test_mse


    def ensemble_train(self):
        self.standardize_input()
        self.model_list = []
        self.test_mse_list = []
        for ensemble in range(self.n_ensemble):
            for n_hidden in range(self.max_hidden):
                print ('Training: Ensemble %s, Network %s' %(ensemble + 1, n_hidden + 1))
                model, test_mse = self.train_network(n_hidden)
                self.model_list += [model]
                self.test_mse_list += [test_mse]
        return 0.


    def ensemble_compress(self, data):
        numerator = 0.
        denominator = 0.
        for i in range(len(self.model_list)):
            numerator += 1. / self.test_mse_list[i] ** 2. * self.model_list[i].predict(data)
            denominator += 1. / self.test_mse_list[i] ** 2.

        return (numerator / denominator)


    def save_model(self):
        np.savetxt(self.save_dir + 'test_mse_list.txt', self.test_mse_list)
        for i in range(len(self.model_list)):
            self.model_list[i].save(self.save_dir + 'model_%s' %i)
        return 0.


    def load_model(self):
        self.test_mse_list = np.loadtxt(self.save_dir + 'test_mse_list.txt')
        self.model_list = []
        for i in range(len(self.test_mse_list)):
            self.model_list += [tf.keras.models.load_model(self.save_dir + 'model_%s' %i)]
        return 0.








class DelfiPreloadRun():

    def __init__(self, compressed_data, sim_params, sim_compressed_data, prior, theta_fiducial, param_limits, param_names, nwalkers, results_dir):


        self.compressed_data = compressed_data
        self.sim_params = sim_params
        self.sim_compressed_data = sim_compressed_data
        self.prior = prior
        self.theta_fiducial = theta_fiducial
        self.param_limits = param_limits
        self.param_names = param_names
        self.nwalkers = nwalkers
        self.results_dir = results_dir   
        self.n_params = np.shape(self.sim_params)[1]


    def initialize(self):

        #create ensemble of NDEs
        NDEs = [ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=self.n_params, n_data=self.n_params, n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=0),
        ndes.MixtureDensityNetwork(n_parameters=self.n_params, n_data=self.n_params, n_components=1, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=1),
        ndes.MixtureDensityNetwork(n_parameters=self.n_params, n_data=self.n_params, n_components=2, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=2),       
        ndes.MixtureDensityNetwork(n_parameters=self.n_params, n_data=self.n_params, n_components=3, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=3),
        ndes.MixtureDensityNetwork(n_parameters=self.n_params, n_data=self.n_params, n_components=4, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=4),
        ndes.MixtureDensityNetwork(n_parameters=self.n_params, n_data=self.n_params, n_components=5, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=5)]

        #create delfi object
        self.DelfiEnsemble = delfi.Delfi(self.compressed_data, self.prior, NDEs, 
                            Finv = None, 
                            theta_fiducial = self.theta_fiducial, 
                            param_limits = self.param_limits,
                            param_names = self.param_names, 
                            nwalkers = self.nwalkers,
                            results_dir = self.results_dir,
                            input_normalization=None)


        #load simulations
        self.DelfiEnsemble.load_simulations(self.sim_compressed_data, self.sim_params)
        return 0.


    def train(self):
        print ("Training Delfi Networks")
        #train the network
        self.DelfiEnsemble.train_ndes()
        return 0.


    def sample(self):
        print ("Drawing Samples From Posterior")
        #sample the learned procedure
        self.posterior_samples, self.posterior_weights, self.log_prob = self.DelfiEnsemble.emcee_sample()
        return 0.


    def get_posterior_samples(self):
        self.initialize()
        self.train()
        self.sample()
        return self.posterior_samples, self.posterior_weights, self.log_prob




