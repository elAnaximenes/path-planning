import tensorflow as tf
import numpy as np
from tensorflow.keras import layers 

class LSTM(tf.keras.Model):

    def __init__(self, inputShape=(3,)):
        
        super(LSTM, self).__init__()
        
        self.inputLayer = layers.InputLayer(input_shape=(inputShape))
        #self.mask = layers.Masking(mask_value = 0.0)
        self.lstmcell = layers.RNN(layers.LSTMCell(4))
        self.H1 = layers.Dense(16, activation='relu')
        self.outputLayer = layers.Dense(5, activation='softmax')
		
    def call(self, x):
        
        #print('shape of network input:', x.shape)
        x = self.inputLayer(x)
        #x = self.mask(x)
        #print('input to LSTM layer')
        #print(x.shape)
        x = self.lstmcell(x)
        #print('output of LSTM layer')
        #print(x.shape)
        x = self.H1(x)

        return self.outputLayer(x)
		
class LSTMTrainer():

    def __init__(self, model):
        
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.trainAccMetric = tf.keras.metrics.CategoricalAccuracy()
        self.valAccMetric = tf.keras.metrics.CategoricalAccuracy()
        self.history = {'trainAcc':[], 'valAcc':[], 'trainLoss':[], 'valLoss':[]}

    def _train_step(self, xBatchTrain, yBatchTrain):

        with tf.GradientTape() as tape:

            tape.watch(xBatchTrain)
            logits = self.model(xBatchTrain, training=True)
            lossValue = self.loss_fn(yBatchTrain, logits)

        grads = tape.gradient(lossValue, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.trainAccMetric.update_state(yBatchTrain, logits)

        return lossValue

    def _save_metrics(self, lossValue, valDataset):

        trainAcc = self.trainAccMetric.result()
        self.history['trainAcc'].append(trainAcc)
        self.history['trainLoss'].append(float(lossValue))
        self.trainAccMetric.reset_states()

        for xBatchVal, yBatchVal in valDataset:

            valLogits = self.model(xBatchVal)
            self.valAccMetric.update_state(yBatchVal, valLogits)
            lossValue = self.loss_fn(yBatchVal, valLogits)

        valAcc = self.valAccMetric.result()
        self.history['valAcc'].append(valAcc)
        self.history['valLoss'].append(float(lossValue))
        self.valAccMetric.reset_states()

    def train(self, trainData, valData, epochs, batchSize, resume = False):

        if resume:
            self.model.load_weights('./data/lstm_weights/lstm_final_weights')
            #self.model = tf.keras.models.load_model('./data/weights/lstm_final_model')

        x_train, y_train = trainData
        x_val, y_val = valData
        print(x_train.shape)

        trainDataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        trainDataset = trainDataset.batch(batchSize)

        valDataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        valDataset = valDataset.batch(batchSize)

        for epoch in range(epochs):
            
            print("\nStart of epoch {}\n".format(epoch))

            for step, (xBatchTrain, yBatchTrain) in enumerate(trainDataset):


                print('shape of network input', xBatchTrain.shape)
                lossValue = self._train_step(xBatchTrain, yBatchTrain)

                if step % 2 == 0:
                    print("Training loss at step {}: {:.4f}".format(step, float(lossValue)))
                    print("Seen so far: {} samples".format((step + 1)))

            self._save_metrics(lossValue, valDataset)

        #self.model.save_weights('./data/lstm_weights/lstm_final_weights')
        #self.model.save('./data/lstm_weights/lstm_model')
            
        return self.history

class LSTMTester:

    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model
        self.numSamples = len(dataset[1])
        self.accuracyInfo = {'tp': [], 'label count': []}

    def test(self):

        self.model.load_weights('./data/lstm_weights/lstm_final_weights')

        print('model weights were loaded')
        self.dataset = self.dataset[:1000]
        stepSize = 100
        timeSteps = [x*stepSize for x in range(10)]

        for instance, label in self.dataset:

            newInstance = np.zeros((1,3,9000))

            for i, timeStep in enumerate(timeSteps):

                if timeStep+stepSize >= len(instance[0]):
                    break

                newInstance[0,:,timeStep:timeStep + stepSize] += instance[:, timeStep:timeStep+stepSize]
                inputTensor = newInstance 
                
                #inputTensor = tf.tensor(newInstance )
                logits = self.model(inputTensor)
                prediction = np.argmax(logits)

                if len(self.accuracyInfo['tp']) < i+1:
                    self.accuracyInfo['tp'].append(0)
                    self.accuracyInfo['label count'].append(0)

                self.accuracyInfo['label count'][i] += 1
                if prediction == np.argmax(label):
                    self.accuracyInfo['tp'][i] += 1

        return self.accuracyInfo


                
