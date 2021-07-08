import tensorflow as tf
import numpy as np
from tensorflow.keras import layers 

class CNN(tf.keras.Model):

    def __init__(self, inputShape=(1,3)):
        
        super(CNN, self).__init__()
        
        self.inputLayer = layers.Inputlayer(inputShape)
        self.conv1 = layers.Conv1D(64, 7, padding='causal', activation='relu')
        self.maxPool = layers.MaxPooling1D(5)
        self.conv2 = layers.Conv1D(64, 7, padding='causal', activation='relu')
        self.globalMaxPool = layers.GlobalMaxPooling1D()
        self.h1 = layers.Dense(64, activation='relu')
        self.outputLayer = layers.Dense(5, activation='softmax')
		
    def call(self, x):
        
        x = self.inputLayer(x)
        x = self.conv1(x)
        x = self.maxPool(x)
        x = self.conv2(x)
        x = self.globalMaxPool(x)
        x = self.h1(x)

        return self.outputLayer(x)
		
class CNNTrainer():

    def __init__(self, model, weightsDir):
        
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.trainAccMetric = tf.keras.metrics.CategoricalAccuracy()
        self.valAccMetric = tf.keras.metrics.CategoricalAccuracy()
        self.history = {'trainAcc':[], 'valAcc':[], 'trainLoss':[], 'valLoss':[]}
        self.weightsDir = weightsDir

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
            self.model.load_weights(os.path.join(self.weightsDir, 'cnn_final_weights'))
            print('loading weights from checkpoint')
            print(self.weightsDir, flush=True)
        else:
            print('\nTraining from scratch\n'

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

                lossValue = self._train_step(xBatchTrain, yBatchTrain)

                if step % 2 == 0:
                    print("Training loss at step {}: {:.4f}".format(step, float(lossValue)))
                    print("Seen so far: {} samples".format((step * batchSize)))


            self._save_metrics(lossValue, valDataset)

        self.model.save_weights(os.path.join(self.weightsDir, 'cnn_final_weights'))
        self.model.save(os.path.join(self.weightsDir, 'cnn_model'))
            
        return self.history

class CNNTester:

    def __init__(self, dataset, model, weightsDir):

        self.dataset = dataset
        self.model = model
        self.numSamples = len(dataset[1])
        self.accuracyInfo = {'tp': [], 'label count': []}
        self.weightsDir = weightsDir

    def _transform_timeseries(self, instance):

        timeseries = []
        for i in range(len(instance[0])):
            timeseries.append([instance[0][i], instance[1][i], instance[2][i]])

        return np.array([timeseries])


    def test(self):

        self.model.load_weights(os.path.join(self.weightsDir,'cnn_final_weights'))

        print('model weights were loaded')
        self.dataset = self.dataset[:1000]
        stepSize = 1
        timeSteps = [x*stepSize for x in range(10000)]

        for instance, label in self.dataset:

            instance = self._transform_timeseries(instance)

            for i, timeStep in enumerate(timeSteps):

                if timeStep+stepSize >= instance.shape[1]:
                    break

                inputTensor = instance[:, timeStep:timeStep+stepSize, :]
                
                #inputTensor = tf.tensor(newInstance )
                logits = self.model(inputTensor)
                prediction = np.argmax(logits)

                if len(self.accuracyInfo['tp']) < i+1:
                    self.accuracyInfo['tp'].append(0)
                    self.accuracyInfo['label count'].append(0)

                self.accuracyInfo['label count'][i] += 1
                if prediction == np.argmax(label):
                    self.accuracyInfo['tp'][i] += 1

            self.model.reset_states()



        return self.accuracyInfo


                
