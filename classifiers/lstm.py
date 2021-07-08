import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers 
import matplotlib.pyplot as plt
from matplotlib import cm

class LSTM(tf.keras.Model):

    def __init__(self, inputShape=(1,3)):
        
        super(LSTM, self).__init__()
        
        self.inputLayer = layers.InputLayer(inputShape)
        self.mask = layers.Masking(mask_value = 0.0)
        self.lstm = layers.LSTM(128, stateful=True)
        self.h1 = layers.Dense(128, activation='relu')
        self.outputLayer = layers.Dense(5, activation='softmax')
		
    def call(self, x):
        
        x = self.inputLayer(x)
        x = self.mask(x)
        x = self.lstm(x)
        x = self.h1(x)

        return self.outputLayer(x)
		
class LSTMTrainer():

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
        print('Accuracy: {}'.format(trainAcc), flush=True)
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

    def _train_epoch(self, trainDataset, valDataset):

        for batchNum, (xBatchTrain, yBatchTrain) in enumerate(trainDataset):

            numTimeSteps = xBatchTrain.shape[1]

            for t in range(numTimeSteps-1):
                xBatchTimeStep = xBatchTrain[:, t:t+1, :]
                lossValue = self._train_step(xBatchTimeStep, yBatchTrain)

            if batchNum % 2 == 0:
                print("Training loss at step {}: {:.4f}".format(batchNum, float(lossValue)))
                print("Seen so far: {} samples".format((batchNum * self.batchSize)), flush=True)

            # reset state after each batch
            self.model.reset_states()

        self._save_metrics(lossValue, valDataset)

    def train(self, trainData, valData, epochs, batchSize, resume = False):

        if resume:
            self.model.load_weights(os.path.join(self.weightsDir, 'lstm_final_weights'))
            print('loading weights from checkpoint')
            print(self.weightsDir, flush=True)
        else:
            print('training from scratch', flush=True)

        self.batchSize = batchSize

        x_train, y_train = trainData
        x_val, y_val = valData
        print('Shape of training dataset:', x_train.shape)

        trainDataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        trainDataset = trainDataset.batch(self.batchSize)

        valDataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        valDataset = valDataset.batch(self.batchSize)

        for epoch in range(epochs):
            
            print("\nStart of epoch {}\n".format(epoch))
            self._train_epoch(trainDataset, valDataset)

        self.model.save_weights(os.path.join(self.weightsDir, 'lstm_final_weights'))
        self.model.save(os.path.join(self.weightsDir, 'lstm_model'))
        print('saving weights {}/lstm_final_weights'.format(self.weightsDir))
            
        return self.history

class LSTMTester:

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

        self.model.load_weights(os.path.join(self.weightsDir,'lstm_final_weights'))
        print(self.weightsDir)
        print('model weights were loaded')

        stepSize = 1
        downSampleStride = 100
        timeSteps = [x*stepSize for x in range(0, 10000, downSampleStride)]

        for instance, label in self.dataset:

            instance = self._transform_timeseries(instance)

            for i, timeStep in enumerate(timeSteps):

                if timeStep+stepSize >= instance.shape[1]:
                    break

                inputTensor = instance[:, timeStep:timeStep+stepSize, :]
                
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

class LSTMGradientVisualizer:

    def __init__(self, model, dataset, scene, weightsDir='./data/optimal_rrt_lstm_weights'):

        self.model = model
        self.weightsDir = weightsDir
        self.dataset = dataset
        self.model.load_weights(os.path.join(self.weightsDir, 'lstm_final_weights'))
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.scene = scene

    def _transform_timeseries(self, instance):

        timeseries = []

        for i in range(len(instance[0])):

            timeseries.append([instance[0][i], instance[1][i], instance[2][i]])

        return np.array([timeseries])
   
    def _draw_obstacle(self, obstacle):

        obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
        self.ax.add_patch(obs)
        self.fig.canvas.draw()

    def _draw_target(self, target, colorSelection='blue'):

        tar = plt.Circle((target[0], target[1]), target[2], color=colorSelection, fill=False)
        self.ax.add_patch(tar)
        self.fig.canvas.draw()

    def _plot_scene(self, targetIdx):

        self.fig = plt.figure()
        self.ax = self.fig.gca()
        plt.xlim(self.scene.dimensions['xmin'] , self.scene.dimensions['xmax'] )
        plt.ylim(self.scene.dimensions['ymin'] , self.scene.dimensions['ymax'] )
        self.ax.set_aspect('equal')
        self.ax.set_ylabel('Y-distance(M)')
        self.ax.set_xlabel('X-distance(M)')

        for obstacle in self.scene.obstacles:

            self._draw_obstacle(obstacle)

        for target in self.scene.targets:

            self._draw_target(target)
 
    def _sensitivity_gradients(self, instance, label):

        grads = []

        for timeStep in range(instance.shape[1]):

            success = False
            x = tf.constant(instance[:,timeStep,:], shape=(1,1,3))

            with tf.GradientTape() as tape:

                tape.watch(x)
                logits = self.model(x)
                label = label.reshape((1,5))
                lossValue = self.loss_fn(label, logits)

                if np.argmax(logits) == np.argmax(label):
                    success = True

            grads.append(tape.gradient(lossValue, x))

        grads = np.array(grads)

        return grads

    def visualize(self):

        for instance, label in self.dataset:

            self._plot_scene(np.argmax(label))

            instance = self._transform_timeseries(instance)#[:,:1000,:]

            grads = self._sensitivity_gradients(instance, label)

            x = []
            y = []
            gradientMagnitudes = []
            for step, position in enumerate(instance[0]):

                if step % 10 == 0:
                    x.append(position[0]*10.0)
                    y.append(position[1]*10.0)
                    gradientMagnitudes.append(np.linalg.norm(grads[step,:]))
                
            gradientMagnitudes = np.array(gradientMagnitudes)
            gradientMagnitudes /= gradientMagnitudes.max() 

            self.ax.scatter(x, y, c=cm.hot(gradientMagnitudes), edgecolor='none')
            plt.show()

            self.model.reset_states()

