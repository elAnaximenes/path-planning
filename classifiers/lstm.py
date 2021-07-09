import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
from matplotlib.lines import Line2D
import math

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
        self.accuracyInfo = {'tp': {}, 'label count': {}}
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
                
                timeToGoal = (math.ceil(instance.shape[1]/float(downSampleStride)) * downSampleStride) - timeStep 

                if timeToGoal not in self.accuracyInfo['tp']:
                    self.accuracyInfo['tp'][timeToGoal] = 1
                    self.accuracyInfo['label count'][timeToGoal] = 1

                self.accuracyInfo['label count'][timeToGoal] += 1
                if prediction == np.argmax(label):
                    self.accuracyInfo['tp'][timeToGoal] += 1

            self.model.reset_states()

        return self.accuracyInfo

class LSTMGradientVisualizer:

    def __init__(self, model, dataset, scene, weightsDir='./data/optimal_rrt_lstm_weights', display=True):

        self.model = model
        self.weightsDir = weightsDir
        self.dataset = dataset
        self.model.load_weights(os.path.join(self.weightsDir, 'lstm_final_weights'))
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.scene = scene
        self.targetColors = ['green', 'blue', 'cyan', 'darkorange', 'purple']
        self.gradientMagnitudes = None
        self.stepSize = 10
        self.timeSteps = None
        self.display = display

    def _plot_gradients(self, grads):

        grads[:, 0] /= np.amax(grads[:, 0])
        grads[:, 1] /= np.amax(grads[:, 1])
        grads[:, 2] /= np.amax(grads[:, 2])

        self.grads_ax_x.plot(self.timeSteps, grads[:, 0])
        self.grads_ax_y.plot(self.timeSteps, grads[:, 1])
        self.grads_ax_theta.plot(self.timeSteps, grads[:, 2])

        self.grads_ax_x.set_title('Gradient Magnitude w.r.t. "X" position')
        self.grads_ax_y.set_title('Gradient Magnitude w.r.t. "Y" position')
        self.grads_ax_theta.set_title('Gradient Magnitude w.r.t. Heading')

        self.grads_ax_x.set_ylim(-1.0, 1.0)
        self.grads_ax_y.set_ylim(-1.0, 1.0)
        self.grads_ax_theta.set_ylim(-1.0, 1.0)

        self.grads_ax_x.set_ylabel('magnitude')
        self.grads_ax_y.set_ylabel('magnitude')
        self.grads_ax_theta.set_ylabel('magnitude')

        self.grads_ax_x.set_xlabel('Timesteps')
        self.grads_ax_y.set_xlabel('Timesteps')
        self.grads_ax_theta.set_xlabel('Timesteps')

    def _plot_classification_over_time(self, predictions, label):

        self.class_ax.set_title('Classifier Confidence over Time')
        self.class_ax.set_xlim(0, self.timeSteps[-1]) 
        self.class_ax.set_ylim(-0.05, 1.05)
        self.scene_ax.set_aspect('equal', adjustable='box', anchor = 'C')
        self.class_ax.set_xlabel('Timestep')
        self.class_ax.set_ylabel('Confidence')

        for targetIdx in range(len(self.targetColors)):

            confidence = []
            time = []
            for t in range(len(self.timeSteps)):

                confidence.append(predictions[t][0][targetIdx])

            self.class_ax.plot(self.timeSteps, confidence, linestyle='--', color=self.targetColors[targetIdx], label = 'Target {}'.format(targetIdx))

        self.class_ax.legend(loc='lower left')
    
    def _plot_gradient_arrows(self, x, y, grads):

        samplingRate = 5
        print(grads.shape[0])
        for step in range(grads.shape[0]):
            pass

    def _annotate_path(self, x, y):
        
        for t in range(0, len(x), 500//self.stepSize):

            self.scene_ax.text(x[t]+0.5, y[t]-0.5, t*self.stepSize, color='red')

    def _show_colorbar(self, scatter):

        cbar = plt.colorbar(cm.ScalarMappable(cmap='hot'), ax=self.scene_ax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Normalized Gradient Magnitude', rotation = 270)
        
    def _plot_path(self, x, y, grads):

        scatter = self.scene_ax.scatter(x, y, c=cm.hot(self.gradientMagnitudes), edgecolor='none')
        self._show_colorbar(scatter)
        # plot line over gradient heat path to show actual state
        self.scene_ax.plot(x,y, 'b--', linewidth=1.0)
        self._annotate_path(x,y)
        #self._plot_gradient_arrows(x, y, grads)

    def _get_path(self, instance):

        x = []
        y = []

        for t in self.timeSteps: 
            x.append(instance[t, 0]*10)
            y.append(instance[t, 1]*10)

        return x, y
   
    def _draw_obstacle(self, obstacle):

        obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
        self.scene_ax.add_patch(obs)
        self.fig.canvas.draw()

    def _draw_target(self, target, index, colorSelection='blue'):

        tar = plt.Circle((target[0], target[1]), target[2], color=colorSelection, fill=False)
        self.scene_ax.add_patch(tar)
        self.fig.canvas.draw()
        self.scene_ax.text(target[0]-0.5, target[1]+0.5, index)

    def _scene_legend(self):

        legend_elements = [ Line2D([0], [0], marker='o', linestyle='', fillstyle='none', label='Target 0', markeredgecolor='green', markersize=15),
                            Line2D([0], [0], marker='o', fillstyle='none', linestyle='', label='Target 1', markeredgecolor='blue', markersize=15),
                            Line2D([0], [0], marker='o', fillstyle='none', linestyle='', label='Target 2', markeredgecolor='cyan', markersize=15),
                            Line2D([0], [0], marker='o', fillstyle='none', linestyle='', label='Target 3', markeredgecolor='darkorange', markersize=15),
                            Line2D([0], [0], marker='o', fillstyle='none', linestyle='', label='Target 4', markeredgecolor='purple', markersize=15),
                            Line2D([0], [0], marker='o', fillstyle='none', linestyle='', label='Obstacle', markeredgecolor='red', markersize=15)]

        self.scene_ax.legend(handles=legend_elements, loc='best')

    def _plot_scene(self, targetIdx, timeSteps):

        self.scene_ax.set_title('LSTM Gradient Magnitude w.r.t. Dubins State')
        self.scene_ax.set_xlim(self.scene.dimensions['xmin'] , self.scene.dimensions['xmax'] )
        self.scene_ax.set_ylim(self.scene.dimensions['ymin'] , self.scene.dimensions['ymax'] )
        self.scene_ax.set_aspect('equal')
        self.scene_ax.set_ylabel('Y-distance(M)')
        self.scene_ax.set_xlabel('X-distance(M)')

        for obstacle in self.scene.obstacles:

            self._draw_obstacle(obstacle)

        for index, target in enumerate(self.scene.targets):

            self._draw_target(target, index, self.targetColors[index])

    def _initialize_plot(self):

        self.fig = plt.figure(figsize=(12,12))
        gs = gridspec.GridSpec(4,2, width_ratios=(2,1), hspace=0.5)
        self.scene_ax = plt.subplot(gs[:3, 0])
        self.class_ax = plt.subplot(gs[3, 0])
        self.grads_ax_x = plt.subplot(gs[0, 1])
        self.grads_ax_y = plt.subplot(gs[1, 1])
        self.grads_ax_theta = plt.subplot(gs[2, 1])

    def _show_plots(self, instance, grads, predictions, label):

        self._initialize_plot()
        self._plot_scene(targetIdx = np.argmax(label), timeSteps = instance.shape[0] )
        x, y = self._get_path(instance)
        self._plot_path(x, y, grads)
        self._plot_classification_over_time(predictions, label)
        self._plot_gradients(grads)

        if self.display:
            plt.show()

    def _calculate_gradient_magnitudes(self, grads):

        gradientMagnitudes = []

        for step in range(grads.shape[0]):
            gradientMagnitudes.append(np.linalg.norm(grads[step,:]))
            
        gradientMagnitudes = np.array(gradientMagnitudes)
        gradientMagnitudes /= gradientMagnitudes.max() 

        self.gradientMagnitudes = gradientMagnitudes

    def _get_gradients_and_predictions(self, instance, label):

        grads = []
        predictions = []

        for t in self.timeSteps:

            success = False
            x = tf.constant(instance[:, t,:], shape=(1,1,3))

            with tf.GradientTape() as tape:

                tape.watch(x)
                logits = self.model(x)
                label = label.reshape((1,5))
                lossValue = self.loss_fn(label, logits)

                if np.argmax(logits) == np.argmax(label):
                    success = True

            gradient = tape.gradient(lossValue, x)[0][0]
            grads.append(gradient)
            predictions.append(logits)

        self.model.reset_states()
        grads = np.array(grads)

        return grads, predictions

    def _transform_timeseries(self, instance):

        timeSeries = []

        for i in range(len(instance[0])):

            timeSeries.append([instance[0][i], instance[1][i], instance[2][i]])

        self.timeSteps = range(0, len(timeSeries), self.stepSize)

        return np.array([timeSeries])

    def visualize(self):

        for instance, label in self.dataset:

            instance = self._transform_timeseries(instance)
            grads, predictions = self._get_gradients_and_predictions(instance, label)
            self._calculate_gradient_magnitudes(grads)
            self._show_plots(instance[0], grads, predictions, label)
