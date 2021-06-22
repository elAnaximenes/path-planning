import tensorflow as tf
from tensorflow.keras import layers 

class LSTM(tf.keras.Model):

    def __init__(self, inputShape):
        
        super(LSTM, self).__init__()
        self.lstmcell = layers.RNN(layers.LSTMCell(4))
        self.H1 = layers.Dense(64, activation='relu')
        self.outputLayer = layers.Dense(5, activation='softmax')
		
    def call(self, x, training=False):
        


        x = self.lstmcell(x)
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

        xBatchTrain = tf.constant(xBatchTrain, shape = (1, len(xBatchTrain[0]), 3))

        with tf.GradientTape() as tape:

            tape.watch(xBatchTrain)
            logits = self.model(xBatchTrain, training=True)
            print(logits)
            print(yBatchTrain.shape)
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

            valLogits = self.model(xBatchVal, training=False)
            self.valAccMetric.update_state(yBatchVal, valLogits)
            lossValue = self.loss_fn(yBatchVal, valLogits)

        valAcc = self.valAccMetric.result()
        self.history['valAcc'].append(valAcc)
        self.history['valLoss'].append(float(lossValue))
        self.valAccMetric.reset_states()

    def train(self, trainData, valData, epochs, batchSize):

        x_train, y_train = trainData
        print('y train shape', y_train.shape)
        x_val, y_val = valData

        #trainDataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        #trainDataset = trainDataset.batch(1)
        trainDataset = zip(x_train, y_train)

        #valDataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        #valDataset = valDataset.batch(1)
        valDataset = zip(x_val, y_val)

        for epoch in range(epochs):
            
            print("\nStart of epoch {}\n".format(epoch))

            for step, (xBatchTrain, yBatchTrain) in enumerate(trainDataset):

                lossValue = self._train_step(xBatchTrain, yBatchTrain)

                if step % 2 == 0:
                    print("Training loss at step {}: {:.4f}".format(step, float(lossValue)))
                    print("Seen so far: {} samples".format((step + 1) * batchSize))

            self._save_metrics(lossValue, valDataset)
            
        return self.history
