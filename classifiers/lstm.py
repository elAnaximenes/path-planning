import tensorflow as tf
from tensorflow.keras import layers 

class LSTM(tf.keras.Model):

    def __init__(self, inputShape):
        
        super(LSTM, self).__init__()
        self.lstmcell = layers.LSTMCell(1)
        self.H1 = layers.Dense(64, activation='relu')
        self.outputLayer = layers.Dense(5, activation='softmax')
		
    def call(self, x, training=False):
        
        print(x.shape)
        exit(1)
        c = self.lstmcell.get_initial_state()
        numTimeSteps = x.shape[-1]

        for i in range(numTimeSteps):

            x, [x,c] = self.lstmcell(x, [x,c], training=training)

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

            valLogits = self.model(xBatchVal, training=False)
            self.valAccMetric.update_state(yBatchVal, valLogits)
            lossValue = self.loss_fn(yBatchVal, valLogits)

        valAcc = self.valAccMetric.result()
        self.history['valAcc'].append(valAcc)
        self.history['valLoss'].append(float(lossValue))
        self.valAccMetric.reset_states()

    def train(self, trainData, valData, epochs, batchSize):

        x_train, y_train = trainData
        x_val, y_val = valData

        trainDataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        trainDataset = trainDataset.batch(1)

        valDataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        valDataset = valDataset.batch(1)

        for epoch in range(epochs):
            
            print("\nStart of epoch {}\n".format(epoch))

            for step, (xBatchTrain, yBatchTrain) in enumerate(trainDataset):

                lossValue = self._train_step(xBatchTrain, yBatchTrain)

                if step % 2 == 0:
                    print("Training loss at step {}: {:.4f}".format(step, float(lossValue)))
                    print("Seen so far: {} samples".format((step + 1) * batchSize))

            self._save_metrics(lossValue, valDataset)
            
        return self.history
