import tensorflow as tf
from tensorflow.keras import layers 

class LSTM(tf.keras.Model):

    def __init__(self, inputShape):
        
        super(FeedForward, self).__init__()
        self.inputLayer = layers.InputLayer(input_shape=(inputShape))
        self.flattenLayer = (layers.Flatten())
        self.H1 = layers.Dense(128, activation='relu')
        self.H2 = layers.Dense(64, activation='relu')
        self.H3 = layers.Dense(32, activation='relu')
        self.outputLayer = layers.Dense(5, activation='softmax')
		
    def call(self, x):
        
        x = self.inputLayer(x)
        x = self.flattenLayer(x)
        x = self.H1(x)
        x = self.H2(x)
        x = self.H3(x)

        return self.outputLayer(x)
		
class LSTMTrainer():

    def __init__(self, model):
        
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def train(self, trainData, valData, epochs, batchSize):

        x_train, y_train = trainData
        x_val, y_val = valData

        trainDataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        trainDataset = trainDataset.shuffle(buffer_size=1024).batch(batchSize)

        valDataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        valDataset = valDataset.batch(batchSize)

        for epoch in range(epochs):
            
            print("\nStart of epoch {}\n".format(epoch))

            for step, (xBatchTrain, yBatchTrain) in enumerate(trainDataset):

                with tf.GradientTape() as tape:

                    tape.watch(xBatchTrain)
                    logits = self.model(xBatchTrain)
                    lossValue = self.loss_fn(yBatchTrain, logits)

                grads = tape.gradient(lossValue, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                if step % 2 == 0:
                    print("Training loss at step {}: {:.4f}".format(step, float(lossValue)))
                    print("Seen so far: {} samples".format((step + 1) * batchSize))

