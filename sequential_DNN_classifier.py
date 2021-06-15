from keras import models
from keras import layers
import argparse

def load_training_data(sceneName):
    
    pass

def train_DNN(data):

    x_train, y_train = data
    model = models.sequential()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='sequential deep neural network model for classifying a path\'s target based on the beginning of the path\'s trajectory.')
    parser.add_argument('--scene', type=str, help='specify scene', default='simple_room')
    
    args = parser.parse_args()

    sceneName = args.scene

    trainingData = load_training_data(sceneName)
    train_DNN(trainingData)





