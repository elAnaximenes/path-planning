######
Testing Classifiers
######
Testing a classifier means that data will be loaded from ../data/[_scene_name_]dataset/[_algorithm_name_]batches_validate/. 
This script will take the first "n" batches in that folder and run a classifier of your choice on that data. 
The product is a graph of accuracy over time steps from goal.
Weights will be loaded from ../data/[_scene_name_]dataset/[_algorithm_name_][_classifier_name_]weights/[_classifier_name_]final_weights

examples:
if using the lstm predictor to test compare optimal rrt to adversarial rrt and regular rrt, use this

@@@@@
python test_classifier.py --model [lstm/feedforward/cnn] --batches [number of batches] --algo [rrt/optimal_rrt/adversarial_optimal_rrt] --scene tower_defense --adv predictor
@@@@@

######
Training Classifiers
######
Training a classifier means that data will be loaded from ../data/[_scene_name_]dataset/[_algorithm_name_]batches_train[_predictor/planner_]/.
The trainer will begin training at batch "k" and will train for "n" batches.
The split parameter will split the dataset into [param] percent for training, and [1-param] percent for validation.
A set of weights will be saved after each epoch into ../data/[_scene_name_]dataset/[_algorithm_name_][_classifier_name_]weights/

examples:
the --predictor boolean param means that you are training a predictor(aka the discriminator) in the planner predictor pair. If you are training a lstm for onboard advRRTstar planning, leave this boolean off

@@@@@
python train_classifier.py --model lstm --epochs 20 --batch_size 5000 --split 0.95 --batches 50 --scene tower_defense --algo optimal_rrt --predictor
@@@@@





