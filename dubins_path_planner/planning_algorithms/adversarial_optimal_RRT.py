import math
import os 
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import csv
import tensorflow as tf
import scipy
from scipy import stats

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from car_models.dubins_optimal_planner import DubinsOptimalPlanner
from car_models.dubins_optimal_planner_final_heading_extended import DubinsOptimalPlannerFinalHeading, DubinsError
from car_models.dubins_model import DubinsCar
from matplotlib.lines import Line2D
from scene import Scene

class DubinsCarAdversarialOptimalRRT:

    class NodeRRT:

        def __init__(self, position, pathLength=None, path=None, name=None, entropy = None):

            self.x = position[0] 
            self.y = position[1] 
            self.theta = position[2] 
            self.position = position
            self.parent = None
            self.numChildren = 0
            self.pathLength = 0.0
            self.path = {'x':[], 'y':[], 'theta':[]}
            if path is not None:
                self.path = path
                self.pathLength = pathLength
            self.plottedPath = None
            self.name = name
            self.entropy = entropy 

        def _set_position(self, position):
            self.x = position[0] 
            self.y = position[1] 
            self.theta = position[2] 
            self.position = position           

        def __eq__(self, otherNode):
            return otherNode is not None and self.name == otherNode.name and self.name != 'Temp'

        def __str__(self):
            rep = 'Node {} (cost={:.1f})\n'.format(self.name, self.pathLength)
            if self.parent is not None:
                rep += 'Parent node is: ' + self.parent.__str__() 
            return rep

    def __init__(self, dubinsCar, scene, model, animate = False):

        # tree primitives
        self.car = dubinsCar
        self.root = self.NodeRRT(scene.carStart, name='Root') 
        self.nodeList = [self.root]
        self.goalNode = None
        self.goalNodeList = []
        self.minCostGoalPath = []
        self.scene = scene
        self.nearestNeighborRadius = 6.0
        self.goalNearestNeighborRadius = 2.0
        self.targetIdx = None

        # integrated classifier
        self.model = model

        # path to goal
        self.pathToGoalNodes = None
        self.plottedPathToGoal = [] 

        # animation
        self.animate = animate
        self.fig = None
        self.ax = None
        self.maxIter = 150 
        self.iteration=0
        self.leg=None
        if self.animate:
            self._setup_animation()
        self.text = None
        self.imgcount = 0

    def _select_random_target(self):

        self.targetIdx = random.randint(0, len(self.scene.targets) - 1)
        target = self.scene.targets[self.targetIdx]
        self.target = self.NodeRRT(target, name="Goal")

    def _sample_random_point(self):

        randomPoint = np.zeros(2,)
        randomPoint[0] = random.uniform(self.scene.dimensions['xmin'], self.scene.dimensions['xmax']) 
        randomPoint[1] = random.uniform(self.scene.dimensions['ymin'], self.scene.dimensions['ymax']) 

        return randomPoint

    def _is_path_out_of_bounds(self, x, y):

        if x > self.scene.dimensions['xmax']\
                or x < self.scene.dimensions['xmin']\
                or y > self.scene.dimensions['ymax']\
                or y < self.scene.dimensions['ymin']:
                    return True
        return False

    def _is_point_reachable(self, originNode, destinationPoint, path=None):

        if originNode == None:
            return False

        # checking if target is reachable
        if path is None:
            path = self._get_path_from_node_to_point(originNode, destinationPoint)
            if path is None:
                return False
        
        i = 0
        # check for obstacle collisions
        for x,y in zip(path['x'], path['y']):
            point = np.array([x,y])

            if i % 10 == 0:

                if self._is_path_out_of_bounds(x, y):
                    return False

                # invalid path if point collides with obstacle
                for obstacle in self.scene.obstacles:
                    if abs(np.linalg.norm(point - obstacle[:2])) < obstacle[2]:
                        return False
            i += 1

        return True

    def _get_path_from_node_to_point(self, startNode, destinationPoint):

        if abs(np.linalg.norm(startNode.position[:2] - destinationPoint[:2])) < (2.0 * self.car.minTurningRadius):
            return None 

        dubinsState = np.array([startNode.x, startNode.y, startNode.theta])
        self.car.set_state(dubinsState)
        planner = DubinsOptimalPlanner(self.car, dubinsState, destinationPoint)
        path = planner.run()

        return path

    def _get_dubins_path(self, originNode, destinationPoint):
 
        # re-initialize dubins car to be at origin node
        dubinsState = originNode.position 
        self.car.set_state(dubinsState)

        # instantiate optimal planner
        planner = DubinsOptimalPlanner(self.car, dubinsState, destinationPoint)

        # get optimal path produced by planner
        path = planner.run()

        return path
   
    def _calculate_dubins_path_length(self, originNode, destinationPoint):
        ### Final heading does NOT matter

        # re-initialize dubins car to be at origin node
        dubinsState = originNode.position 
        self.car.set_state(dubinsState)

        # instantiate optimal planner
        planner = DubinsOptimalPlanner(self.car, dubinsState, destinationPoint)

        # get optimal pathlength produced by planner
        pathLength = planner.calculate_shortest_pathlength()

        return pathLength

    def _calculate_dubins_path_length_final_heading(self, originNode, destinationPoint):
        ### Final heading DOES matter (rewiring to non-leaf node)

        dubinsState = originNode.position
        self.car.set_state(dubinsState)

        planner = DubinsOptimalPlannerFinalHeading(self.car, dubinsState, destinationPoint)

        path = planner.run()

        # planner will return None if a path cannot be calculated
        if path is None:
            return None, None
        
        pathLength = planner.totalDistanceTraveled

        return path, pathLength

    def _create_node(self, startNode, shortestPathLength, shortestPath):

        # x, y, heading
        carStateAtPoint = np.array([shortestPath['x'][-1], shortestPath['y'][-1], shortestPath['theta'][-1]])

        # initialize node
        newNode = self.NodeRRT(carStateAtPoint, shortestPathLength, shortestPath)
        newNode.name = 'Temp'
        newNode.parent = startNode
        newNode.path = shortestPath

        return newNode

    def _add_node(self, startNode, shortestPathLength, shortestPath, goal=False):

        nodeToAdd = self._create_node(startNode, shortestPathLength, shortestPath)
        startNode.numChildren += 1

        if goal:
            # adding node to goal list
            nodeToAdd.name = 'Goal {}'.format(len(self.goalNodeList)) 
            self.goalNodeList.append(nodeToAdd)
        else:
            # adding node to rrt* tree
            nodeToAdd.name = '{}'.format(len(self.nodeList)) 
            self.nodeList.append(nodeToAdd)
            
        return nodeToAdd

    def _calculate_entropy(self, path):

        # calculate entropy once per second
        sampleRate = 100
        instance = np.array([path['x'], path['y'], path['theta']]).transpose()
        instance[:, :2] /= 10.0
        instance[:, 2] -= math.pi
        instance[:, 2] /= math.pi
        instance = instance[np.newaxis, ::sampleRate, :]
        cumulativeEntropy = 0.0

        for i in range(1, instance.shape[1]):

            # give model a timestep and ask for prediction
            inputTensor = tf.constant(instance[:, i, :])
            logits = self.model(instance)[0, :].numpy()

            #entropy = -1.0 * np.sum(scipy.special.xlogy(logits, logits))
            entropy = stats.entropy(logits) / math.log(logits.shape[0])
            #entropy = 1.0 - logits[self.targetIdx]
            #entropy = logits[self.targetIdx]
            #print(self.targetIdx)
            #print(logits)

            # debug, entropy should be normalized to timesteps
            if entropy > (0.01 * sampleRate):
                print('entropy cannot be greater than 1')
                exit(1)

            cumulativeEntropy += entropy 

        # reset model after predicting entire path
        self.model.reset_states()

        return cumulativeEntropy 

    def _get_cost(self, node):
        
        if node.name == 'Root':
            return 0.0

        # sum costs from node to root
        distanceCost = node.pathLength

        fullPath = node.path.copy()

        #print(node.name, flush=True)
        while node.name != 'Root':

            node = node.parent
            #print(node.name, flush=True)
            distanceCost += node.pathLength

            fullPath['x'] = node.path['x'] + fullPath['x'] 
            fullPath['y'] = node.path['y'] + fullPath['y'] 
            fullPath['theta'] = node.path['theta'] + fullPath['theta'] 

        entropyCost = 0.0
        entropyCost = self._calculate_entropy(fullPath) 
        alpha = 1.0 
        #print('distanceCost:', distanceCost)
        #print('entropyCost:', math.exp(entropyCost))
        #print('totalCost:', distanceCost + (alpha * math.exp(entropyCost)), flush=True)

        cost = distanceCost + (alpha * entropyCost) 

        # debug, rrt* cannot have negative costs
        if cost < 0.0:
            print(cost)
            exit(1)

        return cost 

    def _rewire(self, newNode, nearestNodes):

        rewire = False

        for nearNode in nearestNodes:

            # cannot rewire root node
            if nearNode.name == 'Root':
                continue

            # Only consider heading for rewiring to non leaf nodes
            if nearNode.numChildren > 0:
                pathToNear, pathLengthToNear = self._calculate_dubins_path_length_final_heading(newNode, nearNode.position)
            else:
                euclideanDistance = abs(np.linalg.norm(newNode.position[:2] - nearNode.position[:2]))
                pathToNear = None
                if euclideanDistance > (2.0 * self.car.minTurningRadius):
                    pathLengthToNear = self._calculate_dubins_path_length(newNode, nearNode.position)
                    pathToNear = self._get_dubins_path(newNode, nearNode.position)

            if pathToNear is None:
                continue

            # check obstacle collisions
            collisionFree = self._is_point_reachable(newNode, nearNode, pathToNear)

            # get current cost to near node
            nearNodeCost = self._get_cost(nearNode)

            # check cost to near node with candidate rewiring
            candidateNode = self._create_node(newNode, pathLengthToNear, pathToNear)
            candidateCost = self._get_cost(candidateNode)

            # rewire from new node
            if candidateCost < nearNodeCost and collisionFree:

                rewire = True
                nearNode.parent.numChildren -= 1
                carStateAtPoint = np.array([pathToNear['x'][-1], pathToNear['y'][-1], pathToNear['theta'][-1]])
                nearNode.parent = newNode
                nearNode._set_position(carStateAtPoint)
                newNode.numChildren += 1
                nearNode.path = pathToNear
                nearNode.pathLength = pathLengthToNear

                if self.animate:
                    self._update_animation(point=nearNode.position, path=pathToNear, event='rewire', node=nearNode)

        return rewire

    def _connect_along_min_cost_path(self, point, nearestNodes, nearestNode):

        # initialize mincost to nearest node
        minPathStartNode = nearestNode 
        minPathLength = self._calculate_dubins_path_length(nearestNode, point)
        minPath = self._get_dubins_path(nearestNode, point)

        # get cost to new node, connecting from nearest node
        tempNode = self._create_node(nearestNode, minPathLength, minPath)
        minCost = self._get_cost(tempNode) 

        for node in nearestNodes:

            # path from near node to new point
            path = self._get_dubins_path(node, point)

            # check obstacle collision
            collisionFree = self._is_point_reachable(node, point, path)
            if not collisionFree:
                continue

            # calculate cost to new node, connecting from near node
            pathLength = self._calculate_dubins_path_length(node, point)
            tempNode = self._create_node(node, pathLength, path)
            costToNewPoint = self._get_cost(tempNode)

            # set min path
            if minPath is None or costToNewPoint < minCost:
                minPath = path
                minPathLength = pathLength
                minPathStartNode = node
                minCost = costToNewPoint

        return minPath, minPathLength, minPathStartNode

    def _find_nearest_nodes_to_new_point(self, randomPoint, goal=False):

        # setup to begin search 
        shortestPath = None
        shortestPathLength = None
        nearestNode = None
        nearestNodes = []
        
        # search tree for nearest neighbor to new point
        for node in self.nodeList:

            euclideanDistance = abs(np.linalg.norm(node.position[:2] - randomPoint[:2]))

            # ignore nodes that are too close to point or too far from point
            if euclideanDistance < (2.0 * self.car.minTurningRadius) or euclideanDistance > self.nearestNeighborRadius:
                continue
            #elif goal and euclideanDistance > self.goalNearestNeighborRadius:
                #continue

            # get dubins optimal path and length
            pathLength = self._calculate_dubins_path_length(node, randomPoint)

            # only care about long path case
            if pathLength < self.nearestNeighborRadius:# and (euclideanDistance / self.car.minTurningRadius) >= 4.0 * self.car.minTurningRadius:
                nearestNodes.append(node)

            # store shortest path
            if shortestPathLength is None or pathLength < shortestPathLength:
                path = self._get_dubins_path(node, randomPoint)
                shortestPathLength = pathLength
                shortestPath = path
                nearestNode = node

                if self.animate:
                    self._update_animation(point=randomPoint, path=shortestPath, event='candidate')

        return shortestPath, shortestPathLength, nearestNode, nearestNodes


    def _extend(self, point=None, goal=False):
 
        if point is None:
            point = self._sample_random_point()

        # get nearest nodes
        shortestPath, shortestPathLength, nearestNode, nearestNodes = self._find_nearest_nodes_to_new_point(point, goal)

        # check for viable path from parent node to new point
        isPointReachable = self._is_point_reachable(nearestNode, point, shortestPath)

        newNode = None

        # collision free
        if isPointReachable:

            # connect along a minimum-cost path
            minCostPath, minCostPathLength, minCostNode = self._connect_along_min_cost_path(point, nearestNodes, nearestNode)
            
            if goal:
                # check if this path is already in goal node list
                for node in self.goalNodeList:
                    if node.parent == minCostNode:
                        return node, None 
            
            # add node to tree/add goal node to goal node list
            newNode = self._add_node(minCostNode, minCostPathLength, minCostPath, goal)

            if goal:
                return newNode, nearestNodes
            elif self.animate:
                self._update_animation(point=point, path=minCostPath, event='valid path', node=newNode)

        elif self.animate:
            self._update_animation(point=point, path=shortestPath, event='invalid path')

        return newNode, nearestNodes

    def _get_nodes_leaf_to_root(self, node):

        # build node list from leaf to root
        nodesLeafToRoot = [node]

        while node.name != 'Root':
            nodesLeafToRoot.append(node.parent)
            node = node.parent

        return nodesLeafToRoot 

    def _set_min_cost_path_to_goal(self):

        minCostPath = None
        minCostGoal = None

        # check cost of every goal node
        for node in self.goalNodeList:

            cost = self._get_cost(node)
            if minCostPath is None or cost < minCostPath:
                minCostGoal = node
                minCostPath = cost

        if len(self.goalNodeList) > 0:
            self.minCostGoalPath = self._get_nodes_leaf_to_root(minCostGoal)

    def _sample_goal(self):

        # treat goal like a new node, try to reach from nearest neighbors
        newGoalNode, _ =  self._extend(self.target.position[:-1], goal=True)

        self._set_min_cost_path_to_goal()

        if self.animate:
            self._draw_min_cost_path_to_goal()

    def _get_final_path_start_to_goal(self):

        # build path backwards from goal to root
        finalPath = {'x': [], 'y': [], 'theta': []}

        for node in self.minCostGoalPath:

            if node.name != 'Root':

                # Debug: rewiring has not been propogated to goal path  
                if len(finalPath['x']) > 1 and abs(finalPath['x'][0]) - abs(node.path['x'][-1]) > 0.1:
                    print('DISCONTINUITY IN FINAL PATH!')
                    sys.exit(-1)

                finalPath['x'] = node.path['x'] + finalPath['x']
                finalPath['y'] = node.path['y'] + finalPath['y']
                finalPath['theta'] = node.path['theta'] + finalPath['theta']

        return finalPath

    ########################
    #### RRT* ALGORITHM ####
    ########################

    def simulate(self):
        
        # get target
        self._select_random_target()
        if self.animate:
            self._draw_target(self.target.position, 'lime')
            self.leg.remove()
        
        # ALGORITHM
        while self.iteration < self.maxIter:

            # extend tree
            newNode, nearestNodes = self._extend()
            if newNode is not None:
                rewire = self._rewire(newNode, nearestNodes)
                self._sample_goal()

            # debug
            if self.iteration % 25 == 0:
                print('iteration:', self.iteration, flush=True)

            self.iteration += 1

        print(self.iteration, flush=True)

        # show legend
        if self.animate:
            self._display_final_legend()
            plt.show()

        # package output path and target pair
        sample = {}
        sample['target'] = {'index': self.targetIdx }
        sample['path'] = self._get_final_path_start_to_goal()

        if len(sample['path']['x']) <= 1:
            return None

        return sample

    ###################
    #### ANIMATION ####
    ###################

    def _draw_point_and_path(self, point, pathToPlot, colorSelection, style = '-', goal=False ):

        plottedPoint, = self.ax.plot(point[0], point[1], color='black', marker = 'x', markersize=5)
        if goal:
            plottedPath, = self.ax.plot(pathToPlot['x'], pathToPlot['y'], color=colorSelection, linestyle=style, markersize = 2)
        else:
            plottedPath, = self.ax.plot(pathToPlot['x'], pathToPlot['y'], color=colorSelection, linestyle=style, markersize = 2)

        return plottedPoint, plottedPath

    def _draw_obstacle(self, obstacle):

        obs = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=False)
        self.ax.add_patch(obs)
        self.fig.canvas.draw()

    def _draw_target(self, target, colorSelection='blue'):

        tar = plt.Circle((target[0], target[1]), target[2], color=colorSelection, fill=False)
        self.ax.add_patch(tar)
        self.fig.canvas.draw()

    def on_press(self, event):

        print('press', event.key)

        if event.key == 'escape':

            plt.close('all')
            sys.exit(1)

    def _setup_animation(self):

        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.ax = self.fig.gca()
        self.ax.set_title('Dubins Car Adversarial RRT* - {}'.format(self.scene.name))
        plt.xlim(self.scene.dimensions['xmin'] , self.scene.dimensions['xmax'] )
        plt.ylim(self.scene.dimensions['ymin'] , self.scene.dimensions['ymax'] )
        self.ax.set_aspect('equal')
        self.ax.set_ylabel('Y-distance(M)')
        self.ax.set_xlabel('X-distance(M)')
        self.ax.plot(self.root.x, self.root.y, 'x')

        legend_elements = [ Line2D([0], [0], marker='o', linestyle='', fillstyle='none', label='Non-selected Target',
                            markeredgecolor='blue', markersize=15),
                            Line2D([0], [0], marker='o', fillstyle='none', linestyle='', label='Selected Target',
                            markeredgecolor='green', markersize=15),
                            Line2D([0], [0], marker='o', fillstyle='none', linestyle='', label='Obstacle',
                            markeredgecolor='red', markersize=15)]

        for obstacle in self.scene.obstacles:

            self._draw_obstacle(obstacle)

        for target in self.scene.targets:

            self._draw_target(target)

        self.leg = self.ax.legend(handles=legend_elements, loc='best')

    def _write_caption(self, event):

        x = self.scene.dimensions['xmin'] + 0.5
        y = self.scene.dimensions['ymax'] - 1.0
        if event == 'candidate':
            self.text = plt.text(x, y, 'Calculating shortest path from tree to new point')
        elif event == 'valid path':
            self.text = plt.text(x, y, 'Path is possible without collisions')
        elif event == 'invalid path':
            self.text = plt.text(x, y, 'Path causes collision')
        elif event == 'reached target':
            self.text = plt.text(x, y, 'Path connects tree root to target')
        elif event == 'rewire':
            self.text = plt.text(x, y, 'Rewiring tree...')
        elif event == 'Goal':
            self.text = plt.text(x, y, 'Sampling goal...')

    def _draw_min_cost_path_to_goal(self):

        for plottedPath in self.plottedPathToGoal:
            plottedPath.remove()

        self.plottedPathToGoal = []

        for node in self.minCostGoalPath:
            if node.name != 'Root':
                plottedPath = self._update_animation(point=node.position, path=node.path, event='Goal', node=node)
                if plottedPath is not None:
                    self.plottedPathToGoal.append(plottedPath)


    def _draw_final_path_to_goal(self):

        if self.plottedPathToGoal is not None:
            self.plottedPathToGoal.remove()

        finalPath = {'x':[], 'y':[], 'theta':[]} 
        for path in self.pathToGoal:

            finalPath['x'].extend(path['x'])
            finalPath['y'].extend(path['y'])
            finalPath['theta'].extend(path['theta'])

        plottedGoal, plottedPathToGoal= self._draw_point_and_path(self.goal, finalPath, 'pink')
        self.plottedPathToGoal = plottedPathToGoal

    def _display_final_legend(self):

        legend_elements = [ Line2D([0], [0], marker='o', linestyle='', fillstyle='none', label='Non-selected Target',
                        markeredgecolor='blue', markersize=15),
                        Line2D([0], [0], marker='o', fillstyle='none', linestyle='', label='Selected Target',
                        markeredgecolor='green', markersize=15),
                        Line2D([0], [0], marker='o', fillstyle='none', linestyle='', label='Obstacle',
                        markeredgecolor='red', markersize=15),
                        Line2D([0], [0], marker='_', fillstyle='none', linestyle='', label='RRT Tree Branches',
                        markeredgecolor='pink', markersize=15),
                        Line2D([0], [0], marker='_', fillstyle='none', linestyle='', label='RRT* Rewiring',
                        markeredgecolor='green', markersize=15),
                        Line2D([0], [0], marker='_', fillstyle='none', linestyle='', label='Final Path',
                        markeredgecolor='blue', markersize=15)
                        ]
        leg = self.ax.legend(handles=legend_elements, loc='best')
    
    def _update_animation(self, point, path, event, node=None):
 
        if path is None:
            return 

        eventToColorDict = {'candidate': 'orange',\
                'valid path':'pink',\
                'invalid path': 'red',\
                'Goal': 'blue',\
                'rewire': 'green'}

        pathToPlot = {'x': [], 'y':[]}
        i = 0
        for x, y in zip(path['x'], path['y']):

            if i % 1000 == 0:
                pathToPlot['x'].append(x)
                pathToPlot['y'].append(y)
            i += 1       

        plottedPoint, plottedPath = self._draw_point_and_path(point, path, eventToColorDict[event])
        
        self._write_caption(event)

        if event == 'rewire':
            if node.name != 'Root':
                node.plottedPath.remove()
                node.plottedPath = plottedPath

        #plt.savefig('./saved-images/fig-{}.png'.format(self.imgcount))
        plt.pause(0.00001)
        self.imgcount += 1
        self.text.remove()

        if event == 'invalid path' or event == 'candidate':
            plottedPoint.remove()
            plottedPath.remove()
            return None
        elif event == 'Goal':
            return plottedPath

        node.plottedPath = plottedPath
        return None
