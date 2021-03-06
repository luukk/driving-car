import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon, Point
from math import exp
import time as tm
import traceback as tb
import simpylc as sp
import numpy as np
from torch import nn, LongTensor, FloatTensor, ones, tensor, zeros
from numpy.random import randint
from numpy.random import rand
import random as rnd
import time
import sys
import copy
import matplotlib.pyplot as plt

from driving.main import myprint

class NeuralNet:
    def __init__(self):
        self.NeuralNet = None

        self.initializeNetwork()

    def initializeNetwork(self):
        input_size = 4
        hidden_sizes = [2, 2]
        output_size = 2# Build a feed-forward network
        model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0], bias=False), #
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=False),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=False),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=False),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], output_size, bias=False),
            nn.Tanh())

        self.NeuralNet = model


    def forward(self, distance1, angle1, distance2, angle2):
        x_data = FloatTensor([[distance1, angle1, distance2, angle2]])
        predict = self.NeuralNet(x_data)
        na = predict.detach().numpy()

        return self.sigmoid(na[0][0]), na[0][1] #velocity, angle. adjust weights every iteration. randomly mutate weights

    def getWeights(self):
        weights = []
        for name, param in self.NeuralNet.named_parameters():
            # print(name,param, param.data)
            weights.append(param.data)

        return weights

    def setWeights(self, s):
        for i, x in enumerate(s):
            self.changeWeight(i,x)

    def changeWeight(self, given_index, tensor):
        for index, (name, param) in enumerate(self.NeuralNet.named_parameters()):
            if index == given_index:
                # print(param.data)
                param.data = tensor

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

def dumpDat(dat):
    print(dat)

class LidarPilot:
    maxTrainingTime = 150
    popSize = 9
    population = []
    maxGens = 50
    crossover_rate = 0.9
    currentBest = None
    polygonOut = [(-4.5, -7.5), (-2.5, -7.5), (-1.0, -7.0), (1.0, -6.5), (3.0, -6.5), (5.0, -7.0), (7.0, -6.5), (7.0, -4.5), (7.0, -2.5), (6.5, -0.5), (6.0, 1.5), (6.0, 3.5), (6.5, 5.5), (6.0, 7.0), (4.0, 7.5), (2.0, 7.0), (1.0, 5.5), (0.5, 3.5), (0.5, 1.5), (0.5, -0.5), (-0.5, -2.0), (-2.0, -2.5), (-3.5, -2.0), (-3.5, -0.5), (-3.0, 1.0), (-2.5, 2.5), (-2.5, 4.0), (-2.5, 6.0), (-3.5, 7.0), (-5.0, 7.0), (-7.0, 6.5), (-7.5, 5.0), (-7.0, 3.0), (-7.5, 1.0), (-7.5, -1.0), (-7.0, -2.5), (-7.5, -4.5), (-6.5, -6.5), (-4.5, -7.5)]
    polygonIn = [(-4.5, -6.5), (-3.0, -6.5), (-1.5, -6.0), (0.5, -5.5), (3.0, -5.5), (4.5, -6.0), (6.0, -5.5), (6.0, -4.0), (6.0, -2.5), (5.5, -0.5), (5.0, 1.5), (5.0, 3.5), (5.5, 5.5), (5.0, 6.5), (4.0, 6.5), (2.5, 6.0), (2.0, 5.5), (1.5, 3.5), (1.5, 1.5), (1.5, -0.5), (0.5, -2.5), (-1.5, -3.5), (-3.0, -3.5), (-4.5, -2.5), (-4.5, -0.5), (-4.0, 1.0), (-3.5, 2.5), (-3.5, 4.0), (-3.5, 5.5), (-4.0, 6.0), (-5.0, 6.0), (-6.0, 5.5), (-6.5, 4.5), (-6.0, 3.0), (-6.5, 1.0), (-6.5, -1.0), (-6.0, -2.5), (-6.5, -4.5), (-5.5, -6.0), (-4.5, -6.5)]
    allPolygons = None
    train_loss = []
    highestFitnessScore = 0
    
    lidarDistWithAngle = []


    # [tensor([[-0.4486, -0.7071],
    #     [-0.4354,  0.3067]]), tensor([[-0.4534,  0.4454],
    #     [-0.6106, -0.0016]]), tensor([[ 0.6316, -0.4360],
    #     [ 0.2829,  0.6848]])]
    """
    maxTime is the max amout of seconds the loop can run for
    """
    def __init__ (self):
        self.allPolygons = Polygon(np.concatenate((self.polygonOut, self.polygonIn)))
        self.population = [NeuralNet() for x in range(self.popSize)]
       
        for x in range(self.maxGens):
            scores = [self.testNet(x) for x in self.population]    
            
            sortedPopulation = self.sortPopulation(self.population, scores)

            print("sorted pop: ", sortedPopulation)

            genBestNetwork = sortedPopulation[0]
            print("best network weights", genBestNetwork[0].getWeights(), sortedPopulation[0])

            self.train_loss.append(genBestNetwork[1])

            if genBestNetwork[1] > self.highestFitnessScore:
                self.highestFitnessScore = genBestNetwork[1]
                self.currentBest = genBestNetwork[0]

            p1, p2 = sortedPopulation[0][0], sortedPopulation[1][0]
            print("p1: ", p1)
            print("p2: ", p2)

            c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
            print("c1: ", c1)
            print("c2: ", c2)
            children = list()
            for c in self.crossover(c1, c2):
                m = self.mutation(c)
                children.append(m)

            del sortedPopulation[-1]    
            del sortedPopulation[-2]    

            self.population = [x[0] for x in sortedPopulation] + children

            print("population: ", self.population)

            print("iteration: ", x, "with fitness score: ", genBestNetwork[1])
        
        print("training done. Running with best car")
        self.testNet(self.currentBest)

        plt.plot(list(range(0, self.maxGens)), train_loss)
        plt.xlabel('epochs')
        plt.ylabel('fitness score')
        plt.show()

    def mut(self, c):
        return c
            
    def mutation(self, network, num: int = 4, probability: float = 0.5):
        for _ in range(num):
            # if rnd.random() > probability:
            randomLayer = randint(0, len(network.getWeights()))
    
            tensor = network.getWeights()[randomLayer]
            
            randomEdge = randint(0, len(tensor))
            tensorEdge = tensor[randomEdge]
            randomWeight = randint(0, len(tensorEdge)) 
            tensorEdge[randomWeight] = rnd.uniform(-1, 1)
 
        return network

    def crossover(self, k1, k2):
        p1, p2 = k1.getWeights(), k2.getWeights()
        c1, c2 = p1.copy(), p2.copy()
        if rand() < self.crossover_rate:
            pt = randint(1, len(p1))
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        k1.setWeights(c1)
        k2.setWeights(c2)
        return [k1, k2]

    def sortPopulation(self, population, scores, k=3):
        sortedPopulation = list()
        sort_scores_by_index = [i[0] for i in sorted(enumerate(scores), key=lambda x:x[1])]
        
        for i in reversed(sort_scores_by_index):
            sortedPopulation.append((population[i], scores[i]))

        return sortedPopulation      

    def testNet(self, net):
        endTime = time.time() + self.maxTrainingTime

        while time.time() < endTime:        
            self.sweep()
            # tinput = (self.sortedLidarWithAngle[0] + self.sortedLidarWithAngle[2]) / 2
            # tangle = (self.sortedLidarWithAngle[1] + self.sortedLidarWithAngle[3]) / 2
            # print(self.sortedLidarWithAngle[1], self.sortedLidarWithAngle[3], self.sortedLidarWithAngle[0], self.sortedLidarWithAngle[2])
            v,a = net.forward(round(self.sortedLidarWithAngle[0], 1), self.sortedLidarWithAngle[1] /15, round(self.sortedLidarWithAngle[2], 1), self.sortedLidarWithAngle[3]  /15)
            self.setNextMove(v,a)
            self.output()
            #and not self.checkCollision()
            if not self.x():
                score = self.scoreAndReset()
                print(score)
                return score
            tm.sleep (0.02)
          
        f = self.scoreAndReset() 
        print("score: ", f)
        return f

    def x(self):
        currentX = sp.evaluate(sp.world.physics.positionX)
        currentY = sp.evaluate(sp.world.physics.positionY)
        point = Point(round(currentX, 2), round(currentY, 2))
        if not self.allPolygons.contains(point):
            return False
        return True

    def scoreAndReset(self) -> int:
        distance = self.getDistanceTravelled()
        self.resetCar()
        return round(distance, 2)

    def getDistanceTravelled(self) -> int:
        currentX = sp.evaluate(sp.world.physics.positionX)
        currentY = sp.evaluate(sp.world.physics.positionY)

        point = Point(currentX, currentY)

        closestPoint = self.findClosestPoint(point, self.polygonOut)
        return (closestPoint[0] + closestPoint[1])

    def checkCollision(self) -> bool:
        return sp.evaluate(sp.world.visualisation.collided)

    def findClosestPoint(self, carPos, polyList):
        closestPoint = tuple()
        secondClosest = tuple()

        for index, trackPoint in enumerate(polyList):
            dist = np.linalg.norm(np.asarray(trackPoint) - np.asarray(carPos))

            if not closestPoint or dist < closestPoint[0]:
                closestPoint = (dist, index)
            
            if not secondClosest or (dist > secondClosest[0] and dist < closestPoint[0]):
                secondClosest = (dist, index) 

        return closestPoint

    def resetCar(self) -> None:
            sp.world.physics.steeringAngle.set(0) 
            sp.world.physics.midSteeringAngle.set(0) 
            sp.world.physics.inverseMidCurveRadius.set(0) 
            sp.world.physics.midAngularVelocity.set(0) 
            sp.world.physics.attitudeAngle.set(50) 
            sp.world.physics.courseAngle.set(50) 

            sp.world.physics.positionX.set(2) 
            sp.world.physics.positionY.set(-6)
    
    def isOnTrack(self):
        pass
        
    def sweep (self):   # Control algorithm to be tested
        self.lidarDistances = sp.world.visualisation.lidar.distances
        self.lidarHalfApertureAngle = sp.world.visualisation.lidar.halfApertureAngle
        self.nearestObstacleDistance = sp.finity
        self.nearestObstacleAngle = 0
        
        self.nextObstacleDistance = sp.finity
        self.nextObstacleAngle = 0

        for lidarAngle in range (-self.lidarHalfApertureAngle, self.lidarHalfApertureAngle):
            lidarDistance = self.lidarDistances [lidarAngle]
            # self.lidarDistWithAngle.append((lidarDistance, lidarAngle))
            
            if lidarDistance < self.nearestObstacleDistance:
                self.nextObstacleDistance =  self.nearestObstacleDistance
                self.nextObstacleAngle = self.nearestObstacleAngle
                
                self.nearestObstacleDistance = round(lidarDistance, 2) 
                self.nearestObstacleAngle = lidarAngle

            elif lidarDistance < self.nextObstacleDistance:
                self.nextObstacleDistance = round(lidarDistance, 2)
                self.nextObstacleAngle = lidarAngle
            
        self.sortedLidarWithAngle = [self.nearestObstacleDistance, self.nearestObstacleAngle, self.nextObstacleDistance, self.nextObstacleAngle]
        # self.sortedLidarWithAngle = sorted(self.lidarDistWithAngle, key=lambda e: e[0], reverse=False)
        # print(self.sortedLidarWithAngle[:20])
        # print(self.nearestObstacleDistance)
        # print(self.nextObstacleDistance)
        # print("-----------------")


        # self.targetObstacleDistance = (self.nearestObstacleDistance + self.nextObstacleDistance) / 2
        # self.targetObstacleAngle = (self.nearestObstacleAngle + self.nextObstacleAngle) / 2
    
    def setNextMove(self, velocity: int, angle: int) -> None:
        self.targetVelocity = velocity
        self.steeringAngle = angle * 90

    def output (self):  # Output to simulator
        sp.world.physics.steeringAngle.set (self.steeringAngle)
        sp.world.physics.targetVelocity.set (self.targetVelocity)
        
