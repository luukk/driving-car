# ====== Legal notices
#
# Copyright (C) 2013 - 2020 GEATEC engineering
#
# This program is free software.
# You can use, redistribute and/or modify it, but only under the terms stated in the QQuickLicence.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY, without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the QQuickLicence for details.
#
# The QQuickLicense can be accessed at: http://www.geatec.com/qqLicence.html
#
# __________________________________________________________________________
#
#
#  THIS PROGRAM IS FUNDAMENTALLY UNSUITABLE FOR CONTROLLING REAL SYSTEMS !!
#
# __________________________________________________________________________
#
# It is meant for training purposes only.
#
# Removing this header ends your licence.
#
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon, Point
from math import exp
import time as tm
import traceback as tb
import simpylc as sp
import numpy as np
from torch import nn, LongTensor, FloatTensor, ones
from numpy.random import randint
from numpy.random import rand
import random as rnd
import time
import sys

from driving.main import myprint



class NeuralNet:
    def __init__(self):
        self.NeuralNet = None

        self.initializeNetwork()

    def initializeNetwork(self):
        input_size = 2
        hidden_sizes = [2, 2]
        output_size = 2# Build a feed-forward network
        model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.Tanh())

        self.NeuralNet = model


    def forward(self, distance, angle):
        x_data = FloatTensor([[distance, angle]])
        predict = self.NeuralNet(x_data)
        na = predict.detach().numpy()

        return self.sigmoid(na[0][0]), na[0][1] #velocity, angle. adjust weights every iteration. randomly mutate weights

    def getWeights(self):
        weights = []
        for name, param in self.NeuralNet.named_parameters():
            weights.append(param.data)

        return weights
            # print(param.data)

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
    popSize = 50
    population = []
    genSize = 20
    maxGens = 10
    crossover_rate = 0.9
    hostPort = 8083
    socketServer = None
    currentBest = None
    polygonOut = [(-4.5, -7.5), (-2.5, -7.5), (-1.0, -7.0), (1.0, -6.5), (3.0, -6.5), (5.0, -7.0), (7.0, -6.5), (7.0, -4.5), (7.0, -2.5), (6.5, -0.5), (6.0, 1.5), (6.0, 3.5), (6.5, 5.5), (6.0, 7.0), (4.0, 7.5), (2.0, 7.0), (1.0, 5.5), (0.5, 3.5), (0.5, 1.5), (0.5, -0.5), (-0.5, -2.0), (-2.0, -2.5), (-3.5, -2.0), (-3.5, -0.5), (-3.0, 1.0), (-2.5, 2.5), (-2.5, 4.0), (-2.5, 6.0), (-3.5, 7.0), (-5.0, 7.0), (-7.0, 6.5), (-7.5, 5.0), (-7.0, 3.0), (-7.5, 1.0), (-7.5, -1.0), (-7.0, -2.5), (-7.5, -4.5), (-6.5, -6.5), (-4.5, -7.5)]
    polygonIn = [(-4.5, -6.5), (-3.0, -6.5), (-1.5, -6.0), (0.5, -5.5), (3.0, -5.5), (4.5, -6.0), (6.0, -5.5), (6.0, -4.0), (6.0, -2.5), (5.5, -0.5), (5.0, 1.5), (5.0, 3.5), (5.5, 5.5), (5.0, 6.5), (4.0, 6.5), (2.5, 6.0), (2.0, 5.5), (1.5, 3.5), (1.5, 1.5), (1.5, -0.5), (0.5, -2.5), (-1.5, -3.5), (-3.0, -3.5), (-4.5, -2.5), (-4.5, -0.5), (-4.0, 1.0), (-3.5, 2.5), (-3.5, 4.0), (-3.5, 5.5), (-4.0, 6.0), (-5.0, 6.0), (-6.0, 5.5), (-6.5, 4.5), (-6.0, 3.0), (-6.5, 1.0), (-6.5, -1.0), (-6.0, -2.5), (-6.5, -4.5), (-5.5, -6.0), (-4.5, -6.5)]
    allPolygons = None

    """
    maxTime is the max amout of seconds the loop can run for
    """
    def __init__ (self):
        myprint(" yee")
        self.allPolygons = Polygon(np.concatenate((self.polygonOut, self.polygonIn)))
        self.population = [NeuralNet() for x in range(self.popSize)]
        self.currentBest = self.testNet(self.population[0])
        
        for x in range(self.maxGens):
            scores = [self.testNet(x) for x in self.population]    
            for score in scores:
                if score > self.currentBest:
                    self.currentBest = score
            selected = [self.select(self.population, scores) for _ in range(self.genSize)]
            children = list()

            for i in range(0, self.genSize, 2):
                p1, p2 = selected[i], selected[i+1]
                for c in self.crossover(p1,p2):
                    m = self.mut(c)
                    children.append(m)
                self.population = children
            
    def mut(self, c):
        print(c)
        return c

    def crossover(self, p1, p2):
        c1, c2 = p1.copy(), p2.copy()
        if rand() < r_cross:
            pt = randint(1, len(p1)-2)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    def select(self, population, scores, k=3):
        selection = randint(len(population))
        for n in randint(0, len(population), k-1):
            if scores[n] < scores[selection]:
                selection = n
        return population[selection]         

    def testNet(self, net):
        endTime = time.time() + 20

        while time.time() < endTime:
        
            self.input()
            self.sweep()
            v,a = net.forward(self.targetObstacleDistance, self.targetObstacleAngle)
            self.setNextMove(v,a)
            self.output()

            if not self.x() and not self.checkCollision():
                return self.scoreAndReset()
            tm.sleep (0.02)
            
        f = self.scoreAndReset() 
        myprint(f)
        return f

    def x(self):
        currentX = sp.evaluate(sp.world.physics.positionX)
        currentY = sp.evaluate(sp.world.physics.positionY)
        point = Point(currentX, currentY)
        if not self.allPolygons.contains(point):
            return False
        return True

    def scoreAndReset(self) -> int:
        distance = self.getDistanceTravelled()
        self.resetCar()
        return distance

    def getDistanceTravelled(self) -> int:
        currentX = sp.evaluate(sp.world.physics.positionX)
        currentY = sp.evaluate(sp.world.physics.positionY)

        point = Point(currentX, currentY)

        closestPoint = self.findClosestPoint(point, self.polygonOut)
        return 100 * closestPoint[1] / len(self.polygonOut)

    def checkCollision(self) -> bool:
        return sp.evaluate(sp.world.visualisation.collided)

    def findClosestPoint(self, carPos, polyList):
        closestPoint = tuple()

        for index, trackPoint in enumerate(polyList):
            dist = np.linalg.norm(np.asarray(trackPoint) - np.asarray(carPos))

            if not closestPoint or dist < closestPoint[0]:
                closestPoint = (dist, index)
        return closestPoint

    def resetCar(self) -> None:
            sp.world.physics.positionX.set(-3.5) 
            sp.world.physics.positionY.set(-7)
    
    def isOnTrack(self):
        pass


    def input (self):
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
            
            if lidarDistance < self.nearestObstacleDistance:
                self.nextObstacleDistance =  self.nearestObstacleDistance
                self.nextObstacleAngle = self.nearestObstacleAngle
                
                self.nearestObstacleDistance = lidarDistance 
                self.nearestObstacleAngle = lidarAngle

            elif lidarDistance < self.nextObstacleDistance:
                self.nextObstacleDistance = lidarDistance
                self.nextObstacleAngle = lidarAngle
           
        self.targetObstacleDistance = (self.nearestObstacleDistance + self.nextObstacleDistance) / 2
        self.targetObstacleAngle = (self.nearestObstacleAngle + self.nextObstacleAngle) / 2
    
    def setNextMove(self, velocity: int, angle: int) -> None:
        self.targetVelocity = velocity
        self.steeringAngle = angle * 90

    def output (self):  # Output to simulator
        sp.world.physics.steeringAngle.set (self.steeringAngle)
        sp.world.physics.targetVelocity.set (self.targetVelocity)
        
