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

import time as tm
import traceback as tb

import simpylc as sp

import numpy as np
# import keras as ks


from torch import nn
from driving.main import NeuralNet

# neuralNet = NeuralNet()

# import os
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import keras
# from keras.models import Sequential
# from keras.layers import Dense# Neural network




class LidarPilot:
    def __init__ (self):
        # print ('Use up arrow to start, down arrow to stop')
        
        self.driveEnabled = False

        self.neuralNet = NeuralNet()

        while True:
            self.input ()
            self.sweep ()
            self.output ()
            tm.sleep (0.02)
        
    def input (self):   # Input from simulator
        # key = sp.getKey ()
        
        # if key == 'KEY_UP':
        #     self.driveEnabled = True
        # elif key == 'KEY_DOWN':
        #     self.driveEnabled = False
        
        self.lidarDistances = sp.world.visualisation.lidar.distances
        self.lidarHalfApertureAngle = sp.world.visualisation.lidar.halfApertureAngle
        
    def sweep (self):   # Control algorithm to be tested
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

        nextmove = self.neuralNet.forward(self.targetObstacleDistance, self.targetObstacleAngle)

        self.targetVelocity = nextmove[0]
        self.steeringAngle = nextmove[1] * 90
        
        # self.steeringAngle = self.targetObstacleAngle
        # self.targetVelocity = (sp.abs (90 - self.steeringAngle) / 80) if self.driveEnabled else 0
    
    def output (self):  # Output to simulator
        sp.world.physics.steeringAngle.set (self.steeringAngle)
        sp.world.physics.targetVelocity.set (self.targetVelocity)
        
