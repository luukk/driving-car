import os
import sys as ss

ss.path.append (os.path.abspath ('../../../..')) # If you want to store your simulations somewhere else, put SimPyLC in your PYTHONPATH environment variable

import simpylc as sp

import control as ct
import keyboard_pilot as kp
import lidar_pilot as lp
import lidar_pilot_sp as ls
import physics as ps
import visualisation as vs
import timing as tm

class ControlCar:
    def __init__(self):
        self.isCollided = False

    def initWorld(self):
        sp.World (
            # ct.Control,
            # kp.KeyboardPilot,
            lp.LidarPilot,
            # ls.LidarPilotSp,
            ps.Physics,
            vs.Visualisation,
            # tm.Timing
        )

    def checkCollision(self):
        print("herro")
        # print(sp.Scene.collided)       
        



carController = ControlCar()
carController.initWorld()
# print("herro")
# while(True):
#     print()
#     carController.checkCollision()