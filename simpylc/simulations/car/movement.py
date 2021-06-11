import sys
import os
import sys as ss

ss.path.append (os.path.abspath ('../../..')) # If you want to store your simulations somewhere else, put SimPyLC in your PYTHONPATH environment variable
ss.path.append(os.path.abspath("~/Workdir/self-driving/SimPyLC-lidar_car/simpylc/simulations/car/"))

import simpylc as sp
import control as ct
import keyboard_pilot as kp
import timing as tm
import lidar_pilot as lp
import physics as ps
import visualisation as vs


sp.World (
    # ct.Control,
    # kp.KeyboardPilot,
    lp.LidarPilot,
    # ls.LidarPilotSp,
    ps.Physics,
    vs.Visualisation,
    # tm.Timing
)