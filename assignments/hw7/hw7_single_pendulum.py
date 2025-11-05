## =============================================================================
## PROJECT CHRONO - http:##projectchrono.org
##
## Copyright (c) 2014 projectchrono.org
## All right reserved.
##
## Use of this source code is governed by a BSD-style license that can be found
## in the LICENSE file at the top level of the distribution and at
## http://projectchrono.org/license-chrono.txt.
##
## =============================================================================

import pychrono as chrono
import pychrono.irrlicht as vsg
import numpy as np

system = chrono.ChSystemNSC()
system.SetGravitationalAcceleration(chrono.ChVector3d(0, -9.81, 0))
## Create the rigid body of the single pendulum mechanical system.
##    For the body, specify:
##    - a unique identifier
##    - mass and moments of inertia
##    - position and orientation of the (centroidal) body frame
##    - visualization assets (defined with respect to the body frame)

## Ground (as a box)
ground = chrono.ChBodyEasyBox(1.0, 0.1, 1.0, 1000, True, True)
ground.SetPos(chrono.ChVector3d(0,0,0))
ground.SetFixed(True)
system.Add(ground)

## specify initial orientation and position of COG of the single pendulum
singpend_rot = chrono.ChQuaterniond()
singpend_rot.SetFromAngleAxis(-chrono.CH_PI / 4.0, chrono.ChVector3d(0, 0, 1))
singpend_pos = chrono.ChVector3d(np.sqrt(2)/2, -np.sqrt(2)/2, 0)

# length of bar 1 and 2 in x' direction
bar_length = 2

## Single pendulum (as a box)
singpend = chrono.ChBodyEasyBox(bar_length, 0.1, 0.1, 1000, True, True)
singpend.SetPos(singpend_pos)
singpend.SetRot(singpend_rot)
singpend.SetName("singpend")
system.Add(singpend)
singpend.GetVisualShape(0).SetColor(chrono.ChColor(0.6, 0.2, 0.2))


## Revolute joint between ground and single pendulum.
motor = chrono.ChLinkMotorRotationAngle()
motor.SetName("motor")
motor.Initialize(singpend,
                 ground,
                 chrono.ChFramed(chrono.ChVector3d(0, 0, 0)))

my_angularspeed = chrono.ChFunctionSine(chrono.CH_PI / 4, 2, chrono.CH_PI / 2)
# combine the two functions
motor.SetMotorFunction(my_angularspeed)
system.AddLink(motor)

# Prepare visualization with VSG
vis = vsg.ChVisualSystemIrrlicht()
vis.AttachSystem(system)
vis.SetWindowSize(1024,768)
vis.SetWindowTitle('single pendulum demo')
vis.AddCamera(chrono.ChVector3d(-0.5, 0.5, 5), chrono.ChVector3d(0,0,0))
vis.SetCameraVertical(chrono.CameraVerticalDir_Y)
#vis.EnableSkyBox()

torqs = []
vis.Initialize()
while vis.Run():
    vis.BeginScene()
    vis.Render()
    vis.EndScene()
    system.DoStepDynamics(1e-3)
    print(motor.GetMotorTorque())

