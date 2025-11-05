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
#import pychrono.vsg3d as vsg
import pychrono.irrlicht as chronoirr
import numpy as np

system = chrono.ChSystemNSC()
system.SetGravitationalAcceleration(chrono.ChVector3d(0, -9.81, 0))
## Create the rigid bodies of the slider-crank mechanical system.
##    For each body, specify:
##    - a unique identifier
##    - mass and moments of inertia
##    - position and orientation of the (centroidal) body frame
##    - visualization assets (defined with respect to the body frame)

## Ground (as a box)
ground = chrono.ChBodyEasyBox(1.0, 0.1, 1.0, 1000, True, True)
ground.SetPos(chrono.ChVector3d(0,0,0))
ground.SetFixed(True)
system.Add(ground)

## specify inital orientation and position of COG of the crank
crank_rot = chrono.ChQuaterniond()
crank_rot.SetFromAngleAxis(chrono.CH_PI / 4.0, chrono.ChVector3d(0, 0, 1))
crank_pos = chrono.ChVector3d(0.5, 0.5, 0)

## create variables for the inital orientation and position of COG of the rod
rod_rot = chrono.ChQuaterniond()
rod_rot.SetFromAngleAxis(-chrono.CH_PI / 6.0, chrono.ChVector3d(0, 0, 1))
rod_pos = chrono.ChVector3d(np.sqrt(3)/2 + 1, 0.5, 0)

# length of bar 1 and 2 in x' direction
bar1_length = np.sqrt(2)
bar2_length = 2

## Crank (as a box)
crank = chrono.ChBodyEasyBox(bar1_length, 0.1, 0.1, 1000, True, True)
crank.SetPos(crank_pos)
crank.SetRot(crank_rot)
crank.SetName("crank")
system.Add(crank)
crank.GetVisualShape(0).SetColor(chrono.ChColor(0.6, 0.2, 0.2))

## Rod (as a box)
rod = chrono.ChBodyEasyBox(bar2_length, 0.1, 0.1, 1000, True, True)
rod.SetPos(rod_pos)
rod.SetRot(rod_rot)
rod.SetName("rod")
system.Add(rod)
rod.GetVisualShape(0).SetColor(chrono.ChColor(0, 0, 1))

## Create joint constraints.
## All joint frames are specified in the global frame.

## Revolute joint between ground and crank.
motor = chrono.ChLinkMotorRotationSpeed()
motor.SetName("motor")
motor.Initialize(crank,
                 ground,
                 chrono.ChFramed(chrono.ChVector3d(0, 0, 0)))
my_angularspeed = chrono.ChFunctionConst(chrono.CH_PI) # ang.speed: 180°/s
motor.SetMotorFunction(my_angularspeed)
system.AddLink(motor)

# create a joint between crank and rod
joint_a = chrono.ChLinkLockRevolute()  # or other joint type
joint_a.SetName("joint_a")
joint_a.Initialize(crank, 
                   rod, 
                   chrono.ChFramed(chrono.ChVector3d(1, 1, 0)))
system.AddLink(joint_a)

# Prepare visualization with VSG
#vis = vsg.ChVisualSystemVSG()
vis = chronoirr.ChVisualSystemIrrlicht()
vis.AttachSystem(system)
vis.SetWindowSize(1024,768)
vis.SetWindowTitle('slider crank demo')
vis.AddCamera(chrono.ChVector3d(-0.5, 0.5, 5), chrono.ChVector3d(0,0,0))
vis.SetCameraVertical(chrono.CameraVerticalDir_Y)
#vis.EnableSkyBox()
# Irrlicht
vis.AddSkyBox()
vis.AddTypicalLights()

vis.Initialize()
while vis.Run():
    vis.BeginScene()
    vis.Render()
    vis.EndScene()
    system.DoStepDynamics(1e-4)
