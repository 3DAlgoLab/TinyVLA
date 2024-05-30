import time

import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("scara_bot.urdf")
p.setGravity(0, 0, -9.8)

for i in range(p.getNumJoints(robot)):
    print(p.getJointInfo(robot, i))

target_arm1 = -0.46
target_arm2 = 0.24

while True:
    p.setJointMotorControlArray(
        robot, [1, 2], p.POSITION_CONTROL, targetPositions=[target_arm1, target_arm2]
    )
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
    # time.sleep(0.1)
