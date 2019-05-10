import gym
import gym_gazebo
import time
import random
import numpy as np
import rospy
import roslaunch

from random import randint
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from controller_manager_msgs.srv import SwitchController
from gym.utils import seeding


"""Data generation for the case of a single block pick and taking to a goal position"""

ep_returns = []
actions = []
observations = []
rewards = []
infos = []

def main():
    env = gym.make('GazeboWAMemptyEnv-v1')
    #env.seed()
    numItr = 100
    initStateSpace = "random"

    print("Reset!")
    env.reset()
    time.sleep(1)
    while len(actions) < numItr:
        print("Reset!")
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)

    fileName = "data_wam_reach"
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    fileName += ".npz"
    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos)


def has_reached(position, threshold):
    truth = 1
    for i in range(len(position)):
        truth += (np.absolute(position[i]) < threshold)
    return not truth==6

def goToGoal(env, lastObs):
    goalPosition = env.goalJS
    goalJS = [goalPosition[0], goalPosition[1], goalPosition[2], goalPosition[3], goalPosition[4]]
    objectPosInitialJS = [env.objInitialJS[0], env.objInitialJS[1], env.objInitialJS[2], env.objInitialJS[3], env.objInitialJS[4]]
    gripperPosJS = env.lastObservationJS


    object_rel_pos_JS = objectPosInitialJS - gripperPosJS

    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    timeStep = 0

    episodeObs.append(lastObs)

    while has_reached(object_rel_pos_JS, 0.05) and (timeStep <= env._max_episode_steps - 1): # go to object
        action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001)]

        gripperPosJS = env.lastObservationJS
        object_rel_pos_JS =  objectPosInitialJS - gripperPosJS


        for i in range(len(object_rel_pos_JS)):
            action[i] = object_rel_pos_JS[i]*1


        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1


        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)


    while env.gripperState is 0 and (timeStep <= env._max_episode_steps - 1): # pick it up
        action = [random.uniform(-0.00001, 0.00001), 0.01, random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(0.7, 1)]

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)


    goal_rel_pos_JS = goalJS - gripperPosJS
    #objectJS = env.objectJS
    #goal_rel_pos_JS = goalJS - objectJS

    while has_reached(goal_rel_pos_JS, 0.05) and (timeStep <= env._max_episode_steps - 1) :
        action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(0.7, 1)]

        gripperPosJS = env.lastObservationJS
        goal_rel_pos_JS = goalJS - gripperPosJS

        # objectJS = env.objectJS
        # goal_rel_pos_JS = goalJS - objectJS

        for i in range(len(goal_rel_pos_JS)):
            action[i] = goal_rel_pos_JS[i]*1


        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)


    while True:
        if timeStep >= env._max_episode_steps: break
        action = [random.uniform(-0.00001, 0.00001), -0.003, random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(0.7, 1)]

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

    print("Toatal timesteps taken ", timeStep)
    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)

    

if __name__ == "__main__":
    main()

