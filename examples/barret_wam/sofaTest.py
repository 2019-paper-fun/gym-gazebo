import gym
import gym_gazebo
import time
import random
import numpy as np
import rospy
import roslaunch

def main():
    env = gym.make('SofaEnv-v1')
    print("Reset!")
    env.reset()

if __name__ == "__main__":
    main()