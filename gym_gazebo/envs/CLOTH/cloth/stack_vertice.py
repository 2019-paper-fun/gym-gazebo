from gym import utils
from gym_gazebo.envs.CLOTH import cloth_on_table

import gym
import gym_gazebo

import numpy as np

class ClothMeshEnv(cloth_on_table.ClothOnTableEnv):
	# called once graph is created, to init some stuff...
	def __init__(self, node):
		print 'Stack vertice.py initGraph called (python side)'
		cloth_on_table.ClothOnTableEnv.__init__(
            self, node = node)