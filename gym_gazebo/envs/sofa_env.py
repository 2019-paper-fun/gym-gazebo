import Sofa

import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding



class SofaEnv(gym.GoalEnv):

	def __init__(self):
		print ("Sofa Gym Environemnt launched!")
		

	def step(self, action):
		raise NotImplementedError

	def _reset(self):
		raise NotImplementedError

	def _get_obs(self):
		raise NotImplementedError

	def _env_setup(self):
		pass

	def _set_action(self, action):
		raise NotImplementedError

	# def reset_mutated(self, demo_state):
 #        raise NotImplementedError

 #    def render(self):

 #        #render the GUI environment
 #        raise NotImplementedError

 #    def close(self):

 #        # Kill sofa environment
 #        pass

 #    def configure(self):

 #        # TODO
 #        # From OpenAI API: Provides runtime configuration to the enviroment
 #        # Maybe set the Real Time Factor?
 #        pass


 #    def _get_obs(self):
 #        """Returns the observation.
 #        """
 #        raise NotImplementedError()

 #    def _set_action(self, action):
 #        """Applies the given action to the simulation.
 #        """
 #        raise NotImplementedError()

 #    def _is_success(self, achieved_goal, desired_goal, obs):
 #        """Indicates whether or not the achieved goal successfully achieved the desired goal.
 #        """
 #        raise NotImplementedError()

 #    def _sample_goal(self):
 #        """Samples a new goal and returns it.
 #        """
 #        raise NotImplementedError()

 #    def _env_setup(self):
 #        """Initial configuration of the environment. Can be used to configure initial state
 #        and extract information from the simulation.
 #        """
 #        pass

 #    def _viewer_setup(self):
 #        """Initial configuration of the viewer. Can be used to set the camera position,
 #        for example.
 #        """
 #        pass

 #    def _render_callback(self):
 #        """A custom callback that is called before rendering. Can be used
 #        to implement custom visualizations.
 #        """
 #        pass

 #    def _step_callback(self):
 #        """A custom callback that is called after stepping the simulation. Can be used
 #        to enforce additional constraints on the simulation state.
 #        """
 #        pass
    
 #    def seed(self):
 #        # TODO
 #        # From OpenAI API: Sets the seed for this env's random number generator(s)
 #        pass