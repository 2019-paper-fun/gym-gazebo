import Sofa
import sys
import gym
import time
import numpy as np
import random
from random import randint

from gym import utils, spaces
from gym_gazebo.envs import sofa_env

from gym.utils import seeding
from numpy import linalg as LA


def goal_distance(goal_a, goal_b): #check how far is the current state from goal
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class ClothOnTableEnv(sofa_env.SofaEnv):

    def __init__(self, node):
        print 'CLOTH ON TABLE ENVIRONMENT'
        super(ClothOnTableEnv, self).__init__()
        self.distance_threshold = 10.0

        self.node = node
        self.node.animate = True
        self.actuator = self.node.getChild('Input').getObject('DOFs')
        self.goal = self._sample_goal()
        
        self.dt = self.node.findData('dt').value
        self.metadata = {
            'semantics.autoreset': False,
        }
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self.can_reset_sim = False
        return 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_reward(self, achieved_goal, desired_goal, info): # same function used for HER but defined there as well due to py2 and 3 incompatibility, any changes here must be duplicated there
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        return -(np.array(d > self.distance_threshold)).astype(np.float32)

    def find_indice(self, gripperPosition):
        clothMesh = np.asarray(self.node.getChild('SquareGravity').getObject('cloth').position)
        position = np.array([gripperPosition[0][0], gripperPosition[0][1], gripperPosition[0][2]])
        
        deltas = clothMesh - position
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        closest = np.argmin(dist_2)
        return closest, dist_2[closest]

    def _set_action(self, action):
        
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        #print ("aCTION RECEIVED ", action)

        pos_ctrl, gripper_ctrl = action[:3], action[3]

        #pos_ctrl *= 0.5  # limit maximum change in position
        free_position = self.actuator.free_position # read the position of the gripper at every step
        
        # Apply action to simulation.
        for x in range(len(pos_ctrl)):
            free_position[0][x] += action[x]*4
        self.actuator.free_position = free_position

        ac = self.node.getObject('attachConstraint')
        closest, dist = self.find_indice(self.actuator.free_position) # can also get the order of the closest indices
        if ac.constraintFactor[0][0] == 0.0: ac.indices2 = closest # only change if not already connected
        if (gripper_ctrl >= 0.9 and gripper_ctrl <= 1.0) and dist <= 100.0:
            ac.constraintFactor = 1
        elif (gripper_ctrl >= -1.0 and gripper_ctrl <= -0.9):
            ac.constraintFactor = 0

        return 0

    def step(self, action):
        
        action = np.clip(action, self.action_space.low, self.action_space.high) #clip action to get inside action space range
        self._set_action(action) 
        obs = self._get_obs()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),
        }
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        done = bool(info['is_success'])
        #print("REWARD and INFO", reward, info['is_success'])
        self.can_reset_sim = True
        
        return obs, reward, done, info

    def _get_obs(self):
        #This will be a multiple goal environment
        ballMesh = self.node.getChild('ball')
        clothMesh = self.node.getChild('SquareGravity')

        self.numVertices = 4

        gripperPosition = np.array(ballMesh.getObject('ballState').position[0][0:3])
        gripperVelocity = np.array(ballMesh.getObject('ballState').velocity[0][0:3])

        ac = self.node.getObject('attachConstraint')
        #vertiPos = self.clothMesh.getObject('cloth').position[0:4]
        #vertiVel = self.clothMesh.getObject('cloth').velocity[0:4]

        #dt = #get the change in time dt value here

        vertice_name = ['vertice0', 'vertice1', 'vertice2', 'vertice3']
        verticePositions, verticeVelocities, vertice_rel_positions, vertice_rel_velocities = [], [], [], []

        for verticeNum in range(self.numVertices):
            verticePositions.append(np.array(clothMesh.getObject('cloth').position[verticeNum]))
            verticeVelocities.append(np.array(clothMesh.getObject('cloth').velocity[verticeNum]))
            vertice_rel_positions.append(np.array(verticePositions[verticeNum]- gripperPosition))
            vertice_rel_velocities.append(np.array(verticeVelocities[verticeNum]- gripperVelocity ))
        

        achieved_goal = verticePositions[3].copy()
        # for x in range(self.numVertices):
        #     if x == 0: continue #done above
        #     achieved_goal = np.squeeze(np.concatenate([achieved_goal, verticePositions[x]]))

        #print 'VERTICE position',  verticePositions[0], verticePositions[1], verticePositions[2], verticePositions[3]
        #print 'gripper positiioons asfafsag X', gripperPosition
        gripperState = ac.constraintFactor[0]
        obs = np.concatenate([
                gripperPosition , gripperVelocity, gripperState,
                ])
        #obs = np.append(obs, gripperState, axis=0)
        # for x in range(self.numVertices):
        #     obs = np.append(obs, verticePositions[x].ravel(), axis=0) #3
        #     obs = np.append(obs, verticeVelocities[x].ravel(), axis=0) #3
            #obs = np.append(obs, vertice_rel_positions[x].ravel(), axis=0) #3
            #obs = np.append(obs, vertice_rel_velocities[x].ravel(), axis=0) #3
        
        obs = np.append(obs, verticePositions[3].ravel(), axis=0) #3
        obs = np.append(obs, verticeVelocities[3].ravel(), axis=0)
        obs = np.append(obs, verticePositions[1].ravel(), axis=0) #3
        obs = np.append(obs, verticeVelocities[1].ravel(), axis=0)
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(), #self.goal.copy(),
        }

    def _reset_sim(self):
        
        #self.node.reset() #start the simulation at the initial state
        self.node.reset()
        # Randomize start position of cloth vertices and position of other vertices - future prospects
        #SAMPLE A NEW GOAL
        #print "reseting aging", self.node
        self.goal = self._sample_goal()
        return True

    def reset(self):
        
        print ("resetting ")
        self.goal = self._sample_goal().copy()
        if self.can_reset_sim:
           self._reset_sim()
        obs = self._get_obs()

        return obs

    def _sample_goal(self):
        
        # Select one of the other three vertices to go to, at random
        #RANDOMIZATION
        #vertice 0 being the starting position

        clothMesh = self.node.getChild('SquareGravity')

        goalObject = self.node.getChild('goal').getObject('goalState')

        #x = random.randint(0, 2)
        x = 1
        vertex = np.array(clothMesh.getObject('cloth').position[x])
        randomness = randint(-50, 20)
        vertex[0] += randomness
        #vertex[1] += randint(15, 20)
        vertex[2] += randomness
        goal = vertex

        
        #pos[0][0] += 10.0
        #goalObject.Data.setValue(position, goal)
        #goalObject.applyTranslation(goal[0], goal[1], goal[2])
        goalObject.position = [[goal[0], goal[1], goal[2], 0.0, 0.0, 0.0, 1.0]]

        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        
        d = goal_distance(achieved_goal, desired_goal) #both vertices at there desired locations
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self):
        # initializations for the environemnt
        return None

	