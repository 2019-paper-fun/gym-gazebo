import gym
import rospy
import roslaunch
import time
import numpy as np
import random

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from controller_manager_msgs.srv import SwitchController
from gym.utils import seeding
from iri_common_drivers_msgs.srv import QueryInverseKinematics
from iri_common_drivers_msgs.srv import QueryForwardKinematics


class GazeboWAMemptyEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "iri_wam.launch")
        #self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.publishers = ['pub1', 'pub2', 'pub3', 'pub4', 'pub5', 'pub6', 'pub7']

        self.publishers[0] = rospy.Publisher('/iri_wam/joint1_position_controller/command', Float64, queue_size=5)
        self.publishers[1] = rospy.Publisher('/iri_wam/joint2_position_controller/command', Float64, queue_size=5)
        self.publishers[2] = rospy.Publisher('/iri_wam/joint3_position_controller/command', Float64, queue_size=5)
        self.publishers[3] = rospy.Publisher('/iri_wam/joint4_position_controller/command', Float64, queue_size=5)
        self.publishers[4] = rospy.Publisher('/iri_wam/joint5_position_controller/command', Float64, queue_size=5)
        self.publishers[5] = rospy.Publisher('/iri_wam/joint6_position_controller/command', Float64, queue_size=5)
        self.publishers[6] = rospy.Publisher('/iri_wam/joint7_position_controller/command', Float64, queue_size=5) # discretely publishing motor actions for now
        
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.minDisplacement = 0.09
        self.baseFrame = 'iri_wam_link_base'

        self.home = np.zeros(7)
        self.high = np.array([2.5, 1.9, 2.7, 3.0, 1.14, 1.47, 3])
        self.low = np.array([-2.5, -1.9, -2.7, -0.8, -4.66, -1.47, -3])

        self.envelopeAugmented = self.high - self.low 


        self.action_space = spaces.MultiBinary(7)
        self.observation_space = spaces.Box(self.low, self.high)
        self.reward_range = (-np.inf, np.inf)

        self._seed()
        #self.lastState = None


    def getRandomGoal(self):  #sample from reachable positions
        frame_ID = self.baseFrame
        tempPosition = []
        for joint in range(7):
            tempPosition.append(random.uniform(self.low[joint], self.high[joint]))
            #print ("GOAL position sampled for joint : ", joint, " is ", tempPosition[joint])

        tempJointState = JointState()
        tempJointState.header.frame_id = self.baseFrame
        tempJointState.position = tempPosition
        #print (self.getForwardKinematics(tempJointState))

        #tempPose = PoseStamped(frame_id = frame_ID , position, orientation)
        #self.desiredGoal = tempPosition
        return tempPosition

        
    # def getInverseKinematics(self, goalPose): #get joint angles for reaching the goal position
    #     goalPoseStamped = goalPose
    #     rospy.wait_for_service('/iri_wam/iri_wam_ik/get_wam_ik')
    #     try:
    #         getIK = rospy.ServiceProxy('/iri_wam/iri_wam_ik/get_wam_ik', QueryInverseKinematics)
    #         jointPositionsReturned = getIK(goalPoseStamped)
    #         return jointPositionsReturned.position
    #     except (rospy.ServiceException) as e:
    #         print ("Service call failed: %s"%e)

    def getForwardKinematics(self, goalPosition): #get catesian coordinates for joint Positions

        rospy.wait_for_service('/iri_wam/iri_wam_ik/get_wam_fk')
        try:
            getFK = rospy.ServiceProxy('/iri_wam/iri_wam_ik/get_wam_fk', QueryForwardKinematics)
            jointPoseReturned = getFK(goalPosition)
            return jointPoseReturned
        except (rospy.ServiceException) as e:
            print ("Service call failed: %s"%e)


    def goToGoal(self, goalPosition, lastObs):
        
        for joint in range(7):
            difference = lastObs - goalPosition
            start_time = time.time()
            while np.absolute(difference[joint]) > self.minDisplacement:
                #print ("Difference in goal for joint : ", joint, " = ", difference[joint] , " and current is : ", lastObs[joint], " and desired ", goalPosition[joint] )
                if difference[joint] > 0: #take negative action, backward
                    action = [joint, 1]  
                elif difference[joint] < 0: #take positive action, forward
                    action = [joint, 0]
                else: #take no action
                    None
                obsData, reward, done, info = self.step(action, lastObs)
                lastObs = obsData
                difference = lastObs - goalPosition
                elapsed_time = time.time() - start_time
                if elapsed_time > 30: 
                    print("exiting from while loop")
                    break

            if np.absolute(difference[joint]) <= self.minDisplacement:
                print ("Goal position reached for joint number ", joint)
            else:
                print("exiting due to bad goal point")
                break

    



    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action, lastObservation):


        lastObs = np.array(lastObservation)
        lastObsForward = (lastObs + self.minDisplacement)
        lastObsBackward = (lastObs - self.minDisplacement)


        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        jointNumber = action[0]
        jointAction = action[1]


        if jointAction == 0: self.publishers[jointNumber].publish(lastObsForward[jointNumber])
        elif jointAction == 1: self.publishers[jointNumber].publish(lastObsBackward[jointNumber])

        

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/joint_states', JointState, timeout=10)
                if (np.array(data.position)<=self.high).all() and (np.array(data.position)>=self.low).all():
                    state = np.array(data.position)
                    #print ("New Authentic observation data received :")
                else:
                    data = None
                    print ("Bad observation data received " )
                    for joint in range(7):
                        self.publishers[joint].publish(self.home[joint]) #homing at every reset
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")


        done = False

        # if not done:
        #     if action == 0:
        #         reward = 5
        #     else:
        #         reward = 1
        # else:
        #     reward = -200

        reward = 0

        return state, reward, done, {}


    def _reset(self):

        # Resets the state of the environment and returns an initial observation.

        # rospy.wait_for_service('/gazebo/reset_simulation')
        # try:
        #     #reset_proxy.call()
        #     self.reset_proxy()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")


        rospy.wait_for_service('/iri_wam/controller_manager/switch_controller')
        try:
            change_controller = rospy.ServiceProxy('/iri_wam/controller_manager/switch_controller', SwitchController)
            ret = change_controller(['joint1_position_controller', 'joint2_position_controller', 'joint3_position_controller', 'joint4_position_controller', 'joint5_position_controller', 'joint6_position_controller', 'joint7_position_controller'], ['iri_wam_controller'], 2)
        except (rospy.ServiceException) as e:
            print ("Service call failed: %s"%e)


        for joint in range(7):
            self.publishers[joint].publish(self.home[joint]) #homing at every reset

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/joint_states', JointState, timeout=10)
                if (np.array(data.position)<=self.high).all() and (np.array(data.position)>=self.low).all():
                    state = np.array(data.position)
                    #print ("New Authentic observation data received :")
                else:
                    data = None
                    print ("Bad observation data received " )
                    for joint in range(7):
                        self.publishers[joint].publish(self.home[joint]) #homing at every reset
            except:
                pass
                

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

         #send the observed state to the robot

        return state
