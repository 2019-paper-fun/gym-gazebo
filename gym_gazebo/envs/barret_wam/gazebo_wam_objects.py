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
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64
from controller_manager_msgs.srv import SwitchController
from gym.utils import seeding
from numpy import linalg as LA
from iri_common_drivers_msgs.srv import QueryInverseKinematics
from iri_common_drivers_msgs.srv import QueryForwardKinematics
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from iri_common_drivers_msgs.msg import tool_closeAction, tool_closeActionGoal, tool_openAction, tool_openActionGoal
from gz_gripper_plugin.srv import CheckGrasped


"""  WAM ENVIRONMENT FOR SINGLE BLOCK PICK AND PLACE LEARNING TASK  """


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class GazeboWAMemptyEnvv2(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "iri_wam_HER.launch")
        self.publishers = ['pub1', 'pub2', 'pub4', 'pub5', 'pub6'] #publishers for the motor commands

        self.publishers[0] = rospy.Publisher('/iri_wam/joint1_position_controller/command', Float64, queue_size=5)
        self.publishers[1] = rospy.Publisher('/iri_wam/joint2_position_controller/command', Float64, queue_size=5)
        self.publishers[2] = rospy.Publisher('/iri_wam/joint4_position_controller/command', Float64, queue_size=5)
        self.publishers[3] = rospy.Publisher('/iri_wam/joint5_position_controller/command', Float64, queue_size=5)
        self.publishers[4] = rospy.Publisher('/iri_wam/joint6_position_controller/command', Float64, queue_size=5)
       
        
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        #self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state",SetModelState)
        self.get_state = rospy.ServiceProxy("/gazebo/get_model_state",GetModelState)
        self.get_gripper = rospy.ServiceProxy('/check_grasped', CheckGrasped)
        self.close_gripper = rospy.ServiceProxy('/gripper/close', CheckGrasped)
        self.open_gripper = rospy.ServiceProxy('/gripper/open', CheckGrasped)


        self.type = 'train' # 'data'
        self.markers = 1
        self.distanceThreshold = 0.059
        self.objectSize = 0.06
        self.fixedObjectSize = 0.09
        self.baseFrame = 'iri_wam_link_base'
        self.homingTime = 0.2 # time given for homing
        self.lenGoal = 6 # goal position list length
        self._max_episode_steps = 40
        self.reward_type = 'sparse'
        self.objectName = {
            "obj1" : 'obs_0', "objFixed" : 'obs_fixed'
        }
              
        if self.markers:
            self.pubMarker = ['marker1', 'marker2', 'marker3']
            self.pubMarker[0] = rospy.Publisher('/goalPose0', Marker, queue_size=5)
            self.pubMarker[1] = rospy.Publisher('/goalPose1', Marker, queue_size=5)
            self.pubMarker[2] = rospy.Publisher('/goalPose2', Marker, queue_size=5)
            
        self.home = np.array([0, 0.6, 1.4, 0, 0]) # what position is the homing
        # self.Xacrohigh = np.array([2.6, 2.0, 2.8, 3.1, 1.24, 1.57, 2.96])
        # self.Xacrolow = np.array([-2.6, -2.0, -2.8, -0.9, -4.76, -1.57, -2.96])
        self.lowConcObs = np.array([-2.6, -1.94, -0.88, -4.76, -1.55]) #1, 2, 4, 6 #check if they lie inside the envelope
        self.highConcObs = np.array([2.6, 1.94, 3.08, 1.24, 1.55])
        self.lowAction = [-1, -1, -1, -1, -1, -1]
        self.highAction = [1, 1, 1, 1, 1, 1]
        self.n_actions = len(self.highAction)
        self.lenObs = 25

        self.lastObservation = None
        self.lastObservationJS = None
        self.lastObservationOrient = None
        self.goal = None
        self.goalJS = None
        self.object = None
        self.objInitial = None
        self.objInitialJS = None

        self.action_space = spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(self.lenGoal,), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(self.lenGoal,), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=(self.lenObs,), dtype='float32'),
        ))

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")


    def getForwardKinematics(self, goalPosition): #get catesian coordinates for joint Positions
        rospy.wait_for_service('/iri_wam/iri_wam_ik/get_wam_fk')
        try:
            getFK = rospy.ServiceProxy('/iri_wam/iri_wam_ik/get_wam_fk', QueryForwardKinematics)
            jointPoseReturned = getFK(goalPosition)
            return jointPoseReturned
        except (rospy.ServiceException) as e:
            print ("Service call failed: %s"%e)

    def getInverseKinematics(self, goalPose): #get joint angles for reaching the goal position
        tempPose = Pose()
        tempPose = goalPose
        goalPoseStamped = PoseStamped()
        goalPoseStamped.header.frame_id = self.baseFrame
        goalPoseStamped.pose = tempPose
        rospy.wait_for_service('/iri_wam/iri_wam_ik/get_wam_ik')
        #rospy.wait_for_service('/iri_wam/iri_wam_tcp_ik/get_wam_ik')
        try:
            getIK = rospy.ServiceProxy('/iri_wam/iri_wam_ik/get_wam_ik', QueryInverseKinematics)
            #getIK = rospy.ServiceProxy('/iri_wam/iri_wam_tcp_ik/get_wam_ik', QueryInverseKinematics)
            jointPositionsReturned = getIK(goalPoseStamped)
            return [jointPositionsReturned.joints.position[0], jointPositionsReturned.joints.position[1], jointPositionsReturned.joints.position[3], jointPositionsReturned.joints.position[4], jointPositionsReturned.joints.position[5]]
        except (rospy.ServiceException) as e:
            print ("Service call failed: %s"%e)


    def getArmPosition(self, joints):
        frame_ID = self.baseFrame
        tempJointState = JointState()
        tempJointState.header.frame_id = self.baseFrame
        tempJointState.position = joints
        tempPoseFK = self.getForwardKinematics(tempJointState)
        return [np.array([tempPoseFK.pose.pose.position.x, tempPoseFK.pose.pose.position.y, tempPoseFK.pose.pose.position.z]), np.array([tempPoseFK.pose.pose.orientation.x, tempPoseFK.pose.pose.orientation.y, tempPoseFK.pose.pose.orientation.z, tempPoseFK.pose.pose.orientation.w ])]


    def sample_goal_onTable(self): #sample from reachable positions
        self.armGoalXtraHeight = 0.02
        if self.objInitial != None:
            sampledGoal = self.objInitial
            xFactor = random.uniform(0.0, 0.2) * np.random.choice([-1, 1], 1)
            sampledGoal.position.x += xFactor
            if np.absolute(xFactor) < 0.15:
                sampledGoal.position.y += random.uniform(0.15, 0.25) * np.random.choice([-1, 1], 1)
            else :
                sampledGoal.position.y += random.uniform(0.05, 0.25) * np.random.choice([-1, 1], 1)
            sampledGoal.position.z += self.armGoalXtraHeight + self.objectSize # goal for arm
            sampledGoal.orientation.x = 0.0
            sampledGoal.orientation.y = 1.0
            sampledGoal.orientation.z = 0.0
            sampledGoal.orientation.w = 0.0

            sampledGoal.position.x = np.asscalar(np.clip(sampledGoal.position.x, 0.4, 0.55))
            sampledGoal.position.y = np.asscalar(np.clip(sampledGoal.position.y, -0.35 , 0.35))

            if self.type == 'data' : self.goalJS = self.getInverseKinematics(sampledGoal)
            return np.array([sampledGoal.position.x, sampledGoal.position.y, sampledGoal.position.z - self.armGoalXtraHeight - self.objectSize, sampledGoal.position.x, sampledGoal.position.y, sampledGoal.position.z - self.armGoalXtraHeight ]) # actual goal
        else:
            sampledGoal.position.x = 0.5001911647282589
            sampledGoal.position.y = 0.1004797189877992
            sampledGoal.position.z = -0.21252162794043228
            sampledGoal.orientation.x = 0.0
            sampledGoal.orientation.y = 1.0
            sampledGoal.orientation.z = 0.0
            sampledGoal.orientation.w = 0.0
            sampledGoal.position.z += 1.075 + self.objectSize + self.armGoalXtraHeight# goal for arm
            if self.type == 'data' : self.goalJS = self.getInverseKinematics(sampledGoal)
            return np.array([sampledGoal.position.x, sampledGoal.position.y, sampledGoal.position.z - self.armGoalXtraHeight - self.objectSize, sampledGoal.position.x, sampledGoal.position.y, sampledGoal.position.z  - self.armGoalXtraHeight])


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        if len(achieved_goal.shape) == 1:
            d = self.relaxed_goal_checking(achieved_goal[:3], desired_goal[:3]) #fixed block
            d2 = self.relaxed_goal_checking(achieved_goal[3:], desired_goal[3:])
            reward = -(np.array(not (d and d2 and self.gripperState==0))).astype(np.float32)
        else:
            reward = -np.ones(achieved_goal.shape[0])
            for x in range(achieved_goal.shape[0]):
                d = self.relaxed_goal_checking(achieved_goal[x][:3], desired_goal[x][:3])
                d2 = self.relaxed_goal_checking(achieved_goal[x][3:], desired_goal[x][3:])
                reward[x] = -(np.array(not (d and d2 and self.gripperState==0))).astype(np.float32)
        return reward

    def _is_success(self, achieved_goal, desired_goal):
        d = self.relaxed_goal_checking(achieved_goal[:3], desired_goal[:3]) #fixed block
        d2 = self.relaxed_goal_checking(achieved_goal[3:], desired_goal[3:])
        return float(d and d2 and self.gripperState==0)

    def relaxed_goal_checking(self, goal_a, goal_b):
        x_distance = np.absolute(goal_a[0] - goal_b[0])
        y_distance = np.absolute(goal_a[1] - goal_b[1])
        z_distance = np.absolute(goal_a[2] - goal_b[2])
        return ((x_distance < self.fixedObjectSize) and (y_distance < self.fixedObjectSize) and (z_distance < self.distanceThreshold))


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high) #clip action to get inside action space range
        self._set_action(action) 
        obs = self._get_obs()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),
        }
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        done = bool(info['is_success'])

        if self.markers:
            self.setMarkers(self.relaxed_goal_checking(obs['achieved_goal'][3:], obs['desired_goal'][3:]), self.goal[3:].copy(), 0) #moving block goal
            self.setMarkers(self.relaxed_goal_checking(obs['achieved_goal'][:3], obs['desired_goal'][:3] ), self.goal[:3].copy(), 2) #marker for the goals #fixed bloack goal
        return obs, reward, done, info

    

    def _set_action(self, action):
        action = action.copy()  # ensure that we don't change the action outside of this scope
        #action *= 0.5
        lastObs = self.lastObservationJS.copy() #get the last position of the arm joints
        for num, joint in enumerate(action[:self.n_actions-1]):
            self.publishers[num].publish(lastObs[num] + joint) # append the action to the last position

        if (action[self.n_actions-1]) > 0.1: #close
            self.close_gripper_func()
            #print("CLSOE")
        if (action[self.n_actions-1]) < -0.1: #open
            self.open_gripper_func()
            #print("OPEn")
        else: None
            #print("NONE")

            
    def _get_obs(self):
        data = None
        objectPos, objectOrientation = np.zeros(3), np.zeros(4)
        fixedObjectPos, fixedObjectOrientation = np.zeros(3), np.zeros(4)
        gripperPos = self.lastObservation
        gripperOrient = self.lastObservationOrient
        self.gripperState = None
        obs = []
        
        while data is None:
            try:
                data = rospy.wait_for_message('/joint_states', JointState, timeout=1)
                [fixedObjectPos, fixedObjectOrientation] = self.get_object_pose(self.objectName['objFixed'])
                [objectPos, objectOrientation] = self.get_object_pose(self.objectName['obj1'])
                if self.markers:
                    self.setMarkers( 1.0, objectPos, 1) #MARKER for current object position, free movable object

                [gripperPos, gripperOrient] = self.getArmPosition(data.position) #cartesian coordinates of the gripper
                dataConc = np.array([data.position[0], data.position[1], data.position[3], data.position[4], data.position[5]]) # joint space coordinates of the robotic arm 1, 2, 4, 6
                self.gripperState = self.get_gripper_state()
                if ((np.array(dataConc)<=self.highConcObs).all()) and ((np.array(dataConc)>=self.lowConcObs).all()): #check if they lie inside allowed envelope
                    self.lastObservation = gripperPos.copy()
                    self.lastObservationJS = dataConc.copy()
                    self.lastObservationOrient = gripperOrient.copy()
                else:
                    data = None
                    print ("Bad observation data received STEP" )
                    for joint in range(len(self.publishers)):
                        self.publishers[joint].publish(self.home[joint]) #homing at every reset
                    time.sleep(self.homingTime)
            except:
                pass

        obs = np.append(obs, gripperPos.ravel()) #3
        obs = np.append(obs, gripperOrient.ravel()) #4
        obs = np.append(obs, self.gripperState) #1
        obs = np.append(obs, fixedObjectPos.ravel()) # 3
        obs = np.append(obs, objectPos.ravel()) #3
        obs = np.append(obs, (objectPos - gripperPos).ravel()) #3
        obs = np.append(obs, fixedObjectOrientation.ravel()) #4
        obs = np.append(obs, objectOrientation.ravel())#4
        
        return {
            'observation': obs.copy(),
            'achieved_goal': np.concatenate([fixedObjectPos, objectPos]),
            'desired_goal': self.goal.copy(),
        }
        

    def get_object_pose(self, objectName):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            temp = self.get_state(objectName, None)
            temp.pose.position.z -= 1.075 #adjust according to table height
            #objectPos = np.array([temp.pose.position.x, temp.pose.position.y, temp.pose.position.z])
            if objectName == self.objectName['obj1'] :
                self.object = temp.pose
            return [np.array([temp.pose.position.x, temp.pose.position.y, temp.pose.position.z]), np.array([temp.pose.orientation.x, temp.pose.orientation.y, temp.pose.orientation.z, temp.pose.orientation.w ])]
        except (rospy.ServiceException) as e:
            print ("/gazebo/get_model_state service call failed")


    def get_gripper_state(self):
        rospy.wait_for_service('/check_grasped')
        try:
            temp = self.get_gripper()
            if temp.grasped == '':
                return 0
            else: return 1 

        except (rospy.ServiceException) as e:
            print ("/check_grasped service call failed")

    def close_gripper_func(self):
        rospy.wait_for_service('/gripper/close')
        try:
            temp = self.close_gripper()
        except (rospy.ServiceException) as e:
            print ("/close gripper service call failed")

    def open_gripper_func(self):
        rospy.wait_for_service('/gripper/open')
        try:
            temp = self.open_gripper()
        except (rospy.ServiceException) as e:
            print ("/open gripper service call failed")

    def reset(self):
        # rospy.wait_for_service('/gazebo/reset_world') # Reset simulation was causing problems, do not reset simulation
        # try:
        #     #reset_proxy.call()
        #     self.reset_proxy()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/reset_world service call failed")


        # rospy.wait_for_service('/iri_wam/controller_manager/list_controllers')
        # try:
        #     readController = rospy.ServiceProxy('/iri_wam/controller_manager/list_controllers', Empty)
        #     control = readController()
        #     #print (control)
        #     #onHai = bool(controllers.iri_wam_controller.state == "stopped")
        # except (rospy.ServiceException) as e:
        #     print ("Service call failed: %s"%e)

        #if not onHai:
        # rospy.wait_for_service('/iri_wam/controller_manager/switch_controller')
        # try:
        #     change_controller = rospy.ServiceProxy('/iri_wam/controller_manager/switch_controller', SwitchController)
        #     ret = change_controller(['joint1_position_controller', 'joint2_position_controller', 'joint3_position_controller', 'joint4_position_controller', 'joint5_position_controller', 'joint6_position_controller', 'joint7_position_controller'], ['iri_wam_controller'], 2)
        # except (rospy.ServiceException) as e:
        #     print ("Service call failed: %s"%e)

        
        for joint in range(len(self.publishers)):
            self.publishers[joint].publish(self.home[joint]) #homing at every reset
        self.open_gripper_func()
        time.sleep(self.homingTime)

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            pose = Pose()
            state = ModelState()
            state.model_name = self.objectName['obj1']
            pose.position.x = 0.5001911647282589
            pose.position.y = 0.1004797189877992
            pose.position.z = 0.9
            pose.orientation.x = 0.00470048637345294
            pose.orientation.y = 0.99998892605584
            pose.orientation.z = 9.419015715062839e-06
            pose.orientation.w = -0.00023044483691539005
            state.pose = pose
            ret = self.set_state(state)
        except (rospy.ServiceException) as e:
            print ("/gazebo/set model pose service call failed")


        _, _ = self.get_object_pose(self.objectName['obj1'])
        self.objInitial = self.object
        self.objInitialJS = self.getInverseKinematics(self.objInitial) # reference for sampling the goal
        self.goal = self.sample_goal_onTable().copy() # get a random goal every time you reset
        obs = self._get_obs()

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            pose = Pose()
            state = ModelState()
            state.model_name = self.objectName['objFixed']
            pose.position.x = self.goal[0]
            pose.position.y = self.goal[1]
            pose.position.z = 0.9
            pose.orientation.x = 0.00470048637345294
            pose.orientation.y = 0.99998892605584
            pose.orientation.z = 9.419015715062839e-06
            pose.orientation.w = -0.00023044483691539005
            state.pose = pose
            ret = self.set_state(state)
        except (rospy.ServiceException) as e:
            print ("/gazebo/set model pose service call failed")

        return obs


    def setMarkers(self, difference, point, objID):
        pointToPose = Point()
        pointToPose.x = point[0]
        pointToPose.y = point[1]
        pointToPose.z = point[2]
        markerObj = Marker()
        markerObj.header.frame_id = self.baseFrame
        markerObj.id = objID
        markerObj.ns = 'iri_wam'
        markerObj.type = markerObj.SPHERE
        markerObj.action = markerObj.ADD
        markerObj.pose.position = pointToPose
        markerObj.pose.orientation.w = 1.0
        
        if objID == 0: #moving block
            markerObj.scale.x = self.fixedObjectSize
            markerObj.scale.y = self.fixedObjectSize
            markerObj.scale.z = self.distanceThreshold
            if difference and self.gripperState==0:
                markerObj.color.g = 1.0
                markerObj.color.a = 1.0

            else: 
                markerObj.color.r = 1.0
                markerObj.color.a = 1.0
        elif objID == 2: #fixed block
            markerObj.scale.x = self.fixedObjectSize
            markerObj.scale.y = self.fixedObjectSize
            markerObj.scale.z = self.distanceThreshold
            if difference :
                markerObj.color.g = 1.0
                markerObj.color.a = 1.0

            else: 
                markerObj.color.r = 1.0
                markerObj.color.a = 1.0
        else:
            markerObj.scale.x = self.distanceThreshold
            markerObj.scale.y = self.distanceThreshold
            markerObj.scale.z = self.distanceThreshold
            if ( difference <= (self.distanceThreshold)):
                markerObj.color.g = 1.0
                markerObj.color.a = 1.0

            else: 
                markerObj.color.r = 1.0
                markerObj.color.a = 1.0            

        self.pubMarker[objID].publish(markerObj)



