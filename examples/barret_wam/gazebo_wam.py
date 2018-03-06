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
from iri_common_drivers_msgs.srv import QueryInverseKinematics
from iri_common_drivers_msgs.srv import QueryForwardKinematics


ep_returns = []
actions = []
observations = []
rewards = []

def main():
    env = gym.make('GazeboWAMemptyEnv-v0')
    #env.seed()
    

    env.reset()
    print("Reset!")
    time.sleep(10)
    while len(actions) < 100:
        obs = env.reset()
        print("Reset!")
        #randomGoalPosition = env.getRandomGoal()
        #print("New Goal received!")
        goToGoal(env, obs)
        # action = randint(0, 13)
        # obsData, reward, done, info = env.step(action)
        #time.sleep(5)
        # for x in range(1000):
        #     # env.render()
        #     #action = [randint(0, 6), randint(0, 1)]
        #     #print("action = ", action)
        #     #observation, reward, done, info = env.step(action) # take a random action
        #     #time.sleep(2)
        #     #print("reward: ", reward, " observation: ", observation)

        #     randomGoalPosition = env.getRandomGoal()
        #     env.goToGoal(randomGoalPosition)
        #     if done: break

    np.savez_compressed('data_wam.npz', acs=actions, obs=observations, ep_rets=ep_returns, rews=rewards)
    #np.save('obs.npy', observations)
    #np.save('ep_rets.npy', ep_returns)
    #np.save('rews.npy', rewards)




def getInverseKinematics(env, goalPose): #get joint angles for reaching the goal position
    tempPose = Pose()
    #tempPose.header.frame_id = env.baseFrame
    tempPose.position.x = goalPose[0]
    tempPose.position.y = goalPose[1]
    tempPose.position.z = goalPose[2]
    tempPose.orientation.x = goalPose[3]
    tempPose.orientation.y = goalPose[4]
    tempPose.orientation.z = goalPose[5]
    tempPose.orientation.w = goalPose[6]

    #print (tempPose)
    goalPoseStamped = PoseStamped()
    goalPoseStamped.header.frame_id = env.baseFrame
    goalPoseStamped.pose = tempPose
    rospy.wait_for_service('/iri_wam/iri_wam_ik/get_wam_ik')
    try:
        getIK = rospy.ServiceProxy('/iri_wam/iri_wam_ik/get_wam_ik', QueryInverseKinematics)
        jointPositionsReturned = getIK(goalPoseStamped)
        #print ("Returned by IK " , jointPositionsReturned)
        return jointPositionsReturned.joints.position
    except (rospy.ServiceException) as e:
        print ("Service call failed: %s"%e)



def actionMapping(env, ac):
    trueAction = ac[0] + ac[1]*7

    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    action[ac[0]] = env.minDisplacement*ac[1]
    # if   Taction == 0: trueAction = np.array([0, 0, 0, 0])
    # elif Taction == 1: trueAction = np.array([0, 0, 0, 1])
    # elif Taction == 2: trueAction = np.array([0, 0, 1, 0])
    # elif Taction == 3: trueAction = np.array([0, 0, 1, 1])
    # elif Taction == 4: trueAction = np.array([0, 1, 0, 0])
    # elif Taction == 5: trueAction = np.array([0, 1, 0, 1])
    # elif Taction == 6: trueAction = np.array([0, 1, 1, 0])
    # elif Taction == 7: trueAction = np.array([0, 1, 1, 1])
    # elif Taction == 8: trueAction = np.array([1, 0, 0, 0])
    # elif Taction == 9: trueAction = np.array([1, 0, 0, 1])
    # elif Taction == 10: trueAction = np.array([1, 0, 1, 0])
    # elif Taction == 11: trueAction = np.array([1, 0, 1, 1])
    # elif Taction == 12: trueAction = np.array([1, 1, 0, 0])
    # elif Taction == 13: trueAction = np.array([1, 1, 0, 1])

    return action


def goToGoal(env, lastObs):
    done = False
    goalPosition = getInverseKinematics(env, lastObs[np.shape(env.high)[0]:])

    
    reached = -1
    ep_return = 0
    episodeAcs = []
    episodeObs = []
    episodeRews = []
    #badDataFlag = False

    while done == False:
        if goalPosition == None: 
            print ("Inverse kinematics failed ")
            break
        badDataFlag = False
        for joint in range(7):
            episodeAcsTemp = np.zeros(env.action_space.shape[0])
            episodeRewsTemp = 0
            obsData = env.lastObservation
            difference = lastObs[:np.shape(env.high)[0]] - goalPosition
            start_time = time.time()
            while (np.absolute(difference[joint]) > env.minDisplacement) and done == False:
                #print ("Difference in goal for joint : ", joint, " = ", difference[joint] , " and current is : ", lastObs[joint], " and desired ", goalPosition[joint] )
                if difference[joint] > 0: #take negative action, backward
                    action = actionMapping(env, [joint, -1])  
                elif difference[joint] < 0: #take positive action, forward
                    action = actionMapping(env, [joint, 1])
                else: #take no action
                    None
                obsData, reward, done, badDataFlag, moved = env.step(action)
                if badDataFlag: 
                    print ("starting again due to bad data ")
                    break
                if moved:#(np.absolute(reward) > 0.001) and moved:
                    #print ("Reward received :", reward)
                    episodeAcsTemp += action
                    episodeRewsTemp += reward
                    ep_return += reward
                lastObs = obsData[:np.shape(env.high)[0]]
                difference = lastObs - goalPosition
                elapsed_time = time.time() - start_time
                if elapsed_time > env.waitTime: 
                    print("exiting from while loop")
                    break
            

            elapsed_time = time.time() - start_time
            if badDataFlag: 
                print ("starting again due to bad data 2 ")
                break

            if np.absolute(difference[joint]) < env.minDisplacement:
                #print ("Difference particular joint  ", np.absolute(difference[joint]), env.minDisplacementCheck, np.absolute(difference[joint]) < env.minDisplacementCheck )
                reached = joint
                print ("Goal position reached for joint number ", joint)
                episodeObs.append(obsData)
                episodeRews.append(episodeRewsTemp)
                episodeAcs.append(episodeAcsTemp)
                print ("Action values appended for joint  ", joint, episodeAcsTemp)
                print ("Reward values appended for joint  ", joint, episodeRewsTemp)
                print ("observation values appended for joint  ", joint, obsData)

            elif reached < 0 or elapsed_time > (env.waitTime*2) :
                print("Bad point, moving out of for loop")
                break
            else: 
                print("Moving to next joint ")

            if done==True :#and reached==6:
                ep_returns.append(ep_return)
                actions.append(np.array(episodeAcs))
                observations.append(np.array(episodeObs))
                rewards.append(np.array(episodeRews))
                print ("total episode return: ", ep_return)
                print ("total episode actions lenght: ", len(episodeAcs))
                print ("total episode observation length : ", len(episodeObs))
                break
            else: reached = 100

        elapsed_time = time.time() - start_time
        if elapsed_time > env.waitTime: 
            print("exiting from the main while loop")
            break

        if badDataFlag: 
                print ("starting again due to bad data 3 ")
                break


    # while done == False:
    #     for joint in range(7):
    #         difference = lastObs[:np.shape(env.high)[0]] - goalPosition
    #         start_time = time.time()
    #         while (np.absolute(difference[joint]) > env.minDisplacement) and done == False:
    #             #print ("Difference in goal for joint : ", joint, " = ", difference[joint] , " and current is : ", lastObs[joint], " and desired ", goalPosition[joint] )
    #             if difference[joint] > 0: #take negative action, backward
    #                 action = actionMapping([joint, 1])  
    #             elif difference[joint] < 0: #take positive action, forward
    #                 action = actionMapping([joint, 0])
    #             else: #take no action
    #                 None
    #             obsData, reward, done, info = env.step(action)
    #             if np.absolute(reward) > 0.001:
    #                 #print ("Reward received :", reward)
    #                 episodeAcs.append(action)
    #                 episodeObs.append(obsData)
    #                 episodeRews.append(reward)
    #                 ep_return += reward
    #             lastObs = obsData[:np.shape(env.high)[0]]
    #             difference = lastObs - goalPosition
    #             elapsed_time = time.time() - start_time
    #             if elapsed_time > env.waitTime: 
    #                 print("exiting from while loop")
    #                 break

    #         if np.absolute(difference[joint]) < env.minDisplacementCheck:
    #             #print ("Difference particular joint  ", np.absolute(difference[joint]), env.minDisplacementCheck, np.absolute(difference[joint]) < env.minDisplacementCheck )
    #             reached = joint
    #             print ("Goal position reached for joint number ", joint)
    #         else:
    #             print("exiting due to bad goal point")
    #             break

    #         if done==True :#and reached==6:
    #             ep_returns.append(ep_return)
    #             actions.append(np.array(episodeAcs))
    #             observations.append(np.array(episodeObs))
    #             rewards.append(np.array(episodeRews))
    #             print ("total episode return: ", ep_return)
    #             print ("total episode actions lenght: ", len(episodeAcs))
    #             print ("total episode observation length : ", len(episodeObs))
    #             break

    #     if elapsed_time > env.waitTime: 
    #         print("exiting from while loop")
    #         break
        


if __name__ == "__main__":
    main()
