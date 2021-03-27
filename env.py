import vrep, math, sys, time, numpy as np, cv2
import tensorflow as tf
import gc
import gym
import json
import os
from PIL import Image
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import pyglet
from pyglet import gl

SCALE = 6.0
TRACK_RAD = 225/SCALE
PLAYFIELD   = 750/SCALE
FPS         = 50

TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40/SCALE
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]

class CarRacingWrapper(CarRacing):
    
    def __init__(self, full_episode=False, repeated_actions=3):
        super(CarRacingWrapper, self).__init__()
        self.repeated_actions = repeated_actions
        self.full_episode = full_episode
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 3)) # , dtype=np.uint8

    def _process_frame(self, frame):
        obs = frame[0:84, :, :]
        obs = Image.fromarray(obs, mode='RGB').resize((64, 64))
        obs = np.array(obs)
        return obs

    def step(self, action):
        sum_rew = 0
        for _ in range(self.repeated_actions):
            obs, rew, done, info = self._step(action)
            sum_rew += rew
            if done: break
        return obs, sum_rew, done, info

    def _step(self, action):
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        self.state = self._render("state_pixels")

        step_reward = 0
        done = False
        if action is not None: # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            #self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count==len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -150

        return self._process_frame(self.state), step_reward, done, {}

    #def _step(self, action):
    #    obs, reward, done, _ = super(CarRacingWrapper, self)._step(action)
    #    if self.full_episode:
    #        return self._process_frame(obs), reward, False, {}
    #    return self._process_frame(obs), reward, done, {}
    def close(self):
        super(CarRacingWrapper, self).close()
        tf.keras.backend.clear_session()
        gc.collect()

    def _create_track(self):
        #SCALE = 3.0
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2*math.pi*c/CHECKPOINTS + self.np_random.uniform(0, 2*math.pi*1/CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD/3, TRACK_RAD)
            if c==0:
                alpha = 0
                rad = 1.5*TRACK_RAD
            if c==CHECKPOINTS-1:
                alpha = 2*math.pi*c/CHECKPOINTS
                self.start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
                rad = 1.5*TRACK_RAD
            checkpoints.append( (alpha, rad*math.cos(alpha), rad*math.sin(alpha)) )

        #print "\n".join(str(h) for h in checkpoints)
        #self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5*TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while 1:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2*math.pi
            while True: # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0: break
                if not failed: break
                alpha -= 2*math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
            while beta - alpha >  1.5*math.pi: beta -= 2*math.pi
            while beta - alpha < -1.5*math.pi: beta += 2*math.pi
            prev_beta = beta
            proj *= SCALE
            if proj >  0.3: beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
            if proj < -0.3: beta += min(TRACK_TURN_RATE, abs(0.001*proj))
            x += p1x*TRACK_DETAIL_STEP
            y += p1y*TRACK_DETAIL_STEP
            track.append( (alpha,prev_beta*0.5 + beta*0.5,x,y) )
            if laps > 4: break
            no_freeze -= 1
            if no_freeze==0: break
        #print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i==0: return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i-1][0] <= self.start_alpha
            if pass_through_start and i2==-1:
                i2 = i
            elif pass_through_start and i1==-1:
                i1 = i
                break
        print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
        assert i1!=-1
        assert i2!=-1

        track = track[i1:i2-1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square( first_perp_x*(track[0][2] - track[-1][2]) ) +
            np.square( first_perp_y*(track[0][3] - track[-1][3]) ))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False]*len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i-neg-0][1]
                beta2 = track[i-neg-1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE*0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i-neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i-1]
            road1_l = (x1 - TRACK_WIDTH*math.cos(beta1), y1 - TRACK_WIDTH*math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH*math.cos(beta1), y1 + TRACK_WIDTH*math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH*math.cos(beta2), y2 - TRACK_WIDTH*math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH*math.cos(beta2), y2 + TRACK_WIDTH*math.sin(beta2))
            t = self.world.CreateStaticBody( fixtures = fixtureDef(
                shape=polygonShape(vertices=[road1_l, road1_r, road2_r, road2_l])
                ))
            t.userData = t
            c = 0.01*(i%3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(( [road1_l, road1_r, road2_r, road2_l], t.color ))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side* TRACK_WIDTH        *math.cos(beta1), y1 + side* TRACK_WIDTH        *math.sin(beta1))
                b1_r = (x1 + side*(TRACK_WIDTH+BORDER)*math.cos(beta1), y1 + side*(TRACK_WIDTH+BORDER)*math.sin(beta1))
                b2_l = (x2 + side* TRACK_WIDTH        *math.cos(beta2), y2 + side* TRACK_WIDTH        *math.sin(beta2))
                b2_r = (x2 + side*(TRACK_WIDTH+BORDER)*math.cos(beta2), y2 + side*(TRACK_WIDTH+BORDER)*math.sin(beta2))
                self.road_poly.append(( [b1_l, b1_r, b2_r, b2_l], (1,1,1) if i%2==0 else (1,0,0) ))
        self.track = track
        return True

    def _render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD/20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k*x + k, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + k, 0)
                gl.glVertex3f(k*x + k, k*y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        gl.glEnd()

#from vae.vae import CVAE
#from rnn.rnn import MDNRNN, rnn_next_state, rnn_init_state
class CarRacingMDNRNN(CarRacingWrapper):
  def __init__(self, args, load_model=True, full_episode=False, with_obs=False):
    super(CarRacingMDNRNN, self).__init__(full_episode=full_episode)
    self.with_obs = with_obs # whether or not to return the frame with the encodings
    self.vae = CVAE(args)
    self.rnn = MDNRNN(args)
     
    if load_model:
      self.vae.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_vae'.format(args.exp_name, args.env_name)).variables])
      self.rnn.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name)).variables])
    self.rnn_states = rnn_init_state(self.rnn)
    
    self.full_episode = False 
    self.observation_space = Box(low=np.NINF, high=np.Inf, shape=(args.z_size+args.rnn_size*args.state_space))
  def encode_obs(self, obs):
    # convert raw obs to z, mu, logvar
    result = np.copy(obs).astype(np.float)/255.0
    result = result.reshape(1, 64, 64, 3)
    z = self.vae.encode(result)[0]
    return z
  def reset(self):
    self.rnn_states = rnn_init_state(self.rnn)
    if self.with_obs:
        [z_state, obs] = super(CarRacingMDNRNN, self).reset() # calls step
        self.N_tiles = len(self.track)
        return [z_state, obs]
    else:
        z_state = super(CarRacingMDNRNN, self).reset() # calls step
        self.N_tiles = len(self.track)
        return z_state
  def _step(self, action):
    obs, reward, done, _ = super(CarRacingMDNRNN, self)._step(action)
    z = tf.squeeze(self.encode_obs(obs))
    h = tf.squeeze(self.rnn_states[0])
    c = tf.squeeze(self.rnn_states[1])
    if self.rnn.args.state_space == 2:
        z_state = tf.concat([z, c, h], axis=-1)
    else:
        z_state = tf.concat([z, h], axis=-1)
    if action is not None: # don't compute state on reset
        self.rnn_states = rnn_next_state(self.rnn, z, action, self.rnn_states)
    if self.with_obs:
        return [z_state, obs], reward, done, {}
    else:
        return z_state, reward, done, {}
  def close(self):
    super(CarRacingMDNRNN, self).close()
    tf.keras.backend.clear_session()
    gc.collect()

def make_env(args, dream_env=False, seed=-1, render_mode=False, full_episode=False, with_obs=False, load_model=True):
    if args.mdrnn == True:
        print('makeing real CarRacing environment')
        env = CarRacingMDNRNN(args=args, full_episode=full_episode, with_obs=with_obs, load_model=load_model)
    else:
        print('makeing real CarRacing environment')
        #env = gym.make('CarRacing-v0')
        env = CarRacingWrapper(full_episode=False, repeated_actions=3)
    if (seed >= 0):
        env.seed(seed)
    return env

class Env:    
    def __init__(self):
        self.torso = torso
    
    def connectRobot(self):
        clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
        if clientID!=-1:
            print('Connected to remote API server with clientID: ', clientID)
            vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
        else:
            print('Failed connecting to remote API server')
            sys.exit(1)        
        return clientID

    
    def disconnectRobot(self):
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
        vrep.simxFinish(clientID)
        print('Program ended')      

 
    #return joint positions in degree
    def getJointConfig(self):        
        pos1 = vrep.simxGetJointPosition(clientID, shoulderZ, vrep.simx_opmode_blocking)
        pos2 = vrep.simxGetJointPosition(clientID, shoulderY, vrep.simx_opmode_blocking)
        pos3 = vrep.simxGetJointPosition(clientID, armX, vrep.simx_opmode_blocking)
        pos4 = vrep.simxGetJointPosition(clientID, elbowY, vrep.simx_opmode_blocking)
        pos5 = vrep.simxGetJointPosition(clientID, wristZ, vrep.simx_opmode_blocking)
        pos6 = vrep.simxGetJointPosition(clientID, wristX, vrep.simx_opmode_blocking)
        
        positions = []
        positions.append(round(pos1[1]/math.pi * 180,1))
        positions.append(round(pos2[1]/math.pi * 180,1))
        positions.append(round(pos3[1]/math.pi * 180,1))
        positions.append(round(pos4[1]/math.pi * 180,1))
        positions.append(round(pos5[1]/math.pi * 180,1))
        positions.append(round(pos6[1]/math.pi * 180,1))

        return positions

    def setJointConfiguration(self, armX_, shoulderY_, elbowY_, shoulderZ_, wristZ_, wristX_): # (-65,35,-50,30,-50,30)
        [res1,shoulderZ]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_z', vrep.simx_opmode_blocking)
        [res2,shoulderY]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_y', vrep.simx_opmode_blocking)
        [res3,armX]=vrep.simxGetObjectHandle(clientID, 'r_arm_x', vrep.simx_opmode_blocking)
        [res4,elbowY]=vrep.simxGetObjectHandle(clientID, 'r_elbow_y', vrep.simx_opmode_blocking)
        [res5,wristZ]=vrep.simxGetObjectHandle(clientID, 'r_wrist_z', vrep.simx_opmode_blocking)
        [res6,wristX]=vrep.simxGetObjectHandle(clientID, 'r_wrist_x', vrep.simx_opmode_blocking)
        
        vrep.simxSetJointTargetPosition(clientID, armX, math.radians(armX_), vrep.simx_opmode_blocking) # simx_opmode_blocking
        time.sleep(sleepFactor)
        vrep.simxSetJointTargetPosition(clientID, shoulderY, math.radians(shoulderY_), vrep.simx_opmode_blocking) # 30
        time.sleep(sleepFactor)
        vrep.simxSetJointTargetPosition(clientID, elbowY, math.radians(elbowY_), vrep.simx_opmode_blocking) # -50
        time.sleep(sleepFactor)
        vrep.simxSetJointTargetPosition(clientID, shoulderZ, math.radians(shoulderZ_), vrep.simx_opmode_blocking) # 30
        time.sleep(sleepFactor)
        vrep.simxSetJointTargetPosition(clientID, wristZ, math.radians(wristZ_), vrep.simx_opmode_blocking) #-50
        time.sleep(sleepFactor)
        vrep.simxSetJointTargetPosition(clientID, wristX, math.radians(wristX_), vrep.simx_opmode_blocking) #30
        time.sleep(sleepFactor)
 
    def getTargetPosition(self):
        [res7,target]=vrep.simxGetObjectHandle(clientID, 'Target', vrep.simx_opmode_blocking)
        [res, pos] = vrep.simxGetObjectPosition(clientID, target, -1, vrep.simx_opmode_blocking)

        return pos 
    
    def resetGoal(self,loc):
        [res7,target]=vrep.simxGetObjectHandle(clientID, 'Target', vrep.simx_opmode_blocking)
        vrep.simxSetModelProperty(clientID, target, 32+64, vrep.simx_opmode_blocking) 
        vrep.simxSetObjectOrientation(clientID,target,-1,[math.pi, 0., 0.],vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(clientID,target,-1, loc, vrep.simx_opmode_blocking)
        vrep.simxSetModelProperty(clientID, target, 0, vrep.simx_opmode_blocking)
        
    def reset(self,loc=[0.45,-0.10,0.53620]):
        setShoulderZ(33)
        time.sleep(sleepFactor+2.5) 
        self.resetGoal(loc)
        time.sleep(sleepFactor/2.)
    
    def dist2goal(self):
        [res7,Dummy]=vrep.simxGetObjectHandle(clientID, 'Dummy', vrep.simx_opmode_blocking)
        
        tPos = self.getTargetPosition()
        [res, gripperPos] = vrep.simxGetObjectPosition(clientID, Dummy, -1, vrep.simx_opmode_blocking)
        dist = np.linalg.norm(np.array(gripperPos)-np.array(tPos))
        
        return dist 
     
    def targetFell(self):
        while True:
            [res7,target]=vrep.simxGetObjectHandle(clientID, 'Target', vrep.simx_opmode_blocking)
            if res7 ==0: break
        while True:
            [res,ori] = vrep.simxGetObjectOrientation(clientID, target,-1,vrep.simx_opmode_blocking) # in radian
            if res ==0 and not -3.2<= ori[0]<= -2.9: break
        dist = np.linalg.norm(np.array([ori[0],ori[1],0.])-np.array([math.pi,0.,0.]))

        return dist > 0.2 
    
    def _getImage(self): 
        while True:
            err, res, img =  vrep.simxGetVisionSensorImage(clientID, handleSensor1, 0, vrep.simx_opmode_blocking) # 0 for RGB
            if err == 0: break
        img = np.array(img,dtype = np.uint8)
        img = img.reshape((res[1],res[0],3))
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img

    def writeImage(self,img):
        filename='nico_' + str(time.clock())
        cv2.imwrite("img2/"+filename +".png",img)
        
    def test_for_collsion(self): # return binary True or False
        [res, robot_table] = vrep.simxGetCollisionHandle(clientID, 'robot_table',vrep.simx_opmode_blocking)
        
        return vrep.simxReadCollision(clientID, robot_table, vrep.simx_opmode_blocking)[1]
        
    def closeHand(self): # every finger joint is in [-75, 75]... Actuating r_fingers_x will actuate the other finger jnts
        [res,indexFing]=vrep.simxGetObjectHandle(clientID, 'r_indexfingers_x', vrep.simx_opmode_blocking)
        [res,thumb]=vrep.simxGetObjectHandle(clientID, 'r_thumb_x', vrep.simx_opmode_blocking)
        
        vrep.simxSetJointTargetPosition(clientID, indexFing, math.radians(-50), vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetPosition(clientID, thumb, math.radians(-50), vrep.simx_opmode_blocking)
        
    def openHand(self):
        [res,indexFing]=vrep.simxGetObjectHandle(clientID, 'r_indexfingers_x', vrep.simx_opmode_blocking)
        [res,thumb]=vrep.simxGetObjectHandle(clientID, 'r_thumb_x', vrep.simx_opmode_blocking)
        
        vrep.simxSetJointTargetPosition(clientID, indexFing, math.radians(0), vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetPosition(clientID, thumb, math.radians(0), vrep.simx_opmode_blocking)
        
    def generate_randomGoals (self, size): # size = 50
        self.setJointConfiguration(-65,35,-40,33,-50,30)
        time.sleep(sleepFactor)
        goals = np.zeros([size,3])
        for i in range(size):
            print("sammple generated", i+1)
            setShoulderZ(i+43)
            time.sleep(sleepFactor)
            [res7,Dummy]=vrep.simxGetObjectHandle(clientID, 'Dummy', vrep.simx_opmode_blocking)
            [res, gripperPos] = vrep.simxGetObjectPosition(clientID, Dummy, -1, vrep.simx_opmode_blocking)
            goals[i] = np.concatenate([gripperPos[0:2],np.array([0.53620])])

        np.savetxt('goals_new',goals)
        
    def _reward (self):
        self.targetFell()
        if self.targetFell():
            reward = -1.
        else:      
            dist2goal = self.dist2goal()
            if dist2goal <=0.03:
                #time.sleep(0.5)
                loc_ = self.getTargetPosition()
                time.sleep(sleepFactor/2)
                # test for successful grasp, then reward with +1 for success. Otherwise, move config to the previously registered joint value 
                self.closeHand()
                time.sleep(sleepFactor/2)
                setShoulderZ(getShoulderZ()-20)
                #time.sleep(sleepFactor+0.3) # consider increasing the time b/c of frictions
                if self.dist2goal()<0.03:
                    print("success")
                    reward = 1.
                    self.openHand()
                    #time.sleep(0.8)
                else:
                    self.openHand()
                    time.sleep(0.8)
                    setShoulderZ(getShoulderZ()+20)
                    #time.sleep(sleepFactor)
                    self.resetGoal(loc_)
                    reward = -(dist2goal**2) 
            else:
                reward = -(dist2goal**2)
        
        return reward
    
    def step(self, action):
        prev_config = getShoulderZ() 
        done = False
        
        joint = action[0] + prev_config

        if joint < j1_limit[0]: joint = j1_limit[0]; 
        elif joint > j1_limit[1]: joint = j1_limit[1]

        obs = self.retrieve(joint)
        reward = self._reward()
        
        if reward == 1. or reward == -1:
            done = True
        
        return  obs, reward, done
    
    def retrieve (self, joint):
        setShoulderZ(joint)
        #time.sleep(sleepFactor)
        img = self._getImage().astype(float); self.writeImage(img);
        img/=256.0 
        img = img.reshape((dim2,dim1,3))

        return img

def setArm(angle): # static on -70 [-100,-65]
    [res3,armX]=vrep.simxGetObjectHandle(clientID, 'r_arm_x', vrep.simx_opmode_blocking)
    vrep.simxSetJointTargetPosition(clientID, armX, math.radians(angle), vrep.simx_opmode_blocking)

def setShoulderZ(angle): # init on 33 [-100,-60]
    [res3,shoulderZ]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_z', vrep.simx_opmode_blocking)
    vrep.simxSetJointTargetPosition(clientID, shoulderZ, math.radians(angle), vrep.simx_opmode_blocking)
    
def setShoulderY(angle): # init on 35 [ , ]
    [res3,shoulderY]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_y', vrep.simx_opmode_blocking)
    vrep.simxSetJointTargetPosition(clientID, shoulderY, math.radians(angle), vrep.simx_opmode_blocking)

def setElbow(angle): # init on -40 [-70,-50]
    [res3,elbowY]=vrep.simxGetObjectHandle(clientID, 'r_elbow_y', vrep.simx_opmode_blocking)
    vrep.simxSetJointTargetPosition(clientID, elbowY, math.radians(angle), vrep.simx_opmode_blocking)
    
def setWristZ(angle): # init on -50 [-70,-45]
    [res3,wristZ]=vrep.simxGetObjectHandle(clientID, 'r_wrist_z', vrep.simx_opmode_blocking)
    vrep.simxSetJointTargetPosition(clientID, wristZ, math.radians(angle), vrep.simx_opmode_blocking)
    
def setWristX(angle): # init on 30 [0, 50]
    [res3,wristX]=vrep.simxGetObjectHandle(clientID, 'r_wrist_x', vrep.simx_opmode_blocking)
    vrep.simxSetJointTargetPosition(clientID, wristX, math.radians(angle), vrep.simx_opmode_blocking)
    
#--------------------------get--------------------------------
    
def getArm(): 
    [res3,armX]=vrep.simxGetObjectHandle(clientID, 'r_arm_x', vrep.simx_opmode_blocking)
    return math.degrees(vrep.simxGetJointPosition(clientID, armX, vrep.simx_opmode_blocking)[1])

def getShoulderZ(): 
    [res3,shoulderZ]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_z', vrep.simx_opmode_blocking)
    return math.degrees(vrep.simxGetJointPosition(clientID, shoulderZ, vrep.simx_opmode_blocking)[1])

def getShoulderY(): 
    [res3,shoulderY]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_y', vrep.simx_opmode_blocking)
    return math.degrees(vrep.simxGetJointPosition(clientID, shoulderY, vrep.simx_opmode_blocking)[1])

def getElbow():
    [res3,elbowY]=vrep.simxGetObjectHandle(clientID, 'r_elbow_y', vrep.simx_opmode_blocking)
    return math.degrees(vrep.simxGetJointPosition(clientID, elbowY, vrep.simx_opmode_blocking)[1])
    
def getWristZ(): 
    [res3,wristZ]=vrep.simxGetObjectHandle(clientID, 'r_wrist_z', vrep.simx_opmode_blocking)
    return math.degrees(vrep.simxGetJointPosition(clientID, wristZ, vrep.simx_opmode_blocking)[1])
    
def getWristX(): 
    [res3,wristX]=vrep.simxGetObjectHandle(clientID, 'r_wrist_x', vrep.simx_opmode_blocking)
    return math.degrees(vrep.simxGetJointPosition(clientID, wristX, vrep.simx_opmode_blocking)[1])

