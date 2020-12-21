from collections import deque
import os
import subprocess
import sys
import gym
from mpi4py import MPI
from parameters import *
import tensorflow as tf
from keras.models import load_model, clone_model, save_model
from networks import actor, CAE_critic, world, rolledout_world, mpc, compute_initial_plan, create_session, compute_graph_only, sample_random_action
import random, numpy as np, env
from utils import PARSER
from keras.backend import clear_session


tf.config.experimental.list_physical_devices('GPU')

### MPI related code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class CMC (object):
    def __init__(self, args):
        self.env = env.make_env(args)
        self.actor = actor()
        self.autoencoder_critic, self.encoder, self.critic = CAE_critic()
        self.critic_t = clone_model(self.critic); self.critic_t.set_weights(self.critic.get_weights())
        self.actor_t = clone_model(self.actor); self.actor_t.set_weights(self.actor.get_weights())
        self.world = world()
        self.rolledout_world = rolledout_world()
        self.load_weights()
        self.memory = deque(maxlen=int((1e+5)/8))
        self.goals = np.loadtxt("goals")
        
        self.errors = []
        self.avgs = []
        self.max_avg = 1.
        
    def restart(self): # restart to reset memory and speed up learning
        #self.env = env.make_env(args)
        clear_session()
        create_session()
        self.actor = actor()
        self.autoencoder_critic, self.encoder, self.critic = CAE_critic()
        self.world = world()
        self.rolledout_world = rolledout_world()
        self.load_weights()

    def load_weights(self):
        self.autoencoder_critic.load_weights('./autoencoder_W')
        self.world.load_weights('./world') 
        self.actor.load_weights('./actor_W') 
        self.critic.load_weights('./critic_W')

        self.critic_t = clone_model(self.critic); self.critic_t.set_weights(self.critic.get_weights())
        self.actor_t = clone_model(self.actor); self.actor_t.set_weights(self.actor.get_weights())

    def load_complete_models(self):
        self.autoencoder_critic = load_model('./autoencoder_W')
        self.world = load_model('./world') 
        self.actor = load_model('./actor_W') 
        self.critic = load_model('./critic_W')

        self.critic_t = clone_model(self.critic); self.critic_t.set_weights(self.critic.get_weights())
        self.actor_t = clone_model(self.actor); self.actor_t.set_weights(self.actor.get_weights())

    def lp (self, model, sensorimotor_stimulus, next_s, next_r): # computes the learning progress
        prediction = model.predict(np.array([sensorimotor_stimulus]), batch_size = 1)
        pred_s, pred_r = prediction[0][0], prediction[1][0]
        pred_err = np.linalg.norm(pred_s-next_s) + np.linalg.norm(pred_r-next_r)

        self.errors.append(pred_err)
        if len(self.errors)>40:
            self.errors.pop(0)
        self.avg = np.average(self.errors)
        self.avgs.append(self.avg)
        
        if len(self.avgs)>20:
            self.avgs.pop(0)            
        self.avg_old = 0.0 if len(self.avgs)==1 else self.avgs[0]
        self.max_avg = max(self.avgs)
        normalized_avg, normalized_prev_avg = self.avg/self.max_avg, self.avg_old/self.max_avg
        lp = normalized_prev_avg - normalized_avg # learning progress
        
        return lp
        
    def remember(self, state, action, ext_r, total_r, next_state, done):
        self.memory.append((state, action, ext_r, total_r, next_state, done))
    
    def replay(self):
        batch_size = min(BATCH_SIZE,len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X_c1, X_c2, X_a, X_w = np.zeros((batch_size,64,64,3)), np.zeros((batch_size,3)), [], np.zeros((batch_size,35))
        Y_c1, Y_c2, Y_a, Y_w1, Y_w2 = np.zeros((batch_size,64,64,3)), np.zeros((batch_size,)), [], np.zeros((batch_size,32)), np.zeros((batch_size,1))
        
        for i in range(batch_size):
            state, action, ext_r, total_r, next_state, done = minibatch[i] # action is scalar
            encoded_state = self.encoder.predict(np.array([state]),batch_size=1)[0]
            target0, target1 = state, total_r
            if not done:
                pol_next_t = self.actor_t.predict(np.array([encoded_state]))
                target1 += gamma * self.critic_t.predict([np.array([next_state]), pol_next_t])[0]
            X_c1[i], X_c2[i], Y_c1[i], Y_c2[i] = state, action, target0, target1
            encoded_next_state = self.encoder.predict(np.array([next_state]),batch_size=1)[0]
            X_w[i], Y_w1[i], Y_w2[i]= np.concatenate([encoded_state,action]), encoded_next_state, ext_r
            td_err = target1 - self.critic.predict([np.array([state]), np.array([X_c2[i]])], batch_size = 1)[0]
            if td_err>0.:
                X_a.append(encoded_state); Y_a.append(action)
        
        # update critic and actor (CACLA update)       
        print(Y_c1.shape)
        self.autoencoder_critic.fit([X_c1, X_c2], [Y_c1, Y_c2], batch_size=batch_size, epochs=5, verbose=0)
        if len(X_a)>0: self.actor.fit(x=np.asarray(X_a), y=np.asarray(Y_a), batch_size=len(X_a), epochs=15, verbose=0)
        
        # update the world model
        self.world.fit(X_w, [Y_w1, Y_w2], batch_size=batch_size, epochs=10, verbose=0)
        
        # update target networks
        critic_weights, critic_t_weights = self.critic.get_weights(), self.critic_t.get_weights()
        new_critic_target_weights = []
        w_cnt = len(critic_t_weights)
        for i in range(w_cnt):
            new_critic_target_weights.append((tnet_update_rate*critic_weights[i])+((1-tnet_update_rate)*critic_t_weights[i]))
        self.critic_t.set_weights(new_critic_target_weights)
        
        actor_weights, actor_t_weights = self.actor.get_weights(), self.actor_t.get_weights()
        new_actor_target_weights = []
        w_cnt = len(actor_t_weights)
        for i in range(w_cnt):
            new_actor_target_weights.append((tnet_update_rate*actor_weights[i])+((1-tnet_update_rate)*actor_t_weights[i]))
        self.actor_t.set_weights(new_actor_target_weights)

    
    def learn (self, nb_episodes = 1000, steps_per_episode = 900, sim = 1, nb_simulation = 1, only_mpc = False):
        reliable_planner = False
        steps = np.zeros((nb_episodes,))
        rewards = np.zeros((nb_episodes,))        
        
        for i_episode in range(nb_episodes):
            total_ext_reward_per_episode = 0.
            #goal = self.goals[i_episode%50]
            observation = self.env.reset()
            for t in range(steps_per_episode):
                self.env.render()
                print("Deep ICAC sim: {}/{} episode: {}/{}. Step: {}/{}".format(sim, nb_simulation, i_episode+1, nb_episodes, t+1, steps_per_episode))
                encoded_obs = self.encoder.predict(np.array([observation]), batch_size = 1)[0]
                
                if only_mpc: 
                    initial_plan = sample_random_action()
                    action = compute_graph_only(self.world, self.rolledout_world, encoded_obs, initial_plan) # outputs the optimal plan's 1st action
                elif reliable_planner:
                    initial_plan = compute_initial_plan(encoded_obs, self.actor, self.world)
                    #print("takes long")
                    action = mpc(self.world, self.rolledout_world, encoded_obs, initial_plan) # outputs the optimal plan's 1st action
                else:
                    action = self.actor.predict(np.array([encoded_obs]), batch_size = 1)[0]
                    
                action = np.random.normal(action, std) # exploration noise
                action[0] = action[0]/2
                if action[0] < 0.33 and action[0] > -0.33:
                    action[0] = 0
                #else:
                #    action[0] = 0

                action[1] = np.absolute(action[1]) # There is no negative acceleration, breaking is separate
                if action[1] < 1:
                    action[1] = 1
                #elif action[1]>0:
                #    action[1] = 1

                action[2] = action[2]/2
                if action[2] < 0:
                    action[2] = 0
                elif action[2]>1:
                    action[2] = 1
                

                clipped_act = action
                ready_act = clipped_act
                #print(ready_act)
                #ready_act = np.multiply(np.array([clipped_act]),[20])
                obs_new, r_ext, done, x = self.env.step(ready_act)
                encoded_next_obs = self.encoder.predict(np.array([obs_new]), batch_size = 1)[0]
                lp = self.lp(self.world, np.concatenate([encoded_obs, ready_act]), encoded_next_obs, r_ext)
                r_int = -lp
                reliable_planner = True if lp>=0.005 else False
                r_total = r_ext + r_int/(1+(D*(t+1+sum(steps))))
                total_ext_reward_per_episode += r_ext 
                
                self.remember(observation, clipped_act, r_ext, r_total, obs_new, done)   # was exp_action                          
                
                observation = obs_new
                if done:
                    break
            
            print("--updating--")
            self.replay()
                
            steps [i_episode] = t+1
            
            print("--saving networks--")
            try: 
                self.actor.save_weights("actor_W"); self.autoencoder_critic.save_weights("autoencoder_W"); self.world.save_weights("world")
                self.critic.save_weights("critic_W")
                save_model(self.actor,'actor'); save_model(self.autoencoder_critic, 'autoencoder_critic'); save_model(self.world, 'world')
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))   

            rewards [i_episode] = total_ext_reward_per_episode
            
            print("--saving results--")
            np.savetxt('rewards',rewards)
            np.savetxt('steps',steps)
            self.restart()
            #create_session()
        
        self.critic = None; self.critic_t = None; self.actor = None; self.actor_t = None
        
        return rewards, steps 

def sprint(*args):
    print(args) # if python3, can do print(*args)
    sys.stdout.flush()

def main(args):
    global optimizer, num_episode, num_test_episode, eval_steps, num_worker, num_worker_trial, antithetic, seed_start, retrain_mode, cap_time_mode, env_name, exp_name, batch_mode, config_args

    optimizer = args.controller_optimizer
    num_episode = args.controller_num_episode
    num_test_episode = args.controller_num_test_episode
    eval_steps = args.controller_eval_steps
    num_worker = args.controller_num_worker
    num_worker_trial = args.controller_num_worker_trial
    retrain_mode = (args.controller_retrain == 1)
    cap_time_mode= (args.controller_cap_time == 1)
    seed_start = args.controller_seed_start
    env_name = args.env_name
    exp_name = args.exp_name
    batch_mode = args.controller_batch_mode
    config_args = args

    #initialize_settings(args.controller_sigma_init, args.controller_sigma_decay)
    
    only_mpc = True

    sprint("process", rank, "out of total ", comm.Get_size(), "started")
    if(only_mpc):
        agent = CMC(args)
        rewards, steps = agent.learn(only_mpc=True)
    elif (rank == 0):
        agent = CMC(args)
        rewards, steps = agent.learn()
    else:
        agent = CMC(args)
        rewards, steps = agent.learn()
        #slave()

if __name__ == "__main__":
    print('---------Learning started-----------')
    args = PARSER.parse_args()
    args.mdrnn = False
    main(args)
    #agent = CMC(args)
    #rewards, steps  = agent.learn() 
