from collections import deque
from parameters import *
from keras.models import load_model, clone_model, save_model
from networks import actor, CAE_critic, world, rolledout_world, mpc, compute_initial_plan, CAE_critic_Module, CAE_Actor, Auto_Enc_Critic, Encoder
from torch.nn import MSELoss
import random, numpy as np, env


class CMC (object):
    def __init__(self):
        self.env = env.Env()
        self.actor = actor()
        self.autoencoder_critic, self.encoder, self.critic = CAE_critic()
        self.critic_t = clone_model(self.critic); self.critic_t.set_weights(self.critic.get_weights())
        self.actor_t = clone_model(self.actor); self.actor_t.set_weights(self.actor.get_weights())
        self.world = world()
        self.rolledout_world = rolledout_world()
        self.memory = deque(maxlen=int(1e+5)/8)
        self.goals = np.loadtxt("goals")
        self.errors = []
        self.avgs = []
        self.max_avg = 1.

        self.cae_critic_module = CAE_critic_Module(64*32*3)
        self.cae_critic_module = Auto_Enc_Critic(64*32*3)
        self.cae_critic_module = Encoder(64*32*3)
        self.actor_module = CAE_Actor(32)
        self.criterion = MSELoss()

    def train_loop(self, nb_episodes = 10000, steps_per_episode = 50, sim = 1, nb_simulation = 1):
        learning_rate = 1e-6
        for t in range(nb_episodes):
            # Forward pass: compute predicted y
            h = x.mm(w1)
            h_relu = h.clamp(min=0)
            y_pred = h_relu.mm(w2)

            # Compute and print loss
            loss = (y_pred - y).pow(2).sum().item()
            if t % 100 == 99:
                print(t, loss)

            # Backprop to compute gradients of w1 and w2 with respect to loss
            grad_y_pred = 2.0 * (y_pred - y)
            grad_w2 = h_relu.t().mm(grad_y_pred)
            grad_h_relu = grad_y_pred.mm(w2.t())
            grad_h = grad_h_relu.clone()
            grad_h[h < 0] = 0
            grad_w1 = x.t().mm(grad_h)

            # Update weights using gradient descent
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2

    def train_critic(target):
        critic_optim = optim.Adam(self.cae_critic_module.parameters(), lr=0.001)
        critic_optim.zero_grad()
        critic_output = self.cae_critic_module()
        loss = criterion(critic_output, target)
        loss.backward()
        critic_optim.step()

    def train_actor():
        actor_optim = optim.Adam(self.actor_module.parameters(), lr=0.001)
        actor_optim.zero_grad()
        actor_out = self.actor_module
        loss = criterion(actor_out, target)
        loss.backward()
        actor_optim.step()
        
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
        X_c1, X_c2, X_a, X_w = np.zeros((batch_size,64,32,3)), np.zeros((batch_size,1)), [], np.zeros((batch_size,33))
        Y_c1, Y_c2, Y_a, Y_w1, Y_w2 = np.zeros((batch_size,64,32,3)), np.zeros((batch_size,)), [], np.zeros((batch_size,32)), np.zeros((batch_size,1))
        
        for i in range(batch_size):
            state, action, ext_r, total_r, next_state, done = minibatch[i] # action is scalar
            encoded_state = self.encoder.predict(np.array([state]),batch_size=1)[0]
            target0, target1 = state, total_r
            if not done:
                pol_next_t = self.actor_t.predict(np.array([encoded_state]))
                target1 += gamma * self.critic_t.predict([np.array([next_state]), pol_next_t])[0]
            X_c1[i], X_c2[i], Y_c1[i], Y_c2[i] = state, action, target0, target1
            encoded_next_state = self.encoder.predict(np.array([next_state]),batch_size=1)[0]
            X_w[i], Y_w1[i], Y_w2[i]= np.concatenate([encoded_state,np.array([action])]), encoded_next_state, ext_r
            td_err = target1 - self.critic.predict([np.array([state]), np.array([X_c2[i]])], batch_size = 1)[0]
            if td_err>0.:
                X_a.append(encoded_state); Y_a.append(action)
        
        # update critic and actor (CACLA update)       
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

    
    def learn (self, nb_episodes = 10000, steps_per_episode = 50, sim = 1, nb_simulation = 1):
        reliable_planner = False
        steps = np.zeros((nb_episodes,))
        rewards = np.zeros((nb_episodes,))        
        
        for i_episode in range(nb_episodes):
            total_ext_reward_per_episode = 0.
            goal = self.goals[i_episode%50]
            self.env.reset(goal)
            observation = self.env.retrieve(env.getShoulderZ())
            for t in range(steps_per_episode):
                print "Deep ICAC sim: {}/{} episode: {}/{}. Step: {}/{}".format(sim, nb_simulation, i_episode+1, nb_episodes, t+1, steps_per_episode)
                encoded_obs = self.encoder.predict(np.array([observation]), batch_size = 1)[0]
                
                if reliable_planner:
                    initial_plan = compute_initial_plan(encoded_obs, self.actor, self.world)
                    action = mpc(self.world, self.rolledout_world, encoded_obs, initial_plan) # outputs the optimal plan's 1st action
                else:
                    action = self.actor.predict(np.array([encoded_obs]), batch_size = 1)[0]
                    
                action = np.random.normal(action[0], std) # exploration noise
                
                if action>1: 
                    action = 1
                else: 
                    if action<-1: action = -1
                
                clipped_act = action
             
                ready_act = np.multiply(np.array([clipped_act]),[20])
                obs_new, r_ext, done = self.env.step(ready_act)
                encoded_next_obs = self.encoder.predict(np.array([obs_new]), batch_size = 1)[0]
                lp = self.lp(self.world, np.concatenate([encoded_obs, ready_act]), encoded_next_obs, r_ext)
                r_int = -lp
                reliable_planner = True if lp>=0 else False
                r_total = r_ext + r_int/(1+(D*(t+1+sum(steps))))
                total_ext_reward_per_episode += r_ext 
                
                self.remember(observation, clipped_act, r_ext, r_total, obs_new, done)   # was exp_action                          
                
                observation = obs_new
                if done:
                    break  
            
            print "--updating--"
            self.replay()
                
            steps [i_episode] = t+1
            
            print "--saving networks--"
            try: 
                self.actor.save_weights("actor_W"); self.autoencoder_critic.save_weights("autoencoder_W"); self.world.save_weights("world")
                save_model(self.actor,'actor'); save_model(self.autoencoder_critic, 'autoencoder_critic'); save_model(self.world, 'world')
            except IOError as e:
                print "I/O error({0}): {1}".format(e.errno, e.strerror)   

            rewards [i_episode] = total_ext_reward_per_episode
            
            print "--saving results--"
            np.savetxt('rewards',rewards)
            np.savetxt('steps',steps)
        
        self.critic = None; self.critic_t = None; self.actor = None; self.actor_t = None
        
        return rewards, steps 
    
if __name__ == "__main__":
    print('---------Learning started-----------')
    agent = CMC()
    rewards, steps  = agent.learn() 
