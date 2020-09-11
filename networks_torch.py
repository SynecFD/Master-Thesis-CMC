import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from torch.nn import Module, Conv2d, Linear, MSELoss
import torch.nn.functional as F
import torch.optim as optim
from keras import backend as K
from keras.layers import Input, Dense, Flatten, Reshape, UpSampling2D, concatenate
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import TensorBoard


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

laten_code_dim = 32
action_dim = 1


# pytorch
device = torch.device("cuda:0")

class CAE_critic_Module(Module):
    def __init__(self, D_in):
        super(CAE_critic, self).__init__
        self.lin3 = Linear(3, 20)
        self.out = Linear(20, 1)
        #self.upsample1 = Upsample(size=(2,2))

    def forward(self, x):
        x = F.relu(self.lin3(x))
        output_val = self.out(x)
        return output_val

#cae_critic_module = CAE_critic_Module(64*32*3)
#criterion = MSELoss()

class CAE_Actor(Module):
    def __init__(self, D_in):
        super(CAE_critic, self).__init__
        self.lin1 = Linear(D_in, 20)
        self.out = Linear(20, 1)

    def forward(self, x):
        x = F.tanh(self.lin1(x))
        x = F.tanh(self.out(x))
        return x

class Encoder(Module):
    def __init__(self, D_in, H, D_out):
        super(CAE_critic, self).__init__
        self.conv1 = Conv2d(1, 8, kernel_size=8, stride=4)
        self.conv2 = Conv2d(8, 16, kernel_size=4)
        self.lin1 = Linear(((D_in-6)//4) * 16, laten_code_dim)
        self.lin2 = Linear(laten_code_dim, 768)
        self.conv3 = Conv2d(12, 16, kernel_size=4, padding_mode='replicate')
        self.conv4 = Conv2d(16, 8, kernel_size=(10,2))
        self.conv5 = Conv2d(8, 3, kernel_size=(12,4))
        self.lin3 = Linear(3, 20)
        self.out = Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        encoded = F.sigmoid(self.lin1(x))
        return encoded

class Auto_Enc_Critic(Module):
    def __init__(self, D_in, H, D_out):
        super(CAE_critic, self).__init__
        self.conv1 = Conv2d(1, 8, kernel_size=8, stride=4)
        self.conv2 = Conv2d(8, 16, kernel_size=4)
        self.lin1 = Linear(((D_in-6)//4) * 16, laten_code_dim)
        self.lin2 = Linear(laten_code_dim, 768)
        self.conv3 = Conv2d(12, 16, kernel_size=4, padding_mode='replicate')
        self.conv4 = Conv2d(16, 8, kernel_size=(10,2))
        self.conv5 = Conv2d(8, 3, kernel_size=(12,4))
        self.lin3 = Linear(3, 20)
        self.out = Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.lin2(x))
        x = x.view(12, 4, 16)
        x = F.relu(self.conv3(x))
        x = F.interpolate(size=(2,2), input=x)
        x = F.relu(self.conv4(x))
        x = F.interpolate(size=(5,5), input=x)
        decoded = F.sigmoid(self.conv5(x))
        return decoded

class World_Module(Module):
    def __init__(self, D_in, H, D_out):
        super(CAE_critic, self).__init__
        
    def forward(self, x):
        return 0

#def train_net():
#    critic_optim = optim.Adam(cae_critic_module.parameters(), lr=0.001)
#    critic_optim.zero_grad()
#    critic_output = cae_critic_module()
#    loss = criterion(critic_output)


def CAE_critic() :  # Minimized Conv AutoEncoder-Critic Net    
    input_img = DataLoader(64, 32, 3)
    input_act = Input(shape=(action_dim,),name='input_act')   
    x = Conv2D(8, (8, 8), strides=4, activation='relu', name='conv1')(input_img) 
    x = Conv2D(16, (4, 4), strides=1,  activation='relu', name='conv2')(x) 
    x = Flatten()(x)
    encoded = Dense((laten_code_dim), activation='sigmoid', name = 'dense1')(x)
    encoded_with_act = concatenate([encoded, input_act])
     
    x = Dense((768),activation='relu', name='dense2')(encoded)
    x = Reshape((12,4,16))(x)
    x = Conv2D(16, (4, 4), strides=1, padding= 'same', activation='relu', name='dec_conv1')(x)
    x = UpSampling2D ((2,2))(x) 
    x = Conv2D(8, (10, 2), strides=1, activation='relu', name='dec_conv2')(x)
    x = UpSampling2D ((5,5))(x) 
    decoded = Conv2D(3, (12, 4), strides=1, activation='sigmoid', name='dec_conv3')(x) 
    
    h_val = Dense((20),activation='relu')(encoded_with_act) # the critic branch
    output_val = Dense((1))(h_val)
    critic = Model(inputs = [input_img,input_act], outputs = output_val) 
    encoder = Model(input_img, encoded)
    autoencoder_critic = Model([input_img, input_act], [decoded, output_val])
    adam = Adam(lr = 0.001)
    autoencoder_critic.compile(optimizer=adam, loss=['mse','mse'], loss_weights=[0.1,1.]) # loss weighs for AE and Critic losses 
    
    return autoencoder_critic, encoder, critic 

def actor():
    input_ = Input(shape=(32,))
    x = Dense((20),activation='tanh')(input_)
    output = Dense((action_dim),activation='tanh')(x)
    
    actor = Model(input_,output)
    adam = Adam(lr = 0.001)
    actor.compile(optimizer=adam, loss='mse')
    
    return actor   

def world(): # world model net  
    input_ = Input(shape=(laten_code_dim+action_dim,))
    x = Dense((20),activation='tanh',name='hid')(input_)
    output1 = Dense((laten_code_dim),activation='sigmoid',name='out1')(x) # predicted state
    output2 = Dense((1),activation='tanh',name='out2')(x) # predicted reward
    
    world = Model(input_, [output1, output2])
    world.compile(optimizer='adam', loss=['mse','mse'])
    
    return world

def rolledout_world(): # rolling out the world 3 timesteps into the future
    input1 = Input(shape=(laten_code_dim+action_dim,),name='input1') # input state_action (t)
    input2 = Input(shape=(action_dim,),name='input2') # input action (t+1)
    input3 = Input(shape=(action_dim,),name='input3') # input action (t+2)
    
    x1 = Dense((20),activation='tanh',name='hid1')(input1)
    output11 = Dense(laten_code_dim,activation='sigmoid',name='out11')(x1) # predicted state (t+1)
    output12 = Dense(1, activation='tanh',name='out12')(x1) # predicted reward (t+1)
    
    x = concatenate([output11, input2])
    x2 = Dense((20),activation='tanh',name='hid2')(x)
    output21 = Dense(laten_code_dim, activation='sigmoid', name='out21')(x2) # predicted state (t+2)
    output22 = Dense(1, activation='tanh',name='out22')(x2) # predicted reward (t+2)
    
    x = concatenate([output21, input3])
    x3 = Dense((20),activation='tanh',name='hid3')(x)
    output31 = Dense(laten_code_dim,activation='sigmoid',name='out31')(x3) # predicted state (t+3)
    output32 = Dense(1, activation='tanh',name='out32')(x3) # predicted reward (t+3)
    
    model = Model(inputs = [input1,input2,input3], outputs = [output12, output22, output32])
    model.compile(optimizer='adam', loss=['mse','mse', 'mse'])
    
    return model        

def mpc(world, rolledout_world, state, initial_plan): # Model Predictive Control
    lr, epochs = 0.01, 10 # learning rate and no. of training epochs (optimization iterations)
    _transfer_weights(world, rolledout_world)
    outputTensor1, outputTensor2, outputTensor3 = rolledout_world.output[0], rolledout_world.output[1], rolledout_world.output[2]
    loss = 0.5*(K.square(1.-outputTensor1)+K.square(1.-outputTensor2)+K.square(1.-outputTensor3)) # loss_plan
    gradients = K.gradients(loss, rolledout_world.input)
    func = K.function([rolledout_world.input[0],rolledout_world.input[1],rolledout_world.input[2], K.learning_phase()], gradients)
    
    for i in range(epochs):
        input0 = np.array([np.concatenate((state,initial_plan[0]))])
        grad_val = func ([input0, np.array([initial_plan[1]]), np.array([initial_plan[2]]), 0])
        g1, g2, g3 = grad_val[0][0][laten_code_dim+action_dim-1], grad_val[1][0][0], grad_val[2][0][0] # gradient of loss_plan w.r.t. each individual action in initial_plan
        updated_plan = np.array([initial_plan[0,0]-lr*g1, initial_plan[1,0]-lr*g2, initial_plan[2,0]-lr*g3])
        initial_plan = updated_plan.reshape((-1,1))
	
    return initial_plan[0] # reuturns the optimal plan's 1st action

def compute_initial_plan(state, actor, world): # gives good initial action plan to the MPC to optimize
    actor_output1 = actor.predict(np.array([state]), batch_size = 1)[0]
    next_s = world.predict(np.array([np.concatenate((state, actor_output1))]), batch_size = 1)[0][0]    
    actor_output2 = actor.predict(np.array([next_s]), batch_size = 1)[0]
    next_s = world.predict(np.array([np.concatenate((next_s, actor_output2))]), batch_size = 1)[0][0]
    actor_output3 = actor.predict(np.array([next_s]), batch_size = 1)[0]
    
    initial_plan = np.array([actor_output1,actor_output2,actor_output3]) 
    
    return initial_plan

def _transfer_weights(world, rolledout_world):
    params_hid = world.get_layer('hid').get_weights() 
    params_out1 = world.get_layer('out1').get_weights()
    params_out2 = world.get_layer('out2').get_weights()
    
    rolledout_world.get_layer('hid1').set_weights(params_hid)
    rolledout_world.get_layer('hid2').set_weights(params_hid)
    rolledout_world.get_layer('hid3').set_weights(params_hid)
    
    rolledout_world.get_layer('out11').set_weights(params_out1)
    rolledout_world.get_layer('out21').set_weights(params_out1)
    
    rolledout_world.get_layer('out12').set_weights(params_out2)
    rolledout_world.get_layer('out22').set_weights(params_out2)
    rolledout_world.get_layer('out32').set_weights(params_out2)
