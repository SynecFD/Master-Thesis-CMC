import tensorflow as tf
from keras import backend as K
from tensorflow.compat.v1.keras import backend as Kcompat
from keras.layers import Input, Dense, Flatten, Conv2D, Reshape, UpSampling2D, concatenate
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from numpy.random import default_rng
from keras.callbacks import TensorBoard
from keras.backend import clear_session

config = tf.compat.v1.ConfigProto(device_count={'GPU': 1 , 'CPU': 8}, allow_soft_placement=True, log_device_placement=False)
#new_sess = tf.Session()
#K.set_session(new_sess)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
config.gpu_options.allow_growth = True
graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session(graph=graph, config=config)
tf.compat.v1.keras.backend.set_session(sess)
tf.compat.v1.disable_eager_execution()
init = tf.compat.v1.global_variables_initializer()

laten_code_dim = 32
action_dim = 3

input_img_shape = (64, 64, 3)

rng = default_rng()

def get_graph():
    return graph

def create_session():
    clear_session()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    config.gpu_options.allow_growth = True
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph, config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        tf.compat.v1.disable_eager_execution()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    return sess

def CAE_critic() :  # Minimized Conv AutoEncoder-Critic Net    
    input_img = Input(shape=input_img_shape, name='input_img')
    input_act = Input(shape=(action_dim,),name='input_act')   
    x = Conv2D(8, (8, 8), strides=4, activation='relu', name='conv1')(input_img) 
    x = Conv2D(16, (4, 4), strides=1,  activation='relu', name='conv2')(x) 
    x = Flatten()(x)
    encoded = Dense((laten_code_dim), activation='relu', name = 'dense1')(x)
    encoded_with_act = concatenate([encoded, input_act])
     
    x = Dense((1536),activation='relu', name='dense2')(encoded)
    x = Reshape((12,8,16))(x)
    x = Conv2D(16, (4, 4), strides=1, padding= 'same', activation='relu', name='dec_conv1')(x)
    x = UpSampling2D ((2,2))(x) 
    x = Conv2D(8, (10, 2), strides=1, activation='relu', name='dec_conv2')(x)
    x = UpSampling2D ((5,5))(x) 
    decoded = Conv2D(3, (12, 12), strides=1, activation='relu', name='dec_conv3')(x) 
    
    h_val = Dense((20),activation='relu')(encoded_with_act) # the critic branch
    output_val = Dense((1))(h_val)
    critic = Model(inputs = [input_img,input_act], outputs = output_val) 
    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder_critic = Model([input_img, input_act], [decoded, output_val])
    adam = Adam(lr = 0.001)
    autoencoder_critic.compile(optimizer=adam, loss=['mse','mse'], loss_weights=[0.1,1.]) # loss weighs for AE and Critic losses 
    
    return autoencoder_critic, autoencoder, encoder, critic 

def autoencoder() :  
    input_img = Input(shape=input_img_shape, name='input_img')
    x = Conv2D(8, (8, 8), strides=4, activation='relu', name='conv1')(input_img) 
    x = Conv2D(16, (4, 4), strides=1,  activation='relu', name='conv2')(x) 
    x = Flatten()(x)
    encoded = Dense((laten_code_dim), activation='relu', name = 'dense1')(x)
     
    x = Dense((1536),activation='relu', name='dense2')(encoded)
    x = Reshape((12,8,16))(x)
    x = Conv2D(16, (4, 4), strides=1, padding= 'same', activation='relu', name='dec_conv1')(x)
    x = UpSampling2D ((2,2))(x) 
    x = Conv2D(8, (10, 2), strides=1, activation='relu', name='dec_conv2')(x)
    x = UpSampling2D ((5,5))(x) 
    decoded = Conv2D(3, (12, 12), strides=1, activation='relu', name='dec_conv3')(x) 
    
    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)
    adam = Adam(lr = 0.001)
    autoencoder.compile(optimizer=adam, loss='mse') 
    
    return autoencoder, encoder

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
    output1 = Dense((laten_code_dim),activation='relu',name='out1')(x) # predicted state
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
    lr, epochs = 0.01, 5 # learning rate and no. of training epochs (optimization iterations)
    _transfer_weights(world, rolledout_world)
    outputTensor1, outputTensor2, outputTensor3 = rolledout_world.output[0], rolledout_world.output[1], rolledout_world.output[2]
    loss = 0.5*(K.square(1.-outputTensor1)+K.square(1.-outputTensor2)+K.square(1.-outputTensor3)) # loss_plan
    gradients = K.gradients(loss, rolledout_world.input)
    func = K.function([rolledout_world.input[0],rolledout_world.input[1],rolledout_world.input[2]], gradients)
    for _ in range(epochs):
        input0 = np.array([np.concatenate((state, initial_plan[0]))])
        grad_val = func([input0, np.array([initial_plan[1]]), np.array([initial_plan[2]])])
        g1, g2, g3 = grad_val[0][0][laten_code_dim:], grad_val[1][0], grad_val[2][0] # gradient of loss_plan w.r.t. each individual action in initial_plan
        updated_plan = np.array([initial_plan[0,0]-lr*g1, initial_plan[1,0]-lr*g2, initial_plan[2,0]-lr*g3])
        initial_plan = updated_plan.reshape((-1,3))
	
    return initial_plan[0] # returns the optimal plan's 1st action

def compute_initial_plan(state, actor, world): # gives good initial action plan to the MPC to optimize
    actor_output1 = actor.predict(np.array([state]), batch_size = 1)[0]
    next_s = world.predict(np.array([np.concatenate((state, actor_output1))]), batch_size = 1)[0][0]    
    actor_output2 = actor.predict(np.array([next_s]), batch_size = 1)[0]
    next_s = world.predict(np.array([np.concatenate((next_s, actor_output2))]), batch_size = 1)[0][0]
    actor_output3 = actor.predict(np.array([next_s]), batch_size = 1)[0]
    
    initial_plan = np.array([actor_output1,actor_output2,actor_output3])
    #initial_plan = tf.convert_to_tensor(initial_plan)
    
    return initial_plan

def compute_graph_only(world, rolledout_world, state, initial_plan):
    lr, epochs = 0.01, 10 # learning rate and no. of training epochs (optimization iterations)
    _transfer_weights(world, rolledout_world)
    outputTensor1, outputTensor2, outputTensor3 = rolledout_world.output[0], rolledout_world.output[1], rolledout_world.output[2]
    loss = 0.5*(K.square(1.-outputTensor1)+K.square(1.-outputTensor2)+K.square(1.-outputTensor3)) # loss_plan
    gradients = K.gradients(loss, rolledout_world.input)
    func = K.function([rolledout_world.input[0],rolledout_world.input[1],rolledout_world.input[2]], gradients)
    for _ in range(epochs):
        
        input0 = np.array([np.concatenate((state, initial_plan[0]))])
        grad_val = func([input0, np.array([initial_plan[1]]), np.array([initial_plan[2]])])
	
    return initial_plan[0] # returns the optimal plan's 1st action

def sample_random_action():
    initial_action = (2 * rng.random(size=(3,3))) -1
    #print(initial_action)
    return initial_action

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
