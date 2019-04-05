import numpy as np
import _pickle as pickle
import gym
import os.path

# x is the input vector that holds 210 x 160 X 3 pixels of info from the game

# input
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many iterations do we want to update weights
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 #decay factor for RMSProp leaky sum of grad^2
resume = True
render = True 
np.random.seed(1); # random seed for repeatable results      
running_reward = None


#model initialization
D = 80*80 #input dimensionality (80*80 grid)

if resume:
    model = pickle.load( open( 'save.p', 'rb' ) )
    
    if os.path.isfile('reward.p'):
        reward_v = pickle.load( open( 'reward.p', 'rb') )
        running_reward = reward_v[ "running_mean" ][-1]
    else:
        # store reward vectors    
        reward_v = {}
        reward_v[ "reward_sum" ] = []
        reward_v[ "running_mean" ] = []

else:
    model = {}

    # random initial weights
    model[ "W1" ] = np.random.randn(H, D) / np.sqrt( D ) # "xavier" initialization
    model[ "W2" ] = np.random.randn(H) / np.sqrt( H )


grad_buffer = { k : np.zeros_like(v) for k, v in model.items() } # update buffers that add up gradiens over a batch
rmsprop_cache = { k : np.zeros_like(v) for k, v in model.items() } #rmspop memory

""" "smashing" results into domain [0, 1] """
def sigmoid(x):
    return 1.0 / ( 1.0 + np.exp(-x) ) # sigmoid function (gives propability of going up)  


""" prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
def preprocessing(I):
    I = I[ 35:195 ] # crop
    I = I[::2, ::2, 0] # downsample by factor 2
    I[ I == 144 ] = 0 # erase backround type 1
    I[ I == 109 ] = 0 # erase background type 2
    I[ I != 0 ] = 1 # ball and paddles 
    
    return I.astype(np.float).ravel()


""" take 1D float reward array and compute discounted reward """
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_ad = 0

    for t in reversed( range( 0, r.size ) ):
        if r[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!!)

        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r
    

""" calculate action propabilities based on the last preprocessed frame """
def policy_forward(x):
    h = np.dot( model[ "W1" ], x ) #compude hidden layer neuron activations
    h [ h<0 ] = 0 #ReLU nonlinearity threshold at zero
    logp = np.dot( model[ "W2" ], h ) # compute log propability of going up
    p = sigmoid(logp)

    return p, h # return propability of action and hidden state


""" backwards propagation (eph is array of intermediate hidden states)"""
def policy_backwards( eph, epdlogp ):
    dW2 = np.dot( eph.T, epdlogp ).ravel()
    dh = np.outer( epdlogp, model[ "W2"] )
    dh[ eph <= 0 ] = 0 # backprop preLU
    dW1 = np.dot( dh.T, epx )

    return {"W1": dW1, "W2": dW2}

# RUNNING THE MODEL

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
xs, hs, dlogps, drs = [], [], [], []
reward_sum = 0
episode_number = 0

while True:
    if render: env.render()

    # preprocess input
    cur_x = preprocessing( observation )
    if prev_x is None:
        x = cur_x - np.zeros(D)
    else:
        x = cur_x - prev_x
    prev_x = cur_x

    # calculate propability to go up and take a sample
    aprob, h = policy_forward(cur_x)
    action = 2 if np.random.uniform() < aprob else 3 #sample! i e roll the dice

    # record various intermediates needed for backprop
    xs.append(x) #observation
    hs.append(h) #hidden layer
    y = 1 if action == 2 else 0
    dlogps.append( y - aprob )

    # step the environment to get a new observation
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward) # record reward

    if done:
        # an episode finished
        episode_number += 1

        # stack together all inputs, hidden layers, action gradients and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], [] # reset arrays

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the reward to be unit normal
        discounted_epr -= np.mean( discounted_epr )
        discounted_epr /= np.std( discounted_epr )

        epdlogp = epdlogp * discounted_epr # modulate the gradient with advantage ( the PG magic happens right here )
        grad = policy_backwards( eph, epdlogp )

        for k in model:
            grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every [batch size] episodes
        if episode_number % batch_size == 0:
            
            for k, v in model.items():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / ( np.sqrt( rmsprop_cache[k] ) + 1e-5 )
                grad_buffer[k] = np.zeros_like( v )


        # boring bookkeeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print("resetting env. episode %d finished. reward total was %f. Running mean %f." % ( episode_number, reward_sum, running_reward ))

        # save running mean to model
        reward_v[ "reward_sum" ].append( reward_sum )
        reward_v[ "running_mean" ].append( running_reward )
        pickle.dump( reward_v, open('reward.p', 'wb') )

        if episode_number % 100 == 0:
            pickle.dump( model, open('save.p', 'wb') )
        reward_sum = 0
        observation = env.reset()
        prev_x = None

    # if reward != 0: # pong has either a +1 or -1 reward exactly when the game ends
        # print score
        # print( ( "ep %d: game finished, reward %f" % ( episode_number, reward ) ) + ('' if reward == -1 else '!!!!!!!') )
