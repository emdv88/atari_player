import _pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# plot rewards from the last pong games

def update(frames):
    
    reward = pickle.load( open( 'reward.p', 'rb' ) )
    
    ax.clear()
    ax.plot( reward[ "reward_sum" ] )
    ax.plot( reward[ "running_mean"] )

figure = plt.figure()
ax = figure.add_subplot(1, 1, 1)
a = anim.FuncAnimation(figure, update, repeat=False)
plt.show()

