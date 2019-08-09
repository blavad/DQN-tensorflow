import cv2
import random
import numpy as np
from sklearn.datasets import fetch_openml

from flappyBird_cnn import FlappyBirdCnnEnv

def getState(env, resize = None, linear=False):
    obs = env.state
    if resize is not None:
        obs = cv2.resize(obs, dsize=resize, interpolation=cv2.INTER_LINEAR)
    if linear:
        obs = np.reshape(obs, (obs.shape[0]*obs.shape[1]))
    return obs
    
def dy(env):
    return (env.bird.y + env.bird.rayon/2) - (env._get_current_plateform(env.bird).get_pos_ouv()+env._get_current_plateform(env.bird).get_size_ouv()/2)    

def rndFlappyPosition(env):
    return random.randint(env.bird.rayon, env.height-env.bird.rayon)

def getFlappyData(n_data = 10000):
    n_step = 5000
    data_per_step = n_data/n_step
    env = FlappyBirdCnnEnv()
    obs = env.reset()
    it=0
    n = 0
    X = []
    y = []
    while n < n_data:
        env.step(it%2)
        if data_per_step >1:
            for i in range(int(data_per_step)):
                env.bird.y = rndFlappyPosition(env)        
                X.append(getState(env))
                y.append(0 if dy(env)>0 else 1)
                n+=1
        else:
            if it==int((n/n_data)*n_step):
                env.bird.y = rndFlappyPosition(env)        
                X.append(getState(env))
                y.append(0 if dy(env)>0 else 1)
                n+=1
        it+=1
    return np.array(X), np.array(y), env.action_space.n


def getDataEpisode(model, env):
    done = False
    obs = env.reset()
    X = []
    y = []
    shape_obs = list(obs.shape)
    while not done:
        obs = np.reshape(obs, [1]+shape_obs)
        outputs = model.predict(obs)
        action = np.argmax(outputs)
        obs = np.reshape(obs, shape_obs)
        X.append(obs)
        y.append(action)
        obs, rew, done, info = env.step(action)
    return np.array(X), np.array(y), env.action_space.n

def getMnistData():
    mnist = fetch_openml("mnist_784")
    X = mnist.data / 255.0
    y = mnist.target
    return X, y, 10

