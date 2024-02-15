import gym
import place_env

from place_db import PlaceDB
from comp_res import comp_res 



placedb = PlaceDB('bm7')

env = gym.make('place_env-v0', placedb = placedb).unwrapped
env.reset()

for id in range(placedb.node_cnt-1):#TODO: change to placedb.node_cnt-1 is a tmp solution
# for id in range(50):
    next_state = env.step(env.action_space.sample())
    env.save_fig("./logs/env_test/step_{}.png".format(id))
    env.save_wiremask("./logs/env_test/wiremask_{}.png".format(id))


print("done!")
