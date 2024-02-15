import gym
import place_env

from place_db import PlaceDB


placedb = PlaceDB('adaptec1')

env = gym.make('place_env-v0',
               placedb = placedb,
               placed_num_macro = placedb.node_cnt).unwrapped
env.reset()

# for id in range(placedb.node_cnt):
for id in range(50):
    next_state = env.step(env.action_space.sample())
    env.save_fig("./env_test/step_{}.png".format(id))
    env.save_wiremask("./env_test/wiremask_{}.png".format(id))


print("done!")
