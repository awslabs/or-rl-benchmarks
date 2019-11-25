from bin_packing_environment import BinPackingGymEnvironment, BinPackingActionMaskGymEnvironment
import csv

def get_action(state):
    state = state['real_obs']
    num_bins_level = state[:-1]
    item_size = state[-1]
    bag_capacity = len(state)-1

    if item_size == 0:
        print('item size should be larger than 0')
        return 0
    
    if item_size == bag_capacity:
        return 0 # new bag
    
    for i in range(len(num_bins_level)-item_size, 0, -1):
        if num_bins_level[i] > 0: #there is at least one bin at this level
            return i
    
    return 0

env_config = {
                 "bag_capacity": 100,
                 'item_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9],
#                  'item_probabilities': [0.14, 0.10, 0.06, 0.13, 0.11, 0.13, 0.03, 0.11, 0.19], #bounded waste
                 'item_probabilities': [0.06, 0.11, 0.11, 0.22, 0, 0.11, 0.06, 0, 0.33], #perfect pack
#                  'item_probabilities': [0, 0, 0, 1/3, 0, 0, 0, 0, 2/3], #linear waste
                 'time_horizon': 10000,
             }        
        
env = BinPackingActionMaskGymEnvironment(env_config)
state = env.reset()

done = False
total_reward = 0
while not done:
    action = get_action(state)
    state, reward, done, info = env.step(action)
    total_reward += reward

print("Total reward for best fit baseline agent: ", total_reward)


