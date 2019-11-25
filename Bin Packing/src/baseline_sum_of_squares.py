from bin_packing_environment import BinPackingActionMaskGymEnvironment

def get_action(state):
    state = state['real_obs']
    num_bins_level = state[:-1]
    item_size = state[-1]
    bag_capacity = len(state)-1

    if item_size == bag_capacity:
        return 0 # new bag

    min_difference = bag_capacity 
    chosen_bin_index = 0 #default is new bag
    for i, bins in enumerate(num_bins_level):
        #skip new bag and levels for which bins don't exist
        if bins == 0 or i == 0:
            continue

        #if item fits perfectly into the bag
        elif (i + item_size) == bag_capacity:
            # assuming full bins have count 0
            if -bins < min_difference:
                min_difference = -bins
                chosen_bin_index = i
                return chosen_bin_index
            else:
                continue
        #item should fit in bag and should be at least of size 1
        elif (i + item_size) > bag_capacity:
            continue

        #sum of squares difference that chooses the bin 
        if num_bins_level[i + item_size] - bins < min_difference:
            chosen_bin_index = i 
            min_difference = num_bins_level[i + item_size] - bins 

    return chosen_bin_index

env_config = {
                "bag_capacity": 100,
                'item_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                #'item_probabilities': [0.14, 0.10, 0.06, 0.13, 0.11, 0.13, 0.03, 0.11, 0.19], #bounded waste
                'item_probabilities': [0.06, 0.11, 0.11, 0.22, 0, 0.11, 0.06, 0, 0.33], #perfect pack
                #'item_probabilities': [0, 0, 0, 1/3, 0, 0, 0, 0, 2/3], #linear waste
                'time_horizon': 10000,
            }

env = BinPackingActionMaskGymEnvironment(env_config)
state = env.reset()

done = False
total_reward = 0
while not done:
    action = get_action(state)
    state, reward, done, _ = env.step(action)
    total_reward += reward

print("Total reward for the sum of squares agent: ", total_reward)


