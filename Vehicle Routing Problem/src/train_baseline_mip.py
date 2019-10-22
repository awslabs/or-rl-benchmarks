import numpy as np
from vrp_environment import VRPGymEnvironment
from baseline_model import execute_plan
from baseline_model import get_or_solution
from baseline_model import whether_get_opt_sln

def log_metric(metric_name, value):
    print("Result for Baseline:\
        custom_metrics:\
        {}\
        \
        %s: %f"%(metric_name, value))

if __name__ == "__main__":
    env = VRPGymEnvironment()
    current_state = env.reset()
    done = False
    count = 0
    total_reward = 0
    total_reward_no_late_penalty = 0

    # from rl_operations_research_baselines.VRP.VRP_baseline_MIP import *

    current_o_status = env.o_status
    action_plan = [0]
    optimization_maxtime = 100  # seconds
    while not done:
        count += 1
        print("------- Time:", count)
        # action = env.action_space.sample()
        action, action_plan = execute_plan(env, action_plan)

        prev_o_status = [x for x in env.o_status]
        prev_o_xy = list(zip(env.o_x, env.o_y))
        ### STEP ###
        next_state, reward, done, info = env.step(action)
        ############
        if action_plan:
            action, action_plan = execute_plan(env, action_plan)  # Reevaluate the plan after the step.

        total_reward += reward
        total_reward_no_late_penalty += info['no_late_penalty_reward']
        print("Action: {0}, Reward: {1:.1f}, Done: {2}".format(action, reward, done))
        # if whether_get_opt_sln(env, current_o_status, next_o_status, action_plan):
        if whether_get_opt_sln(env, prev_o_status, prev_o_xy, action_plan):
            print("*** Asking for a new plan")
            try:
                action_plan = get_or_solution(env, optimization_maxtime)
            except:
                print("!!!!! COULD NOT FIND A SOLUTION, MOVING WITH THE PREVIOUS PLAN !!!!!")

        print("*** Total reward as of now:", np.round(total_reward))
        print("*** Total reward as of now (no penalty):", np.round(total_reward_no_late_penalty))

        if reward <= -50:
            break
    log_metric("episode_reward_mean", np.round(total_reward))
    log_metric("episode_reward_min", np.round(total_reward))
    log_metric("episode_reward_max", np.round(total_reward))