# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:34:01 2019

@author: maggiara
"""

import numpy as np
from multiperiod_vlt_heuristic import multiperiod_vlt_heuristic

def baseline(l, N, write = False):
    
    max_value = 100
    max_holding_cost = 5
    max_loss_goodwill = 10
    max_mean = 200
    gamma = 1
    lead_time = l
    periods = 40
    n_paths = 500
    
    if write:
        filename = 'baseline_' + str(lead_time) + '.csv'
    
    def sample_parameters():
        p = np.random.rand() * max_value
        c = np.random.rand() * p
        h = np.random.rand() * min(c, max_holding_cost)
        k = np.random.rand() * max_loss_goodwill
        mu = np.random.rand() * max_mean
        return p,c,h,k,mu
        
    def sample_state(mu):
        inv_dim = max(lead_time, 1)
        state = np.zeros(inv_dim)
#        state[0] = np.random.rand() * inv_dim * mu
#        for i in range(1, inv_dim):
#            state[i] = np.random.rand() * (inv_dim * mu - np.sum(state[:i-1])) 
        return state
    
    
    if write:
        bufsize = 1
        file = open(filename, 'w', buffering=bufsize)
    V = 0
    for i in range(N):
        p,c,h,k,mu = sample_parameters()
        x = sample_state(mu)
        parameters = {}
        parameters['p'] = p
        parameters['c'] = c
        parameters['h'] = h
        parameters['k'] = k
        parameters['mu'] = mu
        parameters['x0'] = x
        parameters['lead_time'] = lead_time
        parameters['gamma'] = gamma
        parameters['periods'] = periods
        parameters['n_paths'] = n_paths
        
        M = multiperiod_vlt_heuristic(parameters)
        value = M.solve()
#        print(p,c,h,k,mu,x,value)
        V += value
        string = str(value) + ',' + str(p) + ',' + str(c) + ',' + str(h) + ',' + str(k) + ',' + str(mu)
        for j in range(len(x)):
            string += ',' + str(x[j])
        print(V / (i + 1))
        string += '\n'
        if write:
            file.write(string)
    if write:
        file.close()
        

if __name__ == "__main__":
    # l is lead time, N is the number of configurations to run
    # when write is True, an output file is generated 
    baseline(l=5, N=5000, write = False)   
