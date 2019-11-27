# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:56:51 2019

@author: maggiara
"""

import numpy as np
from scipy.stats import poisson

class multiperiod_vlt_heuristic():
    
    def __init__(self, parameters):       
        self.gamma = parameters.get('gamma', 0.99)
        self.p = parameters.get('p', 10.)
        self.c = parameters.get('c', 8.)
        self.h = parameters.get('h', 0.1) 
        self.k = parameters.get('k', 4.) 
        self.mu = parameters.get('mu', 20.) 
        self.lead_time = parameters.get('lead_time', 2)
        
        self.periods = parameters.get('periods', 20)
        self.x0 = parameters.get('x0', self.lead_time * [0])
        self.n_paths = parameters.get('n_paths', 1000)
    
    def solve(self):
        """
        Recursively solve the DP
        """
        b = self.p - self.gamma * self.c + self.k
        cr = b / (b + self.h)
        outl = poisson.ppf(cr, (self.lead_time + 1) * self.mu)
        value = self.simulate(outl)
        return value
    
    def outl(self):
        b = self.p - self.gamma * self.c + self.k
        cr = b / (b + self.h)
        outl = poisson.ppf(cr, (self.lead_time + 1) * self.mu)
        return outl
        
    def simulate(self, outl):
        value = 0
        for i in range(self.n_paths):
            value += self.single_path(outl)
        value /= self.n_paths
        return value
            
    def single_path(self, outl):
        x = self.x0.copy()
        value = 0
        for t in range(self.periods):
            
            demand = np.random.poisson(self.mu)
            buys = max(0, outl - np.sum(x))
            if self.lead_time == 0:
                x[0] += buys
            sales = min(x[0], demand)
            over = max(x[0] - demand, 0)
            under = max(demand - x[0], 0)
            sales_rev = sales * self.p
            buys_cost = self.gamma**self.lead_time * buys * self.c
            holding_cost = over * self.h
            penalty_lost_sale = under * self.k
            value += self.gamma**t * (sales_rev - buys_cost - holding_cost - penalty_lost_sale)
            if self.lead_time > 0:
                x[:-1] = x[1:]
                x[0] += over
                x[-1] = buys
#        value /= self.periods
        return value