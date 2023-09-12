from MTL import MTLFormula
from WTS import WTS
import pulp as plp
import matplotlib.pyplot as plt
import numpy as np
import random
import time




def generate_plan_mtl_wts(Demands,wts,horizon):


    #BIG M Constant
    BIG_M = 10000000

    #TODO max value of mu minus
    #is also used to set (fictive) states b_s_t for all t in [-MAX_RIGHT_TIME,-1]
    MAX_RIGHT_TIME = 20
    
    dict_vars = {}
    demands_horizon = max([phi.horizon for phi, priority in Demands])
    global constraints
    constraints = 0

    encoding_start = time.time()
    
    """
        Optimization function
    """
    #to optimize the sum of the left time robustnesses weighted by priority of all the demands
    for phi, priority in Demands:
        mu_plus_phi_t = plp.LpVariable('muplus_'+str(id(phi))+'_t_'+str(0),cat='Integer')
        dict_vars['muplus_'+str(id(phi))+'_t_'+str(0)] = mu_plus_phi_t
    opt_model = plp.LpProblem("MIP Model", plp.LpMaximize)
    opt_model += plp.lpSum([dict_vars['muplus_'+str(id(phi))+'_t_'+str(0)]*priority for phi, priority in Demands])
    
    #alternative, to to optimize the sum of the "min(0, left time robustness)" weighted by priority of all the demands
    # for phi, priority in Demands:
        # X = plp.LpVariable('min_0_muplus_'+str(id(phi))+'_t_0',cat='Integer')
        # dict_vars['min_0_muplus_'+str(id(phi))+'_t_0'] = X
    # opt_model = plp.LpProblem("MIP Model", plp.LpMaximize)
    # opt_model += plp.lpSum([dict_vars['min_0_muplus_'+str(id(phi))+'_t_0']*priority for phi, priority in Demands])
    # for phi, priority in Demands:
        # opt_model += dict_vars['min_0_muplus_'+str(id(phi))+'_t_0'] <= 0
        # opt_model += dict_vars['min_0_muplus_'+str(id(phi))+'_t_0'] <= dict_vars['muplus_'+str(id(phi))+'_t_0']
    
    
    # opt_model = plp.LpProblem("MIP Model")

    
    
    
    
    """
    The WTS part
    """
    #Kurtz and Lin (2022). Introduce a binary variable bs(t) for each state s and each timestep t
    for state in wts.S:
        for t in range(0,horizon):
            b_s_t = plp.LpVariable('b_s_'+state+'_t_'+str(t),cat='Binary')
            dict_vars['b_s_'+state+'_t_'+str(t)] = b_s_t
    
    #also define negative time states, necessary for the computation of the right time robustness
    for state in wts.S:
        for t in range(-MAX_RIGHT_TIME,0):
            b_s_t = plp.LpVariable('b_s_'+state+'_t_'+str(t),cat='Binary')
            opt_model += b_s_t == 0
            constraints += 1
            dict_vars['b_s_'+state+'_t_'+str(t)] = b_s_t
    
    #Kurtz and Lin (2022). Establishes s_0 (5a)
    opt_model += dict_vars['b_s_'+wts.s_0+'_t_'+str(0)] == 1
    constraints += 1
    
    """
    The Kurtz and Lin (2022) encoding of the transition relation
    """
    # #Kurtz and Lin (2022). Ensures that only one state can be occupied at each timestep (5b)
    # for t in range(0,horizon): 
        # opt_model += plp.lpSum([dict_vars['b_s_'+state+'_t_'+str(t)] for state in wts.S]) == 1
    
    # #Kurtz and Lin (2022). Enforces transition relations (5c)
    # for t in range(0,horizon-1): 
        # for state in wts.S:
            # opt_model += plp.lpSum([dict_vars['b_s_'+s_prime+'_t_'+str(t+1)] for s_prime in wts.adjacent(state)]) >= dict_vars['b_s_'+state+'_t_'+str(t)]
    
    """
    Our encoding of the transition relation:
        - 0 or 1 state can be occupied at each time step.
        - transition relations depending on the weights, over time.
    """
    #Encodes that 0 or 1 state can be occupied at each timestep.
    for t in range(0,horizon): 
        opt_model += 0 <= plp.lpSum([dict_vars['b_s_'+state+'_t_'+str(t)] for state in wts.S]) <= 1
        constraints += 1
    
    #transition relations depending on the weights, over time.
    for t in range(0,horizon-1): 
        for state in wts.S:
        
            #add missing variables if necessary!
            adding_at_t = {}
            for s_prime in wts.adjacent(state):
                if t+wts.C[(state,t,s_prime)] > horizon-1:
                    try:
                        b_s_t = dict_vars['b_s_'+s_prime+'_t_'+str(t+wts.C[(state,t,s_prime)])]
                    except KeyError:
                        b_s_t = plp.LpVariable('b_s_'+s_prime+'_t_'+str(t+wts.C[(state,t,s_prime)]),cat='Binary')
                        dict_vars['b_s_'+s_prime+'_t_'+str(t+wts.C[(state,t,s_prime)])] = b_s_t
                        try:
                            adding_at_t[t+wts.C[(state,t,s_prime)]].append('b_s_'+s_prime+'_t_'+str(t+wts.C[(state,t,s_prime)]))
                        except KeyError:
                            adding_at_t[t+wts.C[(state,t,s_prime)]] = ['b_s_'+s_prime+'_t_'+str(t+wts.C[(state,t,s_prime)])]
            for add_t in adding_at_t:
                opt_model += 0 <= plp.lpSum([dict_vars[var] for var in adding_at_t[add_t]]) <= 1
                constraints += 1
            
            opt_model += plp.lpSum([dict_vars['b_s_'+s_prime+'_t_'+str(t+wts.C[(state,t,s_prime)])] for s_prime in wts.adjacent(state)]) >= dict_vars['b_s_'+state+'_t_'+str(t)]
            constraints += 1
            
            #to allow only one of the possible transitions to be taken!
            opt_model += plp.lpSum([dict_vars['b_s_'+s_prime+'_t_'+str(t+wts.C[(state,t,s_prime)])] for s_prime in wts.adjacent(state)]) <= 1
            constraints += 1
            
    
    #Recursive function to model the satisfaction of an MTL formula. 
    #Kurtz and Lin (2022), Eq. (6,7,8a,8b,8c,8d)
    def model_phi(phi,t,opt_model,dict_vars):
        global constraints
        #defines the variables and constraints for an MTL predicate
        if isinstance(phi, MTLFormula.Predicate):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == plp.lpSum([dict_vars['b_s_'+state+'_t_'+str(t)] for state in wts.S if wts.label_of_state(state)==phi.predicate])
            constraints += 1
            
        elif isinstance(phi, MTLFormula.Negation):
            model_phi(phi.formula,t,opt_model,dict_vars)
            zvar1 = dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t)]
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1-zvar1
            constraints += 1
            
        elif isinstance(phi, MTLFormula.Disjunction):
            #define the variable for the conjunction
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            #define the constraints for the conjunction
            for phi_i in phi.list_formulas:
                #retrieve the model of the formulas composing the disjunction
                model_phi(phi_i,t,opt_model,dict_vars)
                #enforcing the constraints on them
                opt_model += zvar >= dict_vars['z1_'+str(id(phi_i))+'_t_'+str(t)]
                constraints += 1
            opt_model += zvar <= plp.lpSum([dict_vars['z1_'+str(id(phi_i))+'_t_'+str(t)] for phi_i in phi.list_formulas])
            constraints += 1
            
        elif isinstance(phi, MTLFormula.Conjunction):
            #define the variable for the conjunction
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            #define the constraints for the conjunction
            for phi_i in phi.list_formulas:
                #retrieve the model of the formulas composing the disjunction
                model_phi(phi_i,t,opt_model,dict_vars)
                #enforcing the constraints on them
                opt_model += zvar <= dict_vars['z1_'+str(id(phi_i))+'_t_'+str(t)]
                constraints += 1
            opt_model += zvar >= 1-len(phi.list_formulas)+ plp.lpSum([dict_vars['z1_'+str(id(phi_i))+'_t_'+str(t)] for phi_i in phi.list_formulas])
            constraints += 1
                
        elif isinstance(phi, MTLFormula.Always):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(t+phi.t1,t+phi.t2+1):
                model_phi(phi.formula,t_i,opt_model,dict_vars)
                opt_model += zvar <= dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)]
                constraints += 1
            opt_model += zvar >= 1 - (phi.t2+1-phi.t1) + plp.lpSum([dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(t+phi.t1,t+phi.t2+1)])
            constraints += 1
            
        elif isinstance(phi, MTLFormula.Eventually):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(t+phi.t1,t+phi.t2+1):
                model_phi(phi.formula,t_i,opt_model,dict_vars)
                opt_model += zvar >= dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)]
                constraints += 1
            opt_model += zvar <= plp.lpSum([dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(t+phi.t1,t+phi.t2+1)])
            constraints += 1
        
        elif isinstance(phi, MTLFormula.Until):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            
            for t_prime in range(t+phi.t1,t+phi.t2+1):
                try:
                    zvar_disj = dict_vars['z1disj_'+str(id(phi))+'_t_'+str(t_prime)]
                except KeyError:
                    zvar_disj = plp.LpVariable('z1disj_'+str(id(phi))+'_t_'+str(t_prime),cat='Binary')
                    dict_vars['z1disj_'+str(id(phi))+'_t_'+str(t_prime)] = zvar_disj
                model_phi(phi.second_formula,t_prime,opt_model,dict_vars)
                opt_model += zvar_disj <= dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t_prime)]
                constraints += 1
                for t_prime_prime in range(t,t_prime):
                    model_phi(phi.first_formula,t_prime_prime,opt_model,dict_vars)
                    opt_model += zvar_disj <= dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t_prime_prime)]
                    constraints += 1
                opt_model += zvar_disj >= 1 - (t_prime-t) + plp.lpSum([dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t_prime_prime)] for t_prime_prime in range(t,t_prime)]) + dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t_prime)]
                opt_model += zvar >= zvar_disj
                constraints += 2
            opt_model += zvar <= plp.lpSum([dict_vars['z1disj_'+str(id(phi))+'_t_'+str(t_prime)] for t_prime in range(t+phi.t1,t+phi.t2+1)])  
            constraints += 1  
    
    
    #Recursive call of model_phi   
    for phi, priority in Demands:
        for i in range(0,horizon-demands_horizon):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(i)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(i),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(i)] = zvar
            model_phi(phi,i,opt_model,dict_vars)
    
        # opt_model += dict_vars['z1_'+str(id(phi))+'_t_0'] == 1

    
        #Definition of the left temporal robustness, Rodionova (2022), Eq. 24 to 28.
        c_1_phi_Hplus1 = plp.LpVariable('c_1_'+str(id(phi))+'_t_'+str(horizon-demands_horizon),cat='Integer')
        dict_vars['c_1_'+str(id(phi))+'_t_'+str(horizon-demands_horizon)] = c_1_phi_Hplus1
        opt_model += dict_vars['c_1_'+str(id(phi))+'_t_'+str(horizon-demands_horizon)] == 0
        constraints += 1
            
        c_0_phi_Hplus1 = plp.LpVariable('c_0_'+str(id(phi))+'_t_'+str(horizon-demands_horizon),cat='Integer')
        dict_vars['c_0_'+str(id(phi))+'_t_'+str(horizon-demands_horizon)] = c_0_phi_Hplus1
        opt_model += dict_vars['c_0_'+str(id(phi))+'_t_'+str(horizon-demands_horizon)] == 0
        constraints += 1
        
        for t in reversed(range(horizon-demands_horizon)):
            c_1_phi_t = plp.LpVariable('c_1_'+str(id(phi))+'_t_'+str(t),cat='Integer')
            dict_vars['c_1_'+str(id(phi))+'_t_'+str(t)] = c_1_phi_t
            #if-then-else product construct of Rodionova (2022), Eq. 24 and 25
            x_l, x_u = -(horizon-demands_horizon), horizon-demands_horizon
            opt_model += x_l*dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] <= c_1_phi_t <= x_u*dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            opt_model += (dict_vars['c_1_'+str(id(phi))+'_t_'+str(t+1)] + 1) - (x_u * (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])) <= c_1_phi_t <= (dict_vars['c_1_'+str(id(phi))+'_t_'+str(t+1)] + 1) - (x_l * (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]))
            constraints += 2
            
            c_1_phi_t_tilde = plp.LpVariable('c_1_'+str(id(phi))+'_t_'+str(t)+'_tilde',cat='Integer')
            dict_vars['c_1_'+str(id(phi))+'_t_'+str(t)+'_tilde'] = c_1_phi_t_tilde
            opt_model += c_1_phi_t_tilde == c_1_phi_t - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            constraints += 1
            
            c_0_phi_t = plp.LpVariable('c_0_'+str(id(phi))+'_t_'+str(t),cat='Integer')
            dict_vars['c_0_'+str(id(phi))+'_t_'+str(t)] = c_0_phi_t
            #if-then-else product construct of Rodionova (2022), Eq. 24 and 25
            x_l, x_u = -(horizon-demands_horizon), horizon-demands_horizon
            opt_model += x_l*(1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]) <= c_0_phi_t <= x_u*(1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])
            opt_model += (dict_vars['c_0_'+str(id(phi))+'_t_'+str(t+1)] - 1) - (x_u * (1 - (1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]))) <= c_0_phi_t <= (dict_vars['c_0_'+str(id(phi))+'_t_'+str(t+1)] - 1) - (x_l * (1 - (1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])))
            constraints += 2
            
            c_0_phi_t_tilde = plp.LpVariable('c_0_'+str(id(phi))+'_t_'+str(t)+'_tilde',cat='Integer')
            dict_vars['c_0_'+str(id(phi))+'_t_'+str(t)+'_tilde'] = c_0_phi_t_tilde
            opt_model += c_0_phi_t_tilde == c_0_phi_t + (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])
            constraints += 1
            
            if t==0:
                #already exits
                mu_plus_phi_t = dict_vars['muplus_'+str(id(phi))+'_t_'+str(t)]
            else:
                mu_plus_phi_t = plp.LpVariable('muplus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
                dict_vars['muplus_'+str(id(phi))+'_t_'+str(t)] = mu_plus_phi_t
            opt_model += mu_plus_phi_t == c_0_phi_t_tilde + c_1_phi_t_tilde
            constraints += 1
    
    
    
        #Definition of the right temporal robustness, inspired by Rodionova (2022), Eq. 24 to 28.
        
        #define satisfaction for negative times
        for i in range(-MAX_RIGHT_TIME,0):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(i)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(i),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(i)] = zvar
            model_phi(phi,i,opt_model,dict_vars)
        
        c_1_minus_phi_min_h = plp.LpVariable('c_1_minus_'+str(id(phi))+'_t_'+str(-MAX_RIGHT_TIME-1),cat='Integer')
        dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(-MAX_RIGHT_TIME-1)] = c_1_minus_phi_min_h
        opt_model += dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(-MAX_RIGHT_TIME-1)] == 0
        constraints += 1
            
        c_0_minus_phi_min1 = plp.LpVariable('c_0_minus_'+str(id(phi))+'_t_'+str(-MAX_RIGHT_TIME-1),cat='Integer')
        dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(-MAX_RIGHT_TIME-1)] = c_0_minus_phi_min1
        opt_model += dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(-MAX_RIGHT_TIME-1)] == 0
        constraints += 1

        for t in range(-MAX_RIGHT_TIME,1):
            c_1_minus_phi_t = plp.LpVariable('c_1_minus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
            dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t)] = c_1_minus_phi_t
            #if-then-else product construct of Rodionova (2022), Eq. 24 and 25
            x_l, x_u = -MAX_RIGHT_TIME-2, MAX_RIGHT_TIME+2
            opt_model += x_l*dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] <= c_1_minus_phi_t <= x_u*dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            opt_model += (dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t-1)] + 1) - (x_u * (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])) <= c_1_minus_phi_t <= (dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t-1)] + 1) - (x_l * (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]))
            constraints += 2
            
            c_1_minus_phi_t_tilde = plp.LpVariable('c_1_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde',cat='Integer')
            dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde'] = c_1_minus_phi_t_tilde
            opt_model += c_1_minus_phi_t_tilde == c_1_minus_phi_t - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            constraints += 1
            
            c_0_minus_phi_t = plp.LpVariable('c_0_minus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
            dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t)] = c_0_minus_phi_t
            #if-then-else product construct of Rodionova (2022), Eq. 24 and 25
            x_l, x_u = -MAX_RIGHT_TIME-2, MAX_RIGHT_TIME+2
            opt_model += x_l*(1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]) <= c_0_minus_phi_t <= x_u*(1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])
            opt_model += (dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t-1)] - 1) - (x_u * (1 - (1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]))) <= c_0_minus_phi_t <= (dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t-1)] - 1) - (x_l * (1 - (1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])))
            constraints += 2
            
            c_0_minus_phi_t_tilde = plp.LpVariable('c_0_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde',cat='Integer')
            dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde'] = c_0_minus_phi_t_tilde
            opt_model += c_0_minus_phi_t_tilde == c_0_minus_phi_t + (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])
            constraints += 2
            

            mu_minus_phi_t = plp.LpVariable('muminus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
            dict_vars['muminus_'+str(id(phi))+'_t_'+str(t)] = mu_minus_phi_t
            opt_model += mu_minus_phi_t == c_0_minus_phi_t_tilde + c_1_minus_phi_t_tilde
            constraints += 1
    
    
    
    encoding_time = time.time()-encoding_start
    
    
    print("Encoding time",encoding_time)
    print("LP variables",len(dict_vars))
    print("LP constrain",constraints)
    
    
    solve_start = time.time()
    opt_model.solve(plp.GUROBI_CMD(msg=True))
    solve_time = time.time()-solve_start
    
    
    
    
    for phi, priority in Demands:
        print(phi, priority)
        print("muplus", dict_vars['muplus_'+str(id(phi))+'_t_0'].varValue, "muminus", dict_vars['muminus_'+str(id(phi))+'_t_0'].varValue)
        print()

        print("muplus", [dict_vars['muplus_'+str(id(phi))+'_t_'+str(t)].varValue for t in range(0,horizon-demands_horizon)])        
        print("z_phi", [dict_vars['z1_'+str(id(phi))+'_t_'+str(t)].varValue for t in range(0,horizon-demands_horizon)])
        print()
        print("muminus", [dict_vars['muminus_'+str(id(phi))+'_t_'+str(t)].varValue for t in range(-MAX_RIGHT_TIME,1)])
        print("z_phi_negs", [dict_vars['z1_'+str(id(phi))+'_t_'+str(t)].varValue for t in range(-MAX_RIGHT_TIME,1)])
        print()
        print()
        print()
    
    # print("\n\n TRAJECTORY")
    # for t in range(0,horizon):
        # print(t, [state for state in wts.S if dict_vars['b_s_'+state+'_t_'+str(t)].varValue == 1.0])
    
    
    print("Encoding time",encoding_time)
    print("Solving time",solve_time)
    print("LP variables",len(dict_vars))
    print("LP constrain",constraints)
    print()
    
    return [[state for state in wts.S if dict_vars['b_s_'+state+'_t_'+str(t)].varValue == 1.0] for t in range(0,horizon)]


if __name__ == '__main__':

    #planning horizon
    PLANNING_HORIZON = 980
    # PLANNING_HORIZON = 50

    #The WTS of LV24, floor 4
    S = ['s_0_0','s_1_0','s_2_0','s_3_0','s_4_0','s_5_0','s_6_0','s_7_0','s_8_0','s_0_1','s_1_1','s_2_1','s_3_1','s_4_1','s_5_1','s_6_1','s_7_1','s_8_1','s_9_1','s_0_2','s_1_2','s_3_20','s_3_21','s_5_2','s_8_2','s_9_2','s_0_3','s_1_3','s_2_3','s_3_3','s_4_3','s_5_3','s_6_3','s_7_3','s_8_3','s_9_3','s_0_4','s_1_4','s_2_4','s_3_4','s_4_4','s_5_4','s_6_4','s_7_4','s_8_4','s_9_4']
    s_0 = 's_0_3'
    T = [('s_0_0','s_0_1'),('s_1_0','s_1_1'),('s_2_0','s_2_1'),('s_3_0','s_3_1'),('s_4_0','s_4_1'),('s_5_0','s_5_1'),('s_6_0','s_6_1'),('s_7_0','s_7_1'),('s_8_0','s_8_1'),('s_0_1','s_1_1'),('s_1_1','s_2_1'),('s_2_1','s_3_1'),('s_3_1','s_4_1'),('s_4_1','s_5_1'),('s_5_1','s_6_1'),('s_6_1','s_7_1'),('s_7_1','s_8_1'),('s_8_1','s_9_1'),('s_1_1','s_1_2'),('s_0_2','s_1_2'),('s_1_2','s_1_3'),('s_3_1','s_3_20'),('s_3_21','s_3_3'),('s_5_1','s_5_2'),('s_5_2','s_5_3'),('s_8_1','s_8_2'),('s_8_2','s_9_2'),('s_8_2','s_8_3'),('s_0_3','s_1_3'),('s_1_3','s_2_3'),('s_2_3','s_3_3'),('s_3_3','s_4_3'),('s_4_3','s_5_3'),('s_5_3','s_6_3'),('s_6_3','s_7_3'),('s_7_3','s_8_3'),('s_0_3','s_1_3'),('s_8_3','s_9_3'),('s_0_3','s_0_4'),('s_1_3','s_1_4'),('s_2_3','s_2_4'),('s_3_3','s_3_4'),('s_4_3','s_4_4'),('s_5_3','s_5_4'),('s_6_3','s_6_4'),('s_7_3','s_7_4'),('s_8_3','s_8_4'),('s_9_3','s_9_4')]
    for state in S:
        T.append((state,state))
    AP = ['E1','E2','K','P','M1','M2','M1','Jana','Alexis']
    
    C = {}
    for t in range(0,PLANNING_HORIZON):
        for s1, s2 in T:
            C[(s1,t,s2)] = 1
            C[(s2,t,s1)] = 1
            C[(s1,t,s1)] = 1
        C[('s_8_1',t,'s_8_2')] = 2
        C[('s_8_2',t,'s_8_3')] = 2
        C[('s_5_1',t,'s_5_2')] = 3
        C[('s_5_2',t,'s_5_3')] = 3
        C[('s_8_2',t,'s_8_1')] = 2
        C[('s_8_3',t,'s_8_2')] = 2
        C[('s_5_2',t,'s_5_1')] = 3
        C[('s_5_3',t,'s_5_2')] = 3
        C[('s_3_21',t,'s_3_3')] = 3
        C[('s_3_3',t,'s_3_21')] = 3
        C[('s_3_20',t,'s_3_1')] = 3
        C[('s_3_1',t,'s_3_20')] = 3

    # for c in C:
        # print(c, C[c])
    
    L = {'s_3_20':'E1','s_3_21':'E2','s_9_2':'K','s_5_2':'P','s_0_2':'M1','s_3_4':'M2','s_9_1':'Jana','s_6_0':'Alexis','s_0_3':'Base'}
    wts = WTS(S,s_0,T,AP,L,C)
    
    

    #Eventually_[4,7] x=1,y=8
    e1 = MTLFormula.Eventually(MTLFormula.Predicate('K'),6,12)
    e2 = MTLFormula.Eventually(MTLFormula.Predicate('M2'),15,18)
    a1 = MTLFormula.Always(MTLFormula.Predicate('E2'),30,35)
    e3 = MTLFormula.Eventually(MTLFormula.Predicate('M2'),43,45)
    e4 = MTLFormula.Eventually(MTLFormula.Predicate('Jana'),610,650)
    a2 = MTLFormula.Always(MTLFormula.Predicate('Alexis'),950,960)
    # u1 = MTLFormula.Until(MTLFormula.Predicate('Base'),MTLFormula.Predicate('K'),0,30)
    
    # phi = MTLFormula.Conjunction([e1,e2])
    # phi = MTLFormula.Conjunction([a1])


    # Demands = [(e1,2),(e2,2),(a1,3)]
    # Demands = [(e1,2),(e2,2),(a1,3),(e3,1)]
    Demands = [(e1,1),(e2,1),(a1,1),(e4,3),(a2,1)]
    # Demands = [(e1,1),(e2,1)]
    
    # Demands = [(u1,1)]

    trajectory = generate_plan_mtl_wts(Demands,wts,PLANNING_HORIZON)
    print(trajectory)

    
    # traj_r1 = [[t[0],t[1]] for t in trajectory]
    # traj_r2 = [[t[2],t[3]] for t in trajectory]
    # traj_r3 = [[t[4],t[5]] for t in trajectory]
    
    # print()
    # print(len(traj_r1))
    # print()
    
    # print(traj_r1)
    # print(traj_r2)
    # print(traj_r3)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # fig.tight_layout()
    
    # plt.gcf().canvas.mpl_connect('key_release_event',
                                 # lambda event: [exit(0) if event.key == 'escape' else None])
    # plt.axis([rand_area[0]-0.2, rand_area[1]+0.2, rand_area[0]-0.2, rand_area[1]+0.2])
    # plt.grid(True)

    # ax.plot([x for (x, y) in traj_r1], [y for (x, y) in traj_r1], '-g', marker='o', label=r'r1')
    # ax.plot([x for (x, y) in traj_r2], [y for (x, y) in traj_r2], '-b', marker='o', label=r'r2')
    # ax.plot([x for (x, y) in traj_r3], [y for (x, y) in traj_r3], '-r', marker='o', label=r'r3')
    # plt.grid(True)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=3)
    # plt.show()

    #Plot


