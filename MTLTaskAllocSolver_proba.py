from MTL import MTLFormula
from WTSProba import WTSProba
import pulp as plp
import matplotlib.pyplot as plt
import numpy as np
import random




def generate_plan_mtl_wts(Demands,wts,horizon):


    #BIG M Constant
    BIG_M = 10000000

    
    dict_vars = {}
    demands_horizon = max([phi.horizon for phi, priority in Demands])
    
    for phi, priority in Demands:
        mu_plus_phi_t = plp.LpVariable('muplus_'+str(id(phi))+'_t_'+str(0),cat='Integer')
        dict_vars['muplus_'+str(id(phi))+'_t_'+str(0)] = mu_plus_phi_t
    
    #to optimize the sum of the time robustnesses weighted by priority of all the demands
    opt_model = plp.LpProblem("MIP Model", plp.LpMaximize)
    opt_model += plp.lpSum([dict_vars['muplus_'+str(id(phi))+'_t_'+str(0)]*priority for phi, priority in Demands])
    
    # opt_model = plp.LpProblem("MIP Model")

    
    
    """
    The WTS part
    """
    #Kurtz and Lin (2022). Introduce a binary variable bs(t) for each state s and each timestep t
    for state in wts.S:
        for t in range(0,horizon):
            b_s_t = plp.LpVariable('b_s_'+state+'_t_'+str(t),cat='Binary')
            dict_vars['b_s_'+state+'_t_'+str(t)] = b_s_t
    
    #Kurtz and Lin (2022). Establishes s_0 (5a)
    opt_model += dict_vars['b_s_'+wts.s_0+'_t_'+str(0)] == 1
    
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
    
    #transition relations depending on the weights, over time.
    for t in range(0,horizon-1): 
        for state in wts.C:
        
            #add missing variables if necessary!
            adding_at_t = {}
            for (t_t,t_prime,s_prime) in wts.C[state]:
                if (t_t == t) and (t+t_prime > horizon-1):
                    try:
                        b_s_t = dict_vars['b_s_'+s_prime+'_t_'+str(t+t_prime)]
                    except KeyError:
                        b_s_t = plp.LpVariable('b_s_'+s_prime+'_t_'+str(t+t_prime),cat='Binary')
                        dict_vars['b_s_'+s_prime+'_t_'+str(t+t_prime)] = b_s_t
                        try:
                            adding_at_t[t+t_prime].append('b_s_'+s_prime+'_t_'+str(t+t_prime))
                        except KeyError:
                            adding_at_t[t+t_prime] = ['b_s_'+s_prime+'_t_'+str(t+t_prime)]
            
            for add_t in adding_at_t:
                opt_model += 0 <= plp.lpSum([dict_vars[var] for var in adding_at_t[add_t]]) <= 1
            
            #to allow only one of the possible transitions to be taken!
            opt_model += plp.lpSum([dict_vars['b_s_'+s_prime+'_t_'+str(t+t_prime)] for (t_t,t_prime,s_prime) in wts.C[state] if t_t == t]) <= 1
            opt_model += plp.lpSum([dict_vars['b_s_'+s_prime+'_t_'+str(t+t_prime)] for (t_t,t_prime,s_prime) in wts.C[state] if t_t == t]) >= dict_vars['b_s_'+state+'_t_'+str(t)]
            
            
            
            
    
    """
        Encoding of the probabilities
    """
    
    
    #To parse the wts
    dict_pred_s_t = {}

    #Introduce a continuous variable ps(t) for each state s and each timestep t
    for state in wts.S:
        for t in range(0,horizon):
            p_s_t = plp.LpVariable('p_s_'+state+'_t_'+str(t),cat='Continuous')
            dict_vars['p_s_'+state+'_t_'+str(t)] = p_s_t
            dict_pred_s_t['p_s_'+state+'_t_'+str(t)] = {}
    
    
    #Parse the wts
    def parse_pred_s_t(state,t,dict_pred_s_t):
        for (t_t,t_prime,s_prime) in wts.C[state]:
            if t==t_t:
                try:
                    dict_pred_s_t['p_s_'+s_prime+'_t_'+str(t_t+t_prime)]['p_s_'+state+'_t_'+str(t)] = None
                except KeyError:
                    dict_pred_s_t['p_s_'+s_prime+'_t_'+str(t_t+t_prime)] = {}
                    dict_pred_s_t['p_s_'+s_prime+'_t_'+str(t_t+t_prime)]['p_s_'+state+'_t_'+str(t)] = None
                parse_pred_s_t(s_prime,t_t+t_prime,dict_pred_s_t)
    
    #start with s_0 at t_0
    parse_pred_s_t(wts.s_0,0,dict_pred_s_t)
    
    for state in dict_pred_s_t:
        print(state,list(dict_pred_s_t[state]))
    
    
    
    #Establishes p_(s_0)(0) to be 1
    opt_model += dict_vars['p_s_'+wts.s_0+'_t_'+str(0)] == 1.0
    for state in list(set(wts.S)-set([wts.s_0])):
        opt_model += dict_vars['p_s_'+state+'_t_'+str(0)] == 0.0
        
    
    #Establishes that p_s(t) = 0 if b_s(t) = 0 
    # for state in wts.S:
        # for t in range(0,horizon):
            # opt_model += dict_vars['p_s_'+state+'_t_'+str(t)] <= dict_vars['b_s_'+state+'_t_'+str(t)]
    
    #Establishes that p_s'(t+t') = p_s(t) * C(s,t,t',s') * (1/|Adj(s)|)
    # for s in wts.C:
        # for (t,t_prime,s_prime) in wts.C[s]:
            # try:
                # # opt_model += dict_vars['p_s_'+s_prime+'_t_'+str(t+t_prime)] == dict_vars['p_s_'+s+'_t_'+str(t)] * (1/len(wts.adjacent(s))) * wts.C[s][(t,t_prime,s_prime)]
                # opt_model += p_s_t >= dict_vars['p_s_'+s+'_t_'+str(t)] * (1/len(wts.adjacent(s))) * wts.C[s][(t,t_prime,s_prime)] - (1-dict_vars['b_s_'+s+'_t_'+str(t)])*BIG_M
                # opt_model += p_s_t <= dict_vars['p_s_'+s+'_t_'+str(t)] * (1/len(wts.adjacent(s))) * wts.C[s][(t,t_prime,s_prime)] + (1-dict_vars['b_s_'+s+'_t_'+str(t)])*BIG_M
            # except KeyError:
                # p_s_t = plp.LpVariable('p_s_'+s_prime+'_t_'+str(t+t_prime),cat='Continuous')
                # dict_vars['p_s_'+s_prime+'_t_'+str(t+t_prime)] = p_s_t
                # opt_model += p_s_t >= dict_vars['p_s_'+s+'_t_'+str(t)] * (1/len(wts.adjacent(s))) * wts.C[s][(t,t_prime,s_prime)] - (1-dict_vars['b_s_'+s+'_t_'+str(t)])*BIG_M
                # opt_model += p_s_t <= dict_vars['p_s_'+s+'_t_'+str(t)] * (1/len(wts.adjacent(s))) * wts.C[s][(t,t_prime,s_prime)] + (1-dict_vars['b_s_'+s+'_t_'+str(t)])*BIG_M
    
    # opt_model += dict_vars['p_s_0_1_t_2'] == dict_vars['p_s_'+s+'_t_'+str(t)] * (1/len(wts.adjacent(s))) * wts.C[s][(t,t_prime,s_prime)]
    
    # opt_model += dict_vars['p_s_s_0_1_t_2'] == dict_vars['p_s_s_0_0_t_0'] * (1/len(wts.adjacent('s_0_0'))) * wts.C['s_0_0'][(0,2,'s_0_1')]
    # opt_model += dict_vars['p_s_s_1_0_t_2'] == dict_vars['p_s_s_0_0_t_0'] * (1/len(wts.adjacent('s_0_0'))) * wts.C['s_0_0'][(0,2,'s_1_0')]
    
    #these two possibilities
    # opt_model += dict_vars['p_s_s_1_1_t_3'] == dict_vars['p_s_s_0_1_t_2'] * (1/len(wts.adjacent('s_0_1'))) * wts.C['s_0_1'][(2,1,'s_1_1')]
    # opt_model += dict_vars['p_s_s_1_1_t_3'] == dict_vars['p_s_s_1_0_t_2'] * (1/len(wts.adjacent('s_1_0'))) * wts.C['s_1_0'][(2,1,'s_1_1')]
    #in one
    # opt_model += dict_vars['p_s_s_1_1_t_3'] >= dict_vars['p_s_s_0_1_t_2'] * (1/len(wts.adjacent('s_0_1'))) * wts.C['s_0_1'][(2,1,'s_1_1')] - (1-dict_vars['b_s_s_0_1_t_2'])*1
    # opt_model += dict_vars['p_s_s_1_1_t_3'] <= dict_vars['p_s_s_0_1_t_2'] * (1/len(wts.adjacent('s_0_1'))) * wts.C['s_0_1'][(2,1,'s_1_1')] + (1-dict_vars['b_s_s_0_1_t_2'])*1
    
    # opt_model += dict_vars['p_s_s_1_1_t_3'] >= dict_vars['p_s_s_1_0_t_2'] * (1/len(wts.adjacent('s_1_0'))) * wts.C['s_1_0'][(2,1,'s_1_1')] - (1-dict_vars['b_s_s_1_0_t_2'])*1
    # opt_model += dict_vars['p_s_s_1_1_t_3'] <= dict_vars['p_s_s_1_0_t_2'] * (1/len(wts.adjacent('s_1_0'))) * wts.C['s_1_0'][(2,1,'s_1_1')] + (1-dict_vars['b_s_s_1_0_t_2'])*1
    
    for s in wts.C:
        for (t,t_prime,s_prime) in wts.C[s]:
            try:
                opt_model += dict_vars['p_s_'+s_prime+'_t_'+str(t+t_prime)] >= dict_vars['p_s_'+s+'_t_'+str(t)] * (1/len(wts.adjacent(s))) * wts.C[s][(t,t_prime,s_prime)] - (1-dict_vars['b_s_'+s+'_t_'+str(t)])*1
                opt_model += dict_vars['p_s_'+s_prime+'_t_'+str(t+t_prime)] <= dict_vars['p_s_'+s+'_t_'+str(t)] * (1/len(wts.adjacent(s))) * wts.C[s][(t,t_prime,s_prime)] + (1-dict_vars['b_s_'+s+'_t_'+str(t)])*1
            except KeyError:
                p_s_t = plp.LpVariable('p_s_'+s_prime+'_t_'+str(t+t_prime),cat='Continuous')
                dict_vars['p_s_'+s_prime+'_t_'+str(t+t_prime)] = p_s_t
                opt_model += dict_vars['p_s_'+s_prime+'_t_'+str(t+t_prime)] >= dict_vars['p_s_'+s+'_t_'+str(t)] * (1/len(wts.adjacent(s))) * wts.C[s][(t,t_prime,s_prime)] - (1-dict_vars['b_s_'+s+'_t_'+str(t)])*1
                opt_model += dict_vars['p_s_'+s_prime+'_t_'+str(t+t_prime)] <= dict_vars['p_s_'+s+'_t_'+str(t)] * (1/len(wts.adjacent(s))) * wts.C[s][(t,t_prime,s_prime)] + (1-dict_vars['b_s_'+s+'_t_'+str(t)])*1
    
    
    
    """
        Encoding the MITL formula
    """
    #Recursive function to model the satisfaction of an MTL formula. 
    #Kurtz and Lin (2022), Eq. (6,7,8a,8b,8c,8d)
    def model_phi(phi,t,opt_model,dict_vars):
    
        #defines the variables and constraints for an MTL predicate
        if isinstance(phi, MTLFormula.Predicate):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == plp.lpSum([dict_vars['b_s_'+state+'_t_'+str(t)] for state in wts.S if wts.label_of_state(state)==phi.predicate])
            
        elif isinstance(phi, MTLFormula.Negation):
            model_phi(phi.formula,t,opt_model,dict_vars)
            zvar1 = dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t)]
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1-zvar1
            
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
            opt_model += zvar <= plp.lpSum([dict_vars['z1_'+str(id(phi_i))+'_t_'+str(t)] for phi_i in phi.list_formulas])
            
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
            opt_model += zvar >= 1-len(phi.list_formulas)+ plp.lpSum([dict_vars['z1_'+str(id(phi_i))+'_t_'+str(t)] for phi_i in phi.list_formulas])
                
        elif isinstance(phi, MTLFormula.Always):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(t+phi.t1,t+phi.t2+1):
                model_phi(phi.formula,t_i,opt_model,dict_vars)
                opt_model += zvar <= dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)]
            opt_model += zvar >= 1 - (phi.t2+1-phi.t1) + plp.lpSum([dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(t+phi.t1,t+phi.t2+1)])
            
        elif isinstance(phi, MTLFormula.Eventually):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(t+phi.t1,t+phi.t2+1):
                model_phi(phi.formula,t_i,opt_model,dict_vars)
                opt_model += zvar >= dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)]
            opt_model += zvar <= plp.lpSum([dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(t+phi.t1,t+phi.t2+1)])
        
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
                for t_prime_prime in range(t,t_prime):
                    model_phi(phi.first_formula,t_prime_prime,opt_model,dict_vars)
                    opt_model += zvar_disj <= dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t_prime_prime)]
                opt_model += zvar_disj >= 1 - (t_prime-t) + plp.lpSum([dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t_prime_prime)] for t_prime_prime in range(t,t_prime)]) + dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t_prime)]
                opt_model += zvar >= zvar_disj
            opt_model += zvar <= plp.lpSum([dict_vars['z1disj_'+str(id(phi))+'_t_'+str(t_prime)] for t_prime in range(t+phi.t1,t+phi.t2+1)])    
    
    
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
            
        c_0_phi_Hplus1 = plp.LpVariable('c_0_'+str(id(phi))+'_t_'+str(horizon-demands_horizon),cat='Integer')
        dict_vars['c_0_'+str(id(phi))+'_t_'+str(horizon-demands_horizon)] = c_0_phi_Hplus1
        opt_model += dict_vars['c_0_'+str(id(phi))+'_t_'+str(horizon-demands_horizon)] == 0
        
        for t in reversed(range(horizon-demands_horizon)):
            c_1_phi_t = plp.LpVariable('c_1_'+str(id(phi))+'_t_'+str(t),cat='Integer')
            dict_vars['c_1_'+str(id(phi))+'_t_'+str(t)] = c_1_phi_t
            #if-then-else product construct of Rodionova (2022), Eq. 24 and 25
            x_l, x_u = -(horizon-demands_horizon), horizon-demands_horizon
            opt_model += x_l*dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] <= c_1_phi_t <= x_u*dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            opt_model += (dict_vars['c_1_'+str(id(phi))+'_t_'+str(t+1)] + 1) - (x_u * (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])) <= c_1_phi_t <= (dict_vars['c_1_'+str(id(phi))+'_t_'+str(t+1)] + 1) - (x_l * (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]))
            
            c_1_phi_t_tilde = plp.LpVariable('c_1_'+str(id(phi))+'_t_'+str(t)+'_tilde',cat='Integer')
            dict_vars['c_1_'+str(id(phi))+'_t_'+str(t)+'_tilde'] = c_1_phi_t_tilde
            opt_model += c_1_phi_t_tilde == c_1_phi_t - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            
            c_0_phi_t = plp.LpVariable('c_0_'+str(id(phi))+'_t_'+str(t),cat='Integer')
            dict_vars['c_0_'+str(id(phi))+'_t_'+str(t)] = c_0_phi_t
            #if-then-else product construct of Rodionova (2022), Eq. 24 and 25
            x_l, x_u = -(horizon-demands_horizon), horizon-demands_horizon
            opt_model += x_l*(1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]) <= c_0_phi_t <= x_u*(1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])
            opt_model += (dict_vars['c_0_'+str(id(phi))+'_t_'+str(t+1)] - 1) - (x_u * (1 - (1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]))) <= c_0_phi_t <= (dict_vars['c_0_'+str(id(phi))+'_t_'+str(t+1)] - 1) - (x_l * (1 - (1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])))
            
            c_0_phi_t_tilde = plp.LpVariable('c_0_'+str(id(phi))+'_t_'+str(t)+'_tilde',cat='Integer')
            dict_vars['c_0_'+str(id(phi))+'_t_'+str(t)+'_tilde'] = c_0_phi_t_tilde
            opt_model += c_0_phi_t_tilde == c_0_phi_t + (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])
            
            if t==0:
                #already exits
                mu_plus_phi_t = dict_vars['muplus_'+str(id(phi))+'_t_'+str(t)]
            else:
                mu_plus_phi_t = plp.LpVariable('muplus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
                dict_vars['muplus_'+str(id(phi))+'_t_'+str(t)] = mu_plus_phi_t
            opt_model += mu_plus_phi_t == c_0_phi_t_tilde + c_1_phi_t_tilde
    
    
    
    
    opt_model.solve(plp.GUROBI_CMD(msg=True))
    
    for phi, priority in Demands:
        print(phi, priority)
        print("muplus", [dict_vars['muplus_'+str(id(phi))+'_t_'+str(t)].varValue for t in range(0,horizon-demands_horizon)])
        print("z_phi", [dict_vars['z1_'+str(id(phi))+'_t_'+str(t)].varValue for t in range(0,horizon-demands_horizon)])
        print()
    
    print("\n\n TRAJECTORY")
    for t in range(0,horizon):
        print(t, [(state, dict_vars['p_s_'+state+'_t_'+str(t)].varValue) for state in wts.S if dict_vars['b_s_'+state+'_t_'+str(t)].varValue == 1.0])
    
    print("\n PROBS")
    print("p_s_s_0_0_t_0",dict_vars['p_s_s_0_0_t_0'].varValue)
    print("p_s_s_0_0_t_1",dict_vars['p_s_s_0_0_t_1'].varValue)
    print("p_s_s_0_1_t_0",dict_vars['p_s_s_0_1_t_0'].varValue)
    print("p_s_s_0_1_t_1",dict_vars['p_s_s_0_1_t_1'].varValue)
    print("p_s_s_1_1_t_1",dict_vars['p_s_s_1_1_t_1'].varValue)
    print("p_s_s_1_1_t_2",dict_vars['p_s_s_1_1_t_2'].varValue)
    print("p_s_s_1_1_t_3",dict_vars['p_s_s_1_1_t_3'].varValue)
    
    
    return [[state for state in wts.S if dict_vars['b_s_'+state+'_t_'+str(t)].varValue == 1.0] for t in range(0,horizon)]


if __name__ == '__main__':

    #planning horizon
    # PLANNING_HORIZON = 965
    PLANNING_HORIZON = 10

    #The WTS of LV24, floor 4
    S = ['s_0_0','s_1_0','s_0_1','s_1_1']
    s_0 = 's_0_0'
    T = [('s_0_0','s_0_1'),('s_0_0','s_1_0'),('s_0_1','s_1_1'),('s_1_0','s_1_1')]
    for state in S:
        T.append((state,state))
    AP = ['Jana','Alexis']
    
    C = {}
    for s1 in S:
        C[s1] = {}
            
    for t in range(0,PLANNING_HORIZON):
        
        #self-loop
        for s1 in S:
            C[s1][(t,1,s1)] = 1
            
        #simple
        # C['s_0_0'][(t,2,'s_1_0')] = .1
        # C['s_1_0'][(t,2,'s_0_0')] = .1
        # C['s_0_0'][(t,2,'s_0_1')] = .2
        # C['s_0_1'][(t,2,'s_0_0')] = .2
        # C['s_0_1'][(t,1,'s_1_1')] = .5
        # C['s_1_1'][(t,1,'s_0_1')] = .5
        # C['s_1_0'][(t,1,'s_1_1')] = .5
        # C['s_1_1'][(t,1,'s_1_0')] = .5

        #from s_0_0 to s_1_0 and from s_1_0 to s_0_0
        C['s_0_0'][(t,1,'s_1_0')] = .1
        C['s_0_0'][(t,2,'s_1_0')] = .3
        C['s_0_0'][(t,3,'s_1_0')] = .3
        C['s_0_0'][(t,4,'s_1_0')] = .3
        C['s_1_0'][(t,1,'s_0_0')] = .1
        C['s_1_0'][(t,2,'s_0_0')] = .3
        C['s_1_0'][(t,3,'s_0_0')] = .3
        C['s_1_0'][(t,4,'s_0_0')] = .3
        #from s_0_0 to s_0_1 and from s_0_1 to s_0_0
        C['s_0_0'][(t,2,'s_0_1')] = .2
        C['s_0_0'][(t,3,'s_0_1')] = .6
        C['s_0_0'][(t,4,'s_0_1')] = .1
        C['s_0_0'][(t,5,'s_0_1')] = .1
        C['s_0_1'][(t,2,'s_0_0')] = .2
        C['s_0_1'][(t,3,'s_0_0')] = .6
        C['s_0_1'][(t,4,'s_0_0')] = .1
        C['s_0_1'][(t,5,'s_0_0')] = .1
        #from s_0_1 to s_1_1 and from s_1_1 to s_0_1
        C['s_0_1'][(t,1,'s_1_1')] = .5
        C['s_0_1'][(t,2,'s_1_1')] = .2
        C['s_0_1'][(t,3,'s_1_1')] = .3
        C['s_1_1'][(t,1,'s_0_1')] = .5
        C['s_1_1'][(t,2,'s_0_1')] = .2
        C['s_1_1'][(t,3,'s_0_1')] = .3
        #from s_1_0 to s_1_1 and from s_1_1 to s_1_0
        C['s_1_0'][(t,3,'s_1_1')] = .4        
        C['s_1_0'][(t,2,'s_1_1')] = .6
        C['s_1_1'][(t,2,'s_1_0')] = .5
        C['s_1_1'][(t,3,'s_1_0')] = .5

        

    L = {'s_1_1':'Jana','s_0_0':'Alexis'}
    wts = WTSProba(S,s_0,T,AP,L,C)
    
    

    #Eventually_[4,7] x=1,y=8
    e1 = MTLFormula.Eventually(MTLFormula.Predicate('Jana'),3,4)
   

    Demands = [(e1,2)]

    trajectory = generate_plan_mtl_wts(Demands,wts,PLANNING_HORIZON)
    print(trajectory)

    



