from MTL import MTLFormula
import pulp as plp

#Computes the left and right time robustnesses of a given trace, against an MITL specificatiom

#MAX_RIGHT_TIME is the max value of mu minus
#is also used to set (fictive) states b_s_t for all t in [-MAX_RIGHT_TIME,-1]
    
def left_right_time_robustness(trace,phi,S,MAX_RIGHT_TIME):


    dict_vars = {}
    
    demands_horizon = phi.horizon    
    horizon = len(trace)
    
    opt_model = plp.LpProblem("MIP Model", plp.LpMaximize)


    mu_plus_phi_t = plp.LpVariable('muplus_'+str(id(phi))+'_t_'+str(0),cat='Integer')
    dict_vars['muplus_'+str(id(phi))+'_t_'+str(0)] = mu_plus_phi_t
    


    #Kurtz and Lin (2022). Introduce a binary variable bs(t) for each state s and each timestep t
    for state in S:
        for t in range(0,horizon):
            b_s_t = plp.LpVariable('b_s_'+state+'_t_'+str(t),cat='Binary')
            dict_vars['b_s_'+state+'_t_'+str(t)] = b_s_t
    
    #also define negative time states, necessary for the computation of the right time robustness
    for state in S:
        for t in range(-MAX_RIGHT_TIME,0):
            b_s_t = plp.LpVariable('b_s_'+state+'_t_'+str(t),cat='Binary')
            opt_model += b_s_t == 0
            dict_vars['b_s_'+state+'_t_'+str(t)] = b_s_t
    
    
    t = 0
    for state in trace:
        if state:
            opt_model += dict_vars['b_s_'+state+'_t_'+str(t)] == 1
            for s_prime in set(S)-set([state]):
                opt_model += dict_vars['b_s_'+s_prime+'_t_'+str(t)] == 0
        else:
            for s_prime in S:
                opt_model += dict_vars['b_s_'+s_prime+'_t_'+str(t)] == 0
        t += 1
    
    
    
    
    
    
    
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
            # opt_model += zvar == plp.lpSum([dict_vars['b_s_'+state+'_t_'+str(t)] for state in wts.S if wts.label_of_state(state)==phi.predicate])
            opt_model += zvar == plp.lpSum([dict_vars['b_s_'+state+'_t_'+str(t)] for state in S if state==phi.predicate])
            
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
        
    c_0_minus_phi_min1 = plp.LpVariable('c_0_minus_'+str(id(phi))+'_t_'+str(-MAX_RIGHT_TIME-1),cat='Integer')
    dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(-MAX_RIGHT_TIME-1)] = c_0_minus_phi_min1
    opt_model += dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(-MAX_RIGHT_TIME-1)] == 0

    for t in range(-MAX_RIGHT_TIME,1):
        c_1_minus_phi_t = plp.LpVariable('c_1_minus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
        dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t)] = c_1_minus_phi_t
        #if-then-else product construct of Rodionova (2022), Eq. 24 and 25
        x_l, x_u = -MAX_RIGHT_TIME-2, MAX_RIGHT_TIME+2
        opt_model += x_l*dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] <= c_1_minus_phi_t <= x_u*dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
        opt_model += (dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t-1)] + 1) - (x_u * (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])) <= c_1_minus_phi_t <= (dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t-1)] + 1) - (x_l * (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]))
        
        c_1_minus_phi_t_tilde = plp.LpVariable('c_1_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde',cat='Integer')
        dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde'] = c_1_minus_phi_t_tilde
        opt_model += c_1_minus_phi_t_tilde == c_1_minus_phi_t - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
        
        c_0_minus_phi_t = plp.LpVariable('c_0_minus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
        dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t)] = c_0_minus_phi_t
        #if-then-else product construct of Rodionova (2022), Eq. 24 and 25
        x_l, x_u = -MAX_RIGHT_TIME-2, MAX_RIGHT_TIME+2
        opt_model += x_l*(1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]) <= c_0_minus_phi_t <= x_u*(1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])
        opt_model += (dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t-1)] - 1) - (x_u * (1 - (1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]))) <= c_0_minus_phi_t <= (dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t-1)] - 1) - (x_l * (1 - (1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])))
        
        c_0_minus_phi_t_tilde = plp.LpVariable('c_0_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde',cat='Integer')
        dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde'] = c_0_minus_phi_t_tilde
        opt_model += c_0_minus_phi_t_tilde == c_0_minus_phi_t + (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])
        

        mu_minus_phi_t = plp.LpVariable('muminus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
        dict_vars['muminus_'+str(id(phi))+'_t_'+str(t)] = mu_minus_phi_t
        opt_model += mu_minus_phi_t == c_0_minus_phi_t_tilde + c_1_minus_phi_t_tilde
    
    
    
    opt_model.solve(plp.GUROBI_CMD(msg=False))
    
    
    return dict_vars['muplus_'+str(id(phi))+'_t_0'].varValue, dict_vars['muminus_'+str(id(phi))+'_t_0'].varValue








def left_right_time_robustness_demands(trace,Demands,S,MAX_RIGHT_TIME):

    dict_vars = {}
    
    demands_horizon = max([phi.horizon for phi, priority in Demands])
    horizon = len(trace)

    for phi, priority in Demands:
        mu_plus_phi_t = plp.LpVariable('muplus_'+str(id(phi))+'_t_'+str(0),cat='Integer')
        dict_vars['muplus_'+str(id(phi))+'_t_'+str(0)] = mu_plus_phi_t
        
    
    opt_model = plp.LpProblem("MIP Model", plp.LpMaximize)

    


    #Kurtz and Lin (2022). Introduce a binary variable bs(t) for each state s and each timestep t
    for state in S:
        for t in range(0,horizon):
            b_s_t = plp.LpVariable('b_s_'+state+'_t_'+str(t),cat='Binary')
            dict_vars['b_s_'+state+'_t_'+str(t)] = b_s_t
    
    #also define negative time states, necessary for the computation of the right time robustness
    for state in S:
        for t in range(-MAX_RIGHT_TIME,0):
            b_s_t = plp.LpVariable('b_s_'+state+'_t_'+str(t),cat='Binary')
            opt_model += b_s_t == 0
            dict_vars['b_s_'+state+'_t_'+str(t)] = b_s_t
    
    
    t = 0
    for state in trace:
        if state:
            opt_model += dict_vars['b_s_'+state+'_t_'+str(t)] == 1
            for s_prime in set(S)-set([state]):
                opt_model += dict_vars['b_s_'+s_prime+'_t_'+str(t)] == 0
        else:
            for s_prime in S:
                opt_model += dict_vars['b_s_'+s_prime+'_t_'+str(t)] == 0
        t += 1
    
    
    
    
    
    
    
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
            # opt_model += zvar == plp.lpSum([dict_vars['b_s_'+state+'_t_'+str(t)] for state in wts.S if wts.label_of_state(state)==phi.predicate])
            opt_model += zvar == plp.lpSum([dict_vars['b_s_'+state+'_t_'+str(t)] for state in S if state==phi.predicate])
            
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
    
    
    
    
    
    for phi, priority in Demands:
        #Recursive call of model_phi   
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
            
        c_0_minus_phi_min1 = plp.LpVariable('c_0_minus_'+str(id(phi))+'_t_'+str(-MAX_RIGHT_TIME-1),cat='Integer')
        dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(-MAX_RIGHT_TIME-1)] = c_0_minus_phi_min1
        opt_model += dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(-MAX_RIGHT_TIME-1)] == 0

        for t in range(-MAX_RIGHT_TIME,1):
            c_1_minus_phi_t = plp.LpVariable('c_1_minus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
            dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t)] = c_1_minus_phi_t
            #if-then-else product construct of Rodionova (2022), Eq. 24 and 25
            x_l, x_u = -MAX_RIGHT_TIME-2, MAX_RIGHT_TIME+2
            opt_model += x_l*dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] <= c_1_minus_phi_t <= x_u*dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            opt_model += (dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t-1)] + 1) - (x_u * (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])) <= c_1_minus_phi_t <= (dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t-1)] + 1) - (x_l * (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]))
            
            c_1_minus_phi_t_tilde = plp.LpVariable('c_1_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde',cat='Integer')
            dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde'] = c_1_minus_phi_t_tilde
            opt_model += c_1_minus_phi_t_tilde == c_1_minus_phi_t - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            
            c_0_minus_phi_t = plp.LpVariable('c_0_minus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
            dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t)] = c_0_minus_phi_t
            #if-then-else product construct of Rodionova (2022), Eq. 24 and 25
            x_l, x_u = -MAX_RIGHT_TIME-2, MAX_RIGHT_TIME+2
            opt_model += x_l*(1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]) <= c_0_minus_phi_t <= x_u*(1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])
            opt_model += (dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t-1)] - 1) - (x_u * (1 - (1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]))) <= c_0_minus_phi_t <= (dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t-1)] - 1) - (x_l * (1 - (1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])))
            
            c_0_minus_phi_t_tilde = plp.LpVariable('c_0_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde',cat='Integer')
            dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde'] = c_0_minus_phi_t_tilde
            opt_model += c_0_minus_phi_t_tilde == c_0_minus_phi_t + (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])
            

            mu_minus_phi_t = plp.LpVariable('muminus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
            dict_vars['muminus_'+str(id(phi))+'_t_'+str(t)] = mu_minus_phi_t
            opt_model += mu_minus_phi_t == c_0_minus_phi_t_tilde + c_1_minus_phi_t_tilde
    
    
    
    opt_model.solve(plp.GUROBI_CMD(msg=False))
    
    
    sum_muplus = sum([dict_vars['muplus_'+str(id(phi))+'_t_'+str(0)].varValue*priority for phi, priority in Demands])
    sum_muminus = sum([dict_vars['muminus_'+str(id(phi))+'_t_'+str(0)].varValue*priority for phi, priority in Demands])
    
    
    return sum_muplus, sum_muminus


if __name__ == '__main__':


    S = ['s_1','s_2','s_3']
    
    trace = ['s_1','','s_2','s_2','s_2','','','s_3','s_3','s_3','s_3','s_3','s_3','s_3','','','s_2','s_2','','','s_1','s_1','s_1','s_1','s_1','s_1','s_1','s_1','s_1','s_1','','','s_2','s_2','s_2','s_2','','','s_1','s_1','s_1','s_1','s_1','s_1','s_1','s_1','s_1','s_1']

    phi1 = MTLFormula.Eventually(MTLFormula.Predicate('s_3'),9,12)
    phi2 = MTLFormula.Eventually(MTLFormula.Predicate('s_3'),9,12)
    phi3 = MTLFormula.Conjunction([phi1,MTLFormula.Always(MTLFormula.Predicate('s_1'),22,24)])

    MAX_RIGHT_TIME_ROBUSTNESS = 20

    muplus, muminus = left_right_time_robustness(trace,phi1,S,MAX_RIGHT_TIME_ROBUSTNESS)
    print(muplus, muminus)
    
    # muplus, muminus = left_right_time_robustness(['s_1', 's_3', 's_3', 's_3', 's_1', 's_1', 's_1', 's_3', '', 's_1'],MTLFormula.Eventually(MTLFormula.Predicate('s_3'),3,5),['s_1', 's_2', 's_3'],10)
    # print(muplus, muminus)
    
    Demands = [(phi1,2),(phi2,3),(phi3,1)]
    sum_muplus, sum_muminus = left_right_time_robustness_demands(trace,Demands,S,MAX_RIGHT_TIME_ROBUSTNESS)
    print(sum_muplus, sum_muminus)

    
    exit()