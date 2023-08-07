#Definition of the right temporal robustness, inspired by Rodionova (2022), Eq. 24 to 28.
c_1_minus_phi_min1 = plp.LpVariable('c_1_minus_'+str(id(phi))+'_t_-1',cat='Integer')
dict_vars['c_1_minus_'+str(id(phi))+'_t_-1'] = c_1_minus_phi_min1
opt_model += dict_vars['c_1_minus_'+str(id(phi))+'_t_-1'] == 0
    
c_0_minus_phi_min1 = plp.LpVariable('c_0_minus_'+str(id(phi))+'_t_-1',cat='Integer')
dict_vars['c_0_minus_'+str(id(phi))+'_t_-1'] = c_0_minus_phi_min1
opt_model += dict_vars['c_0_minus_'+str(id(phi))+'_t_-1'] == 0

for t in range(horizon-demands_horizon):
    c_1_minus_phi_t = plp.LpVariable('c_1_minus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
    dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t)] = c_1_minus_phi_t
    #if-then-else product construct of Rodionova (2022), Eq. 24 and 25
    x_l, x_u = -(horizon-demands_horizon), horizon-demands_horizon
    opt_model += x_l*dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] <= c_1_minus_phi_t <= x_u*dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
    opt_model += (dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t-1)] + 1) - (x_u * (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])) <= c_1_minus_phi_t <= (dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t-1)] + 1) - (x_l * (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]))
    
    c_1_minus_phi_t_tilde = plp.LpVariable('c_1_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde',cat='Integer')
    dict_vars['c_1_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde'] = c_1_minus_phi_t_tilde
    opt_model += c_1_minus_phi_t_tilde == c_1_minus_phi_t - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
    
    c_0_minus_phi_t = plp.LpVariable('c_0_minus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
    dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t)] = c_0_minus_phi_t
    #if-then-else product construct of Rodionova (2022), Eq. 24 and 25
    x_l, x_u = -(horizon-demands_horizon), horizon-demands_horizon
    opt_model += x_l*(1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]) <= c_0_minus_phi_t <= x_u*(1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])
    opt_model += (dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t-1)] - 1) - (x_u * (1 - (1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]))) <= c_0_minus_phi_t <= (dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t-1)] - 1) - (x_l * (1 - (1-dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])))
    
    c_0_minus_phi_t_tilde = plp.LpVariable('c_0_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde',cat='Integer')
    dict_vars['c_0_minus_'+str(id(phi))+'_t_'+str(t)+'_tilde'] = c_0_minus_phi_t_tilde
    opt_model += c_0_minus_phi_t_tilde == c_0_minus_phi_t + (1 - dict_vars['z1_'+str(id(phi))+'_t_'+str(t)])
    
    if t==0:
        #already exits
        mu_minus_phi_t = dict_vars['muminus_'+str(id(phi))+'_t_'+str(t)]
    else:
        mu_minus_phi_t = plp.LpVariable('muminus_'+str(id(phi))+'_t_'+str(t),cat='Integer')
        dict_vars['muminus_'+str(id(phi))+'_t_'+str(t)] = mu_minus_phi_t
    opt_model += mu_minus_phi_t == c_0_minus_phi_t_tilde + c_1_minus_phi_t_tilde