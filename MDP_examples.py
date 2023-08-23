from MDP_Object import MDP
from MTLTaskAllocSolver_MDP2 import timerobust_lp
from MTL import MTLFormula

import time


def construct_timed_MDP(untimed_initial_state, untimed_S, actions, additive_time_transition_dict, H):
    
    initial_state = (untimed_initial_state, 0)

    reachable_states = [(state, h) for state in untimed_S for h in range(H)]
    
    allowable_transitions = {}
    for (state, time) in reachable_states:
        # allowable_transitions_state = {}
        for allowable_action, next_transition_dict in additive_time_transition_dict[state].items():
            all_next_states = [(next_state, time + time_increase) for (next_state, time_increase) in next_transition_dict.keys()]
            valid =  all(item in reachable_states for item in all_next_states)
            if valid:
                for ((next_state, time_increase), probability) in next_transition_dict.items():
                    allowable_transitions[((state,time), allowable_action, (next_state, time + time_increase))] = probability
    
  
    timed_mdp=MDP(states=reachable_states, 
        initial_state=initial_state, 
        actions = actions,
        transitions=allowable_transitions)


    return timed_mdp

def construct_history_MDP(timed_mdp, untimed_S):
    timed_initial_state = timed_mdp.get_initial_state()
    timed_states = timed_mdp.get_states()
    actions = timed_mdp.get_actions()

    
    history_initial_state = (timed_initial_state,)
    
    history_states = []
    history_transitions = {}

    allowable_actions = {}

    next_states = [history_initial_state]


    while len(next_states)>0:
        next_next_states = []
        for hist_s1 in next_states:
            allowable_actions_hist_s1 = set()
            for a in timed_mdp.get_allowable_actions(hist_s1[-1]):
                for s2 in timed_states:
                    prob = timed_mdp.get_transition_prob(hist_s1[-1], a, s2)
                    if prob>0:
                        allowable_actions_hist_s1.add(a)
                        next_next_states.append(hist_s1+(s2,))
                        history_transitions[(hist_s1, a,hist_s1+(s2,) )] = prob
            if len(allowable_actions_hist_s1)>0:
                allowable_actions[hist_s1] = list(allowable_actions_hist_s1)
            elif len(hist_s1) < H+1:
                allowable_actions[hist_s1] = 'time_kill'
                # print(hist_s1+((hist_s1+(hist_s1[-1],))))
                next_next_states.append(hist_s1+(hist_s1[-1],))

        history_states= history_states+next_states
        next_states = next_next_states

    # print(history_states)
    
    history_mdp=MDP(states=history_states, 
        initial_state=history_initial_state, 
        # actions = actions,
        actions = actions+['time_kill'],
        transitions=history_transitions, 
        allowable_actions = allowable_actions)


    return history_mdp

        


def MDP_example_one(H=6):
    untimed_initial_state = 's_1'
    untimed_S = ['s_1','s_2']
    actions = ['s_1_1','s_1_2','s_2_1','s_2_2'] # Alexis called these states
    # this dictionary reprents ADDITIVE TIME, e.g. how much time will increase by, NOT how much time will be total
    additive_time_transition_dict = \
        {'s_1':{'s_1_1':{('s_1',1):1},'s_1_2':{('s_2',1):.5, ('s_2',2):.5}},
         's_2':{'s_2_2':{('s_2',1):1},'s_2_1':{('s_1',1):.5, ('s_1',2):.5}}
        }

    timed_mdp = construct_timed_MDP(untimed_initial_state, untimed_S, actions, additive_time_transition_dict,  H)
    return timed_mdp, untimed_S


def MDP_example_two(H=3):

    untimed_initial_state = 's_1'
    untimed_S= ['s_1','s_2','s_3']
    actions = ['s_'+str(s1)+'_'+str(s2) for s1 in range(1,4) for s2 in range(1,4)]


    additive_time_transition_dict = \
        {'s_1':{'s_1_1':{('s_1', 1):1},'s_1_2':{('s_2',1):.8, ('s_2',2):.2},'s_1_3':{('s_3',1):.2, ('s_3',2):.8}},
        's_2':{'s_2_2':{('s_2',1):1},'s_2_1':{('s_1',1):.6, ('s_1',2):.4},'s_2_3':{('s_3',1):.5, ('s_3',2):.5}}, 
        's_3':{'s_3_3':{('s_3',1):1},'s_3_1':{('s_1',1):.2, ('s_1',2):.8},'s_3_2':{('s_2',1):.5, ('s_2',2):.5}}
        }

    timed_mdp = construct_timed_MDP(untimed_initial_state, untimed_S, actions, additive_time_transition_dict,  H)
    return timed_mdp, untimed_S

def MDP_example_three(H=6):
    untimed_initial_state = 's_0'
    untimed_S = ['s_0','s_1']
    actions = ['s_0_0','s_1_0','s_0_1','s_1_1'] # Alexis called these states
    # this dictionary reprents ADDITIVE TIME, e.g. how much time will increase by, NOT how much time will be total
    additive_time_transition_dict = \
        {'s_0':{'s_0_0':{('s_0',1):1},'s_0_1':{('s_1',1):1,}},
         's_1':{'s_1_1':{('s_1',1):1},'s_1_0':{('s_0',1):.1, ('s_0',2):.3, ('s_0',3):.3, ('s_0',4):.3}}
        }

    timed_mdp = construct_timed_MDP(untimed_initial_state, untimed_S, actions, additive_time_transition_dict,  H)
    return timed_mdp, untimed_S



if __name__ == "__main__":
    H=7
    phi = MTLFormula.Conjunction([MTLFormula.Eventually(MTLFormula.Predicate('s_3'),1,4),MTLFormula.Eventually(MTLFormula.Predicate('s_2'),3,6)])
    # phi = MTLFormula.Conjunction([MTLFormula.Eventually(MTLFormula.Predicate('s_2'),1,4)])

    
    start_mdp = time.time()
    timed_mdp, untimed_S = MDP_example_two(H=H)
    history_mdp = construct_history_MDP(timed_mdp, untimed_S)
    end_mdp = time.time()
    print(end_mdp-start_mdp)
    timerobust_lp(history_mdp, H, phi, untimed_S)
