

class MDP(object):
    """ MDP
    pass in:
        states: a list of states, where each state contains a list of pre-defined and ordered state factors
        initial_state_probs: a dictionary where key is intial state, value is probabiliyt that is the intial state
        transitions: a dictionary with values of the form (state list, action, state list pairs) 
    
    """
    
    def __init__(
        self, 
        states, 
        initial_state, 
        actions, 
        transitions,
        allowable_actions = None
    ):
      self.states = states
      self.initial_state = initial_state
      self.actions = actions
      self.transitions = transitions
      self.allowable_actions = allowable_actions


    def get_states(self):  
        return self.states

    def get_actions(self): 
        return self.actions

    def get_initial_state(self):
        return self.initial_state

    def get_transition_prob(self, state, action, next_state):
        return self.transitions.get((state,action,next_state), 0)

    def get_allowable_actions(self, state):
        if self.allowable_actions == None:
            self.calculate_allowable_actions()
        return self.allowable_actions.get(state,[])

    def calculate_allowable_actions(self):
        self.allowable_actions = {}
        for s in self.get_states():
            s_allowable = set()
            for a in self.get_actions():
                for s2 in self.get_states():
                    if self.get_transition_prob( s,a,s2) > 0:
                        s_allowable.add(a)
            self.allowable_actions[s]=list(s_allowable)

    
