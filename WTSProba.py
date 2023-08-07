class WTSProba:
    """
    Class for representing a WTS.
    """    
    def __init__(self,S,s_0,T,AP,L,C):
        self.S = S
        self.s_0 = s_0
        self.T = T
        self.AP = AP
        self.L = L
        self.C = C
        
        self.adjacent_states = {}
        for state in self.S:
            self.adjacent_states[state] = {}
        for s1, s2 in self.T:
            self.adjacent_states[s1][s2] = None
            self.adjacent_states[s2][s1] = None
        
        self.adjacent = lambda s : list(self.adjacent_states[s])
        
        def label_of_state(s):
            try:
                return L[s]
            except KeyError:
                return None
        self.label_of_state = label_of_state
        
        #directed graph G = (V, E) of Kurtz and Lin (2022), Sect. IV.
        self.V = []
        self.E = []
        self.I_i = lambda node : [j for (j,i) in self.E if i==node]
        self.O_i = lambda node : [j for (i,j) in self.E if i==node]
    
    #directed graph G = (V, E) of Kurtz and Lin (2022), Sect. IV.
    def computeGVE(self,horizon):
        self.V = [(state,t) for state in self.S for t in range(0,horizon+1)]
        self.E = [((state,t),(adj,t+1)) for state in self.S for adj in self.adjacent(state) for t in range(0,horizon)]
    



