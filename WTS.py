class WTS:
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
    




if __name__ == '__main__':
    
    #The WTS of LV24, floor 4
    S = ['s_0_0','s_1_0','s_2_0','s_3_0','s_4_0','s_5_0','s_6_0','s_7_0','s_8_0','s_0_1','s_1_1','s_2_1','s_3_1','s_4_1','s_5_1','s_6_1','s_7_1','s_8_1','s_9_1','s_0_2','s_1_2','s_3_20','s_3_21','s_5_2','s_8_2','s_9_2','s_0_3','s_1_3','s_2_3','s_3_3','s_4_3','s_5_3','s_6_3','s_7_3','s_8_3','s_9_3','s_0_4','s_1_4','s_2_4','s_3_4','s_4_4','s_5_4','s_6_4','s_7_4','s_8_4','s_9_4']
    s_0 = 's_3_20'
    T = [('s_0_0','s_0_1'),('s_1_0','s_1_1'),('s_2_0','s_2_1'),('s_3_0','s_3_1'),('s_4_0','s_4_1'),('s_5_0','s_5_1'),('s_6_0','s_6_1'),('s_7_0','s_7_1'),('s_8_0','s_8_1'),('s_0_1','s_1_1'),('s_1_1','s_2_1'),('s_2_1','s_3_1'),('s_3_1','s_4_1'),('s_4_1','s_5_1'),('s_5_1','s_6_1'),('s_6_1','s_7_1'),('s_7_1','s_8_1'),('s_8_1','s_9_1'),('s_1_1','s_1_2'),('s_0_2','s_1_2'),('s_1_2','s_1_3'),('s_3_1','s_3_20'),('s_3_21','s_3_3'),('s_5_1','s_5_2'),('s_5_2','s_5_3'),('s_8_1','s_8_2'),('s_8_2','s_9_2'),('s_8_2','s_8_3'),('s_0_3','s_1_3'),('s_1_3','s_2_3'),('s_2_3','s_3_3'),('s_3_3','s_4_3'),('s_4_3','s_5_3'),('s_5_3','s_6_3'),('s_6_3','s_7_3'),('s_7_3','s_8_3'),('s_0_3','s_1_3'),('s_8_3','s_9_3'),('s_0_3','s_0_4'),('s_1_3','s_1_4'),('s_2_3','s_2_4'),('s_3_3','s_3_4'),('s_4_3','s_4_4'),('s_5_3','s_5_4'),('s_6_3','s_6_4'),('s_7_3','s_7_4'),('s_8_3','s_8_4'),('s_9_3','s_9_4')]
    AP = ['E1','E2','K','P','M1','M2','M1','Jana','Alexis']
    
    C = {}
    for s1, s2 in T:
        C[(s1, s2)] = 1
        C[(s2, s1)] = 1
        C[(s1, s1)] = 1
    C[('s_8_1','s_8_2')] = 2
    C[('s_8_2','s_8_3')] = 2
    C[('s_5_1','s_5_2')] = 3
    C[('s_5_2','s_5_3')] = 3
    
    L = {'s_3_20':'E1','s_3_21':'E2','s_9_2':'K','s_5_2':'P','s_0_2':'M1','s_3_4':'M2','s_9_1':'Jana','s_6_0':'Alexis'}
    wts = WTS(S,s_0,T,AP,L,C)
    
    print(wts.adjacent('s_3_1'))
    print([state for state in wts.S if wts.label_of_state(state)=='Jana'])
    
    wts.computeGVE(5)
    print(wts.V)
    print(wts.E)
    print(wts.I_i(('s_3_3', 2)))
    print(wts.O_i(('s_3_3', 2)))