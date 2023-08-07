class MTLFormula:
    """
    Class for representing an STL Formula.
    """
    
    
    
    class TrueF:
        """
        Class representing the True boolean constant
        """
        def __init__(self):
            # self.robustness = lambda s, t : float('inf')
            self.horizon = 0
            
        def __str__(self):
            return "\\top"



    class FalseF:
        """
        Class representing the False boolean constant
        """
        def __init__(self):
            # self.robustness = lambda s, t : float('-inf')
            self.horizon = 0
            
        def __str__(self):
            return "\\bot"



    class Predicate:
        """
        Class representing a Predicate, s.t. f(s) \sim \mu
        The constructor takes 4 arguments:
            * dimension: string/name of the dimension
            * operator: operator (geq, lt...)
            * mu: \mu
            * pi_index_signal: in the signal, which index corresponds to the predicate's dimension
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,(f(s) \sim \mu),t) & = \begin{cases} \mu-f(s_t) & \sim=\le \\ f(s_t)-\mu & \sim=\ge \end{cases}
            * horizon: 0
        """
        def __init__(self,predicate):
            self.predicate = predicate
            self.horizon = 0
        
        def __str__(self):
            return str(self.predicate)



    class Conjunction: 
        """
        Class representing the Conjunction operator, s.t. \phi_1 \land \phi_2.
        The constructor takes 2 arguments:
            * formula 1: \phi_1
            * formula 2: \phi_2
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\phi_1 \land \phi_2,t) = \min(\rho(s,\phi_1,t),\rho(s,\phi_2,t) )
            * horizon: \left\|\phi_1 \land \phi_2\right\|= \max\{\left\|\phi_1\right\|, \left\|\phi_2\right\|\}
        """
        def __init__(self,list_formulas):
            self.list_formulas = list_formulas
            self.horizon = max([formula.horizon for formula in self.list_formulas])
        
        def __str__(self):
            s = "("
            for formula in self.list_formulas:
                s += str(formula) + " \wedge "
            s += ")"
            return s



    class Negation: 
        """
        Class representing the Negation operator, s.t. \neg \phi.
        The constructor takes 1 argument:
            * formula 1: \phi
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\neg \phi,t) = - \rho(s,\phi,t)
            * horizon: \left\|\phi\right\|=\left\|\neg \phi\right\|
        """
        def __init__(self,formula):
            self.formula = formula
            # self.robustness = lambda s, t : -formula.robustness(s,t)
            self.horizon = formula.horizon
        
        def __str__(self):
            return "\lnot ("+str(self.formula)+")"

    
    
    class Disjunction: 
        """
        Class representing the Disjunction operator, s.t. \phi_1 \vee \phi_2.
        The constructor takes 2 arguments:
            * formula 1: \phi_1
            * formula 2: \phi_2
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\phi_1 \lor \phi_2,t) = \max(\rho(s,\phi_1,t),\rho(s,\phi_2,t) )
            * horizon: \left\|\phi_1 \lor \phi_2\right\|= \max\{\left\|\phi_1\right\|, \left\|\phi_2\right\|\}
        """
        def __init__(self,list_formulas):
            self.list_formulas = list_formulas
            self.horizon = max([formula.horizon for formula in self.list_formulas])
        
        def __str__(self):
            s = "("
            for formula in self.list_formulas:
                s += str(formula) + " \vee "
            s += ")"
            return s

    
    
    class Always: 
        """
        Class representing the Always operator, s.t. \mathcal{G}_{[t1,t2]} \phi.
        The constructor takes 3 arguments:
            * formula: a formula \phi
            * t1: lower time interval bound
            * t2: upper time interval bound
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\mathcal{G}_{[t1,t2]}~ \phi,t) = underset{t' \in t+[t1,t2]}\min~  \rho(s,\phi,t').
            * horizon: \left\|\mathcal{G}_{[t1, t2]} \phi\right\|=t2+ \left\|\phi\right\|
        """
        def __init__(self,formula,t1,t2):
            self.formula = formula
            self.t1 = t1
            self.t2 = t2
            # self.robustness = lambda s, t : min([ formula.robustness(s,k) for k in range(t+t1, t+t2+1)])
            self.horizon = t2 + formula.horizon
        
        def __str__(self):
            return "\mathcal{G}_{["+str(self.t1)+","+str(self.t2)+"]}("+str(self.formula)+")"



    class Eventually: 
        """
        Class representing the Eventually operator, s.t. \mathcal{F}_{[t1,t2]} \phi.
        The constructor takes 3 arguments:
            * formula: a formula \phi
            * t1: lower time interval bound
            * t2: upper time interval bound
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\mathcal{F}_{[t1,t2]}~ \phi,t) = underset{t' \in t+[t1,t2]}\max~  \rho(s,\phi,t').
            * horizon: \left\|\mathcal{F}_{[t1, t2]} \phi\right\|=t2+ \left\|\phi\right\|
        """
        def __init__(self,formula,t1,t2):
            self.formula = formula
            self.t1 = t1
            self.t2 = t2
            # self.robustness = lambda s, t :  max([ formula.robustness(s,k) for k in range(t+t1, t+t2+1)])
            self.horizon = t2 + formula.horizon
        
        def __str__(self):
            return "\mathcal{F}_{["+str(self.t1)+","+str(self.t2)+"]}("+str(self.formula)+")"



    class Until: 
        """
        Class representing the Eventually operator, s.t. \mathcal{F}_{[t1,t2]} \phi.
        The constructor takes 3 arguments:
            * formula: a formula \phi
            * t1: lower time interval bound
            * t2: upper time interval bound
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\mathcal{F}_{[t1,t2]}~ \phi,t) = underset{t' \in t+[t1,t2]}\max~  \rho(s,\phi,t').
            * horizon: \left\|\mathcal{F}_{[t1, t2]} \phi\right\|=t2+ \left\|\phi\right\|
        """
        def __init__(self,first_formula,second_formula,t1,t2):
            self.first_formula = first_formula
            self.second_formula = second_formula            
            self.t1 = t1
            self.t2 = t2
            # self.robustness = lambda s, t :  max([ formula.robustness(s,k) for k in range(t+t1, t+t2+1)])
            self.horizon = t2 + max(first_formula.horizon,second_formula.horizon)
        
        def __str__(self):
            return "("+str(self.first_formula)+")\mathcal{U}_{["+str(self.t1)+","+str(self.t2)+"]}("+str(self.second_formula)+")"
    