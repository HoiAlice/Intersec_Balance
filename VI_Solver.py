import numpy as np

def projector_to_Rn(dim: int):
    def P(x: np.ndarray):
        y = np.zeros(dim)
        for i in range(dim):
            y[i] = x[i]
        return x
    return P

def projector_to_Rn_plus(dim: int):
    def P(x: np.ndarray):
        y = np.zeros(dim)
        for i in range(dim):
            y[i] = x[i]
            if(x[i] <= 0):
                y[i] = 0
        return y
    return P

def projector_to_Cn(dim: int):
    def P(x: np.ndarray):
        y = np.zeros(dim)
        for i in range(dim):
            y[i] = min(1,max(0,x[i]))
        return y
    return P

def projector_to_Dn_fast(dim: int, a: float = 1): # n log n, но константа хуйня
    p_c = projector_to_Cn(dim)
    
    def partition(g: float, x:np.ndarray): #нужно будет сделать френдом, а не инициализировать внутри
        while(len(x) > 2):
            i = np.random.randint(1, len(x))
            if(g(x[i]) >= 0):
                x = x[i:]
            else:
                x = x[:i+1]
        if(x[0] == x[1]):
            return x[0]
        return ((g(x[1]) * x[0] - g(x[0]) * x[1] )/(g(x[1]) - g(x[0])))
    
    def P(x: np.ndarray):
        y = np.sort(np.concatenate((x, x - np.ones(dim))))
        g = lambda t: sum([min(1,max(0, x_i - t)) for x_i in x]) - a
        t = partition(g = g, x = y)
        x = p_c(x - t * np.ones(dim))
        return x
    return P

def projector_to_Dn_slow(dim: int): #работает только с a = 1
    p_c = projector_to_Cn(dim)
    def P(x: np.ndarray):
        y = np.sort(x)
        for i in range(dim - 1, 0, -1):
            t = (sum(x[i:]) - 1)/(dim - i)
            if(t >= y[i]):
                return p_c(x - t * np.ones(dim))
        t = (sum(x) - 1)/(dim)
        return p_c(x - t * np.ones(dim))
    return P

class Solver:
    def __init__(self,F, n, projector):
        self.F = F
        self.n = n
        self.projector = projector
        self.value = projector(np.zeros(n))
    
    def set_value(self, x_0: np.ndarray):
        self.value = self.projector(x_0)
    
    def step(self, a: float):
        y1 = self.projector(self.value - a * self.F(self.value))
        y2 = self.projector(self.value - a * self.F(y1))
        self.value = self.projector(self.value - a * self.F(y2))
        
    def solve(self, a: float, M: int):
        for m in range(M):
            self.step(a/np.sqrt(m + 1))