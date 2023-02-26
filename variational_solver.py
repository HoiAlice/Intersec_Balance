import numpy as np

def projection_to_Rn(dim: int):
    def P(x: np.ndarray):
        return x
    return P

def projector_to_cube(dim: int):
    def P(x: np.ndarray):
        for i in range(dim):
            if(x[i] >= 1):
                x[i] = 1
            if(x[i] <= 0):
                x[i] = 0
        return x
    return P

def projector_to_Rn_plus(dim: int):
    def P(x: np.ndarray):
        for i in range(dim):
            if(x[i] <= 0):
                x[i] = 0
        return x
    return P

class Var_Solver:
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