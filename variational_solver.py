import numpy as np

def inner_point(x: float, y: float):
    if(x != -1 * np.infty):
        if(y != np.infty):
            return(0.5 * (x + y))
        else:
            return x
    else:
        if(y != np.infty):
            return y
        else:
            return 0
        
class Var_Solver:
    def __init__(self, F, min_bounds: np.ndarray, max_bounds: np.ndarray):
        if(len(min_bounds.shape) != 1 or len(max_bounds.shape) != 1):
            print("Error: bounds have wrong shapes")
            return None
        if(min_bounds.shape[0] != max_bounds.shape[0]):
            print("Error: bounds have different lenght")
            return None
        if((min_bounds > max_bounds).any()):
            print("Error: bounds incorrect")
            return None
        
        self.F = F
        self.n = min_bounds.shape[0]
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.value = np.array([inner_point(min_bounds[i], max_bounds[i]) for i in range(self.n)])
        
    def projection(self, x: np.ndarray):
        y = np.zeros(self.n)
        for i in range(self.n):
            y[i] = x[i]
            if(x[i] < self.min_bounds[i]):
                y[i] = self.min_bounds[i]
            if(x[i] > self.max_bounds[i]):
                y[i] = self.max_bounds[i]
        return y
    
    def set_value(self, x_0: np.ndarray):
        if(len(x_0.shape) != 1):
            print("Error: x_0 have wrong shape")
            return None
        if(x_0.shape[0] != self.n):
            print("Error: x_0 have wrong lenght")
            return None
        
        self.value = self.projection(x_0)
    
    def step(self, a: float):
        if(a <= 0):
            print("Error: step size are negaitve")
            return None
        
        y1 = self.projection(self.value - a * self.F(self.value))
        y2 = self.projection(self.value - a * self.F(y1))
        self.value = self.projection(self.value - a * self.F(y2))
        
    def solve(self, a: float, M: int):
        for m in range(M):
            self.step(a/np.sqrt(m + 1))