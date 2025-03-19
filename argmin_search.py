import typing as tp
import numpy as np
import math

class ArgminSearcher:
    phi = (1 + math.sqrt(5)) / 2
    def __init__(
        self,
        function: tp.Callable[[np.array], float] = lambda arg: np.sum(np.sin(arg)),
        frontier: tp.Tuple = (-10000., 10000.)):
        self.function = function
        self.frontier = frontier
    
    
    def find_argmin(self):
        l, r = self.frontier 
        xl, xr = l + (r - l) / (self.phi + 1), \
                r - (r - l) / (self.phi + 1) 
        while (xr - xl > 1e-6):
            pass
            
    
    
    
    
class ArgminSearcherScipy:
    # Ayaz code here  
    pass