import typing as tp
import numpy as np
import math

class ArgminSearcher:
    phi = (1 + math.sqrt(5)) / 2
    def __init__(
        self,
        function: tp.Callable[[float], float] = lambda arg: np.sum(np.sin(arg)),
        frontier: tp.Tuple = (-10000., 10000.)):
        self.function = function
        self.frontier = frontier
    
    
    def find_argmin(self):
        l, r = self.frontier 
        xl, xr = l + (r - l) / (self.phi + 1), \
                r - (r - l) / (self.phi + 1)
        while (xr - xl > 1e-6):
            fl, fr = self.function(xl), self.function(xr) 
            if (fl < fr):
                r = xr
                xr, fr = xl, fl
                xl = l + (r - l) / (self.phi + 1) 
                fl = self.function(xl)
            else:
                l = xl
                xl, fl = xr, fr
                xr = r - (r - l) / (self.phi + 1)
                fr = self.function(xr)
        return (l + r) / 2    
    
first =  ArgminSearcher(function =lambda x: x**2, frontier=(-100., 100.))
print(first.find_argmin())
