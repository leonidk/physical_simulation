import subprocess
import sys

import numpy as np
from scipy.optimize import differential_evolution, basinhopping


class MyBounds(object):
    def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1] ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

best = 10000.0
def f(x,*args):
    global best
    if best > 0.5:
        res = subprocess.check_output(['./Build/gmake2/bin/Release/Testbed', '0'] + [str(_) for _ in x])
        numerical_res = float(res.strip())
        #print(numerical_res)
        best = min(numerical_res,best)
        return numerical_res
    else:
        return best

niter = 100

if len(sys.argv) > 1:
    niter = int(sys.argv[1])

bounds = [(-10,10), (0, 10), (0, 3.14159), (-10,10), (0, 10), (0, 3.14159), (-10,10), (0, 10), (0, 3.14159),(-10,10), (0, 10), (0, 3.14159)]

if False:
    randn = np.random.random_sample(len(bounds))
    x0 = np.array([x[0] + a*(x[1]-x[0]) for x,a in zip(bounds,randn) ])
    basin_bounds = MyBounds([x[1] for x in bounds], [x[0] for x in bounds])
    result = basinhopping(f,x0,accept_test=basin_bounds,niter=niter,disp=True)
    result_err = result.fun
    final_x = result.x
elif False:
    result = differential_evolution(f, bounds,maxiter=niter,disp=True,tol=0.1)
    result_err = result.fun
    final_x = result.x
elif False:
    i=0
    while best > 0.5:
        randn = np.random.random_sample(len(bounds))
        x0 = np.array([x[0] + a*(x[1]-x[0]) for x,a in zip(bounds,randn) ])
        f(x0)
        i+=1
        if i%100 == 0:
            print(i,best)
    result_err = best
    final_x = x0
elif False:
    import cma
    x0 = np.array([x[0] + 0.5*(x[1]-x[0]) for x in bounds ])
    es = cma.fmin(f,x0, 3.0,options={'popsize': 80, 'ftarget':0.5,'bounds':[[x[0] for x in bounds], [x[1] for x in bounds ]  ]}, restarts=3)
    result_err = es[1]
    final_x =es[0]
else:
    pass
    

strres= [str(x) for x in final_x]
print(result_err)
print(' '.join(strres))
subprocess.check_output(['./Build/gmake2/bin/Release/Testbed', '1'] + strres)
