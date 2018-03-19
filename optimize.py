import os
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.optimize import differential_evolution, basinhopping

class MyBounds(object):
  def __init__(self, xmax=[1.1, 1.1], xmin=[-1.1, -1.1] ):
    self.xmax = np.array(xmax)
    self.xmin = np.array(xmin)

  def __call__(self, **kwargs):
    x = kwargs["x_new"]
    tmax = bool(np.all(x <= self.xmax))
    tmin = bool(np.all(x >= self.xmin))
    return tmax and tmin

def get_prog_path_name(name):
  script_path = os.path.abspath(os.path.dirname(__file__))
  script2prog = os.path.join("Build", "gmake", "bin", "Release", name)
  return os.path.join(script_path, script2prog)

def get_prog_path():
  return get_prog_path_name("Testbed")

def get_lib_path():
  return get_prog_path_name("libTestbed_lib.so")

def run_prog_process(args):
  return list(map(float, subprocess.check_output([get_prog_path()] + args).strip().split()))

import ctypes
#prog_lib = ctypes.cdll.LoadLibrary(get_lib_path())
prog_lib = ctypes.CDLL(get_lib_path())
prog_lib.my_func.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_float, shape=(2,))

def run_prog_lib(args):
  argc = len(args) + 1
  argv_l = ["asdf"] + args

  argv_type = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
  ret_type = ctypes.c_float * 2

  argv = (ctypes.POINTER(ctypes.c_char) * (argc + 1))()
  for i, arg in enumerate(argv_l):
    argv[i] = ctypes.create_string_buffer(arg.encode())

  rets = prog_lib.my_func(argc, argv)
  return rets[0], rets[1]

def go_right(x, y):
  return -x

cost_part_3 = go_right

box_x = 10
box_y = 10.5
def cost_part_4(x, y):
  return (x - box_x) ** 2 + (y - box_y) ** 2

def get_params_part_3(x):
  return np.array((x[0], x[1], x[2] / 10.0, 1.0, 0.1))

def get_params_part_4(x):
  return np.array((
    box_x, box_y - 0.6, 0, 1.0, 0.4,
    box_x + 1.4, box_y, 0, 0.4, 1.0,
    box_x - 1.4, box_y, 0, 0.4, 1.0,
    x[0], x[1], x[2] / 10, 5.0, 0.4,
    x[3], x[4], x[5] / 10, 5.0, 0.4,
    x[6], x[7], x[8] / 10, 5.0, 0.4
  ))

def get_bounds_part_3():
  return [[-50, 0], [-20, 40], [0, 10 * np.pi]]

def get_bounds_part_4():
  return [[-40, 20], [0, 40], [0, 10 * np.pi],
          [-40, 20], [0, 40], [0, 10 * np.pi],
          [-40, 20], [0, 40], [0, 10 * np.pi]]

def fdlib(*args):
  return f(args)

def f(x, *args):
  real_x = get_params(x)
  ball_x, ball_y = run_prog_lib(['0'] + list(map(str, real_x)))
  cost = cost_function(ball_x, ball_y)

  return cost

def method_basinhopping():
  randn = np.random.random_sample(len(bounds))
  x0 = np.array([x[0] + a*(x[1]-x[0]) for x,a in zip(bounds, randn) ])
  basin_bounds = MyBounds([x[1] for x in bounds], [x[0] for x in bounds])
  result = basinhopping(f, x0, accept_test=basin_bounds, niter=args.opt_iters, disp=True)
  return result.fun, result.x

def method_differential_evolution():
  result = differential_evolution(f, bounds, maxiter=args.opt_iters, disp=True, tol=0.1)
  return result.fun, result.x

def get_random_x0():
  randn = np.random.random_sample(len(bounds))
  return np.array([x[0] + a*(x[1]-x[0]) for x,a in zip(bounds, randn) ])

def _method_cg_eps(eps):
  result = scipy.optimize.minimize(f, get_random_x0(), method='CG', options={'disp' : False, 'eps' : eps })
  return result.fun, result.x

def method_cg_eps_low():
  return _method_cg_eps(1e-8)

def method_cg_eps_high():
  return _method_cg_eps(0.1)

def method_slsqp_high():
  result = scipy.optimize.minimize(f, get_random_x0(), method='SLSQP', options={'disp' : False, 'eps' : 0.1 })
  return result.fun, result.x

def method_slsqp_low():
  result = scipy.optimize.minimize(f, get_random_x0(), method='SLSQP', options={'disp' : False, 'eps' : 1e-8 })
  return result.fun, result.x

def method_random():
  best = 1e20
  best_x = None
  for i in range(args.opt_iters):
    x0 = get_random_x0()
    cost = f(x0)

    if cost < best:
      best = cost
      best_x = x0

  return best, best_x

def method_dlib():
  import dlib
  x = dlib.find_min_global(fdlib, *list(map(list, zip(*bounds))), args.opt_iters)
  return x[1], x[0]

def method_cma():
  import cma
  x0 = np.array([x[0] + 0.5*(x[1]-x[0]) for x in bounds])
  es = cma.fmin(f, x0, 30.0, options={
                              'popsize': 80,
                              #'tolfun': 1e-2,
                              'maxfevals': args.opt_iters,
                              #'tolx' : 1e-3,
                              'bounds': [[x[0] for x in bounds], [x[1] for x in bounds ] ]},
                              restarts=0)
  return es[1], es[0]

def run_n(f, n):
  costs = []
  times = []
  xs = []

  for i in range(n):
    t = time.time()
    cost, x = f()
    times.append(time.time() - t)
    costs.append(cost)
    xs.append(x)

  costs = np.array(costs)
  return costs.min(), costs.mean(), costs.std(), sum(times), xs[np.argmin(costs)]

if __name__ == "__main__":
  import argparse

  def get_map(prefix):
    return {x[len(prefix):] : y for x, y in globals().items() if x.startswith(prefix)}

  method_map = get_map('method_')
  params_map = get_map('get_params_part_')
  bounds_map = get_map('get_bounds_part_')
  cost_map = get_map('cost_part_')

  parser = argparse.ArgumentParser()
  parser.add_argument("method", default="random", type=str, nargs='?', choices=method_map.keys())
  parser.add_argument("--opt_iters", default=100, type=int, nargs='?')
  parser.add_argument("--exp_iters", default=1, type=int, nargs='?')
  parser.add_argument("--part", default='4', type=str, nargs='?', choices=params_map.keys())
  args = parser.parse_args()

  cost_function = cost_map[args.part]
  method = method_map[args.method]
  get_params = params_map[args.part]
  bounds = bounds_map[args.part]()

  result_err, result_mean, result_stddev, total_time, final_x = run_n(method, args.exp_iters)

  strres= [str(x) for x in get_params(final_x)]
  print("Final cost is", result_err)
  print("Optimal parameters are", ' '.join(strres))
  print("Mean is", result_mean)
  print("Stddev is", result_stddev)
  print("Avg time is", total_time / args.exp_iters)
  run_prog_process(['1'] + strres)
