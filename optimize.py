import os
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.optimize import differential_evolution, basinhopping

def get_prog_path_name(name):
  script_path = os.path.abspath(os.path.dirname(__file__))
  script2prog = os.path.join("Build", "gmake", "bin", "Release", name)
  return os.path.join(script_path, script2prog)

def get_traj_file():
  script_path = os.path.abspath(os.path.dirname(__file__))
  return os.path.join(script_path, "trajectory.txt")

def get_prog_path():
  return get_prog_path_name("Testbed")

def get_lib_path():
  from sys import platform
  return get_prog_path_name("libTestbed_lib" + ('.dylib' if platform == "darwin" else '.so'))

def run_prog_process(args):
  return list(map(float, subprocess.check_output([get_prog_path()] + args).strip().split()))

import ctypes
#prog_lib = ctypes.cdll.LoadLibrary(get_lib_path())
prog_lib = ctypes.CDLL(get_lib_path())

def run_prog_lib(args):
  argc = len(args) + 1
  argv_l = ["asdf"] + args

  argv_type = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
  ret_type = ctypes.c_float * 2

  argv = (ctypes.POINTER(ctypes.c_char) * (argc + 1))()
  for i, arg in enumerate(argv_l):
    argv[i] = ctypes.create_string_buffer(arg.encode())

  prog_lib.my_func.restype = ctypes.POINTER(ctypes.c_float)
  rets = prog_lib.my_func(argc, argv)
  size = int(round(np.ctypeslib.as_array(rets, shape=(1,))[0]))
  positions = np.frombuffer((ctypes.c_float * (size + 1)).from_address(ctypes.addressof(rets.contents)), np.float32).copy()
  return positions[1:]

def go_right(positions):
  return -positions[-2]

cost_part_3 = go_right

box_x_4 = 10
box_y_4 = 10.5
def cost_part_4(positions):
  x, y = positions[-2], positions[-1]
  return (x - box_x_4) ** 2 + (y - box_y_4) ** 2

def my_normalize(vec):
  """ Vec is n by 2 """
  start = vec[0, :]
  end = vec[-1, :]

  return (vec - start) / (end - start)

traj = np.loadtxt(get_traj_file())
traj[:, 1] = -traj[:, 1]
traj = my_normalize(traj)

box_x_5 = -18
box_y_5 = 25
#box_x_5 = -20
#box_y_5 = 30
box_5 = np.array((box_x_5+0.5, box_y_5-0.6))
def cost_part_5(positions):
  assert len(positions) % 2 == 0
  positions = np.reshape(positions, (len(positions) // 2, 2))

  # Cutting
  vels = np.sum(np.abs(np.diff(positions)), axis=1)
  pos_errs = np.sum((positions - box_5) ** 2, axis=1)
  first_box_ind = (np.logical_and((pos_errs < 0.5), (vels < 0.3))).nonzero()[0]
  if len(first_box_ind):
    positions = positions[:, :first_box_ind[0]]

  # Scaling
  start = positions[0, :]
  end = np.array((box_x_5, box_y_5))
  scaled_traj = traj * (end - start) + start

  # Interpolation
  pos_interped1 = np.interp(np.linspace(0, 1, len(scaled_traj)), np.linspace(0, 1, len(positions)), positions[:, 0])
  pos_interped2 = np.interp(np.linspace(0, 1, len(scaled_traj)), np.linspace(0, 1, len(positions)), positions[:, 1])
  pos_interped = np.vstack((pos_interped1, pos_interped2)).T

  traj_interped1 = np.interp(np.linspace(0, 1, len(positions)), np.linspace(0, 1, len(scaled_traj)), scaled_traj[:, 0])
  traj_interped2 = np.interp(np.linspace(0, 1, len(positions)), np.linspace(0, 1, len(scaled_traj)), scaled_traj[:, 1])
  traj_interped = np.vstack((traj_interped1, traj_interped2)).T

  discount = 0.99
  traj_weights = discount**np.arange(len(traj_interped))
  scaled_discount =  (discount**(len(traj_interped)))**(1.0/len(pos_interped))
  pos_weights = scaled_discount**np.arange(len(pos_interped))
  return np.mean(((pos_interped - scaled_traj) ** 2).T * pos_weights) + \
         np.mean(((traj_interped - positions) ** 2).T * traj_weights)

box_x_6 = -8
box_y_6 = 30
def cost_part_6(positions):
  x, y = positions[-2], positions[-1]
  return (x - box_x_6) ** 2 + (y - box_y_6) ** 2

# params are
# gravity, friction, restitution
# then N times of
# x, y, rotation, width, height, <is gravity used>
def get_params_part_3(x):
  return np.array((-100, 0.0, 0.65, x[0], x[1], x[2] / 10.0, 1.0, 0.1, 0))

def get_params_part_4(x):
  return np.array((
    -100, 0.04, 0.45,
    box_x_4, box_y_4 - 0.6, 0, 1.0, 0.4, 0,
    box_x_4 + 1.4, box_y_4, 0, 0.4, 1.0, 0,
    box_x_4 - 1.4, box_y_4, 0, 0.4, 1.0, 0,
    x[0], x[1], x[2] / 10, 4.0, 0.8, 0,
    x[3], x[4], x[5] / 10, 4.0, 0.8, 0,
    x[6], x[7], x[8] / 10, 4.0, 0.8, 0
  ))

def get_params_part_5(x):
  return np.array((
    x[9], x[10] / 10, x[11] / 10,
    box_x_5, box_y_5 - 0.6, 0, 1.0, 0.4, 0,
    box_x_5 + 1.4, box_y_5, 0, 0.4, 1.0, 0,
    box_x_5 - 1.4, box_y_5, 0, 0.4, 1.0, 0,
    x[0], x[1], x[2] / 10, 4.0, 0.8, 0,
    x[3], x[4], x[5] / 10, 4.0, 0.8, 0,
    x[6], x[7], x[8] / 10, 4.0, 0.8, 0
  ))

def get_params_part_6(x):
  return np.array((
    -100, 0.04, 0.45,
    box_x_6, box_y_6 - 0.6, 0, 1.0, 0.4, 0,
    box_x_6 + 1.4, box_y_6, 0, 0.4, 1.0, 0,
    box_x_6 - 1.4, box_y_6, 0, 0.4, 1.0, 0,
    -18, box_y_6, 0, 1.5, 0.8, 0,
    -25, box_y_6, 0, 2.0, 0.8, 0,
    x[0], x[1], x[2]   / 10, 4.0, 0.4, 1,
    x[3], x[4], x[5]   / 10, 2.5, 0.4, 1,
    x[6], x[7], x[8]   / 10, 2.5, 0.4, 1,
    x[9], x[10], x[11] / 10, 2.5, 0.4, 1
  ))

def get_bounds_part_3():
  return [[-50, 0], [-20, 40], [0, 10 * np.pi]]

def get_bounds_part_4():
  return [[-40, 20], [0, 40], [0, 10 * np.pi],
          [-40, 20], [0, 40], [0, 10 * np.pi],
          [-40, 20], [0, 40], [0, 10 * np.pi]]

def get_bounds_part_5():
  return [[-50, 0], [20, 40], [0, 10 * np.pi],
          [-50, 0], [20, 40], [0, 10 * np.pi],
          [-50, 0], [20, 40], [0, 10 * np.pi],
          [-100, -20], [0, 25], [0, 5.0]]

def get_bounds_part_6():
  return [[-30, 0], [25, 35], [0, 1 * np.pi],
          [-30, 0], [25, 35], [0, 1 * np.pi],
          [-30, 0], [25, 35], [0, 1 * np.pi],
          [-30, 0], [25, 35], [0, 1 * np.pi]]


def fdlib(*args):
  return f(args)

def f(x, *args):
  real_x = get_params(x)
  positions = run_prog_lib(['0'] + list(map(str, real_x)))
  cost = cost_function(positions)

  return cost

def method_basinhopping():
  class MyBounds(object):
    def __init__(self, xmax=[1.1, 1.1], xmin=[-1.1, -1.1] ):
      self.xmax = np.array(xmax)
      self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
      x = kwargs["x_new"]
      tmax = bool(np.all(x <= self.xmax))
      tmin = bool(np.all(x >= self.xmin))
      return tmax and tmin

  randn = np.random.random_sample(len(bounds))
  x0 = np.array([x[0] + a*(x[1]-x[0]) for x,a in zip(bounds, randn) ])
  basin_bounds = MyBounds([x[1] for x in bounds], [x[0] for x in bounds])
  result = basinhopping(f, x0, accept_test=basin_bounds, niter=args.opt_iters, disp=True)
  return result.fun, result.x

def method_differential_evolution():
  result = differential_evolution(f, bounds, maxiter=args.opt_iters, disp=True)
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
  es = cma.fmin(f, x0, 2.0, options={
                              'verb_log': 0,
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
  parser.add_argument("--part", default='5', type=str, nargs='?', choices=params_map.keys())
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
