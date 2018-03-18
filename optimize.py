import os
import subprocess
import sys

import numpy as np
from scipy.optimize import differential_evolution, basinhopping


class MyBounds(object):
    def __init__(self, xmax=[1.1, 1.1], xmin=[-1.1, -1.1]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


def get_prog_path():
    script_path = os.path.abspath(os.path.dirname(__file__))
    script2prog = os.path.join("Build", "gmake", "bin", "Release", "Testbed")
    return os.path.join(script_path, script2prog)


def run_prog_process(args):
    return list(
        map(float,
            subprocess.check_output([get_prog_path()] + args).strip().split()))


import ctypes
prog_lib = ctypes.cdll.LoadLibrary(get_prog_path())


def run_prog_lib(args):
    argc = len(args) + 1
    argv = ["asdf"] + args
    prog_lib.main(argc, argv)


run_prog = run_prog_process


def go_right(x, y):
    return -x


def hit_target(x, y):
    x_target = 0.965
    y_target = 0.365
    return (x - x_target)**2 + (y - y_target)**2


def get_params_one(x):
    return np.array((x[0], 13.29349, 2.17307, -100, -100, -100, -100, -100,
                     -100, -100, -100, -100))


def get_params_two(x):
    return np.array((x[0], x[1], x[2], -100, -100, -100, -100, -100, -100,
                     -100, -100, -100))


get_params = get_params_two

best = 10000.0
best_x = None


def f(x, *args):
    #xx = x[:]
    #xx[2] /= 10
    #real_x = np.zeros(12)
    #real_x[0:3] = xx
    #real_x[3:] = np.array([-100] * 9)
    ##xx[5] /= 10
    ##xx[8] /= 10
    ##xx[11] /= 10

    real_x = get_params(x)

    global best, best_x
    ball_x, ball_y = run_prog(['0'] + list(map(str, real_x)))

    cost = cost_function(ball_x, ball_y)

    if cost < best:
        best = cost
        best_x = x

    return cost


#bounds = [(-50, 0), (5, 40), (0, 10 * np.pi),
#          (-50, 0), (5, 40), (0, 10 * np.pi),
#          (-50, 0), (5, 40), (0, 10 * np.pi),
#          (-50, 0), (5, 40), (0, 10 * np.pi)]
#bounds = [(-40, -20), (0, 40), (0, 10 * np.pi)]
bounds = [(-50, 0), (-20, 40), (0, np.pi)]


def method_basinhopping():
    randn = np.random.random_sample(len(bounds))
    x0 = np.array([x[0] + a * (x[1] - x[0]) for x, a in zip(bounds, randn)])
    basin_bounds = MyBounds([x[1] for x in bounds], [x[0] for x in bounds])
    result = basinhopping(
        f, x0, accept_test=basin_bounds, niter=niter, disp=True)
    return result.fun, result.x


def method_differential_evolution():
    result = differential_evolution(
        f, bounds, maxiter=niter, disp=True, tol=0.1)
    return result.fun, result.x


def method_random():
    i = 0
    while i < niter:
        randn = np.random.random_sample(len(bounds))
        x0 = np.array(
            [x[0] + a * (x[1] - x[0]) for x, a in zip(bounds, randn)])
        cost = f(x0)
        i += 1
        if i % 100 == 0:
            print(i, best)

    return best, best_x


def method_cma():
    import cma
    x0 = np.array([x[0] + 0.5 * (x[1] - x[0]) for x in bounds])
    es = cma.fmin(
        f,
        x0,
        3.0,
        options={
            'popsize': 10,
            'bounds': [[x[0] for x in bounds], [x[1] for x in bounds]]
        },
        restarts=2)
    return es[1], es[0]


if __name__ == "__main__":
    import argparse

    method_map = {
        x[7:]: y
        for x, y in globals().items() if x.startswith('method_')
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method",
        default="cma",
        type=str,
        nargs='?',
        choices=method_map.keys())
    parser.add_argument("--niter", default=100, type=int, nargs='?')
    args = parser.parse_args()

    niter = args.niter

    cost_function = go_right
    method = method_map[args.method]

    result_err, final_x = method()

    strres = [str(x) for x in get_params(final_x)]
    print("Final cost is", result_err)
    print("Optimal parameters are", ' '.join(strres))
    run_prog(['1'] + strres)
