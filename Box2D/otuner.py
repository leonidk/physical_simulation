import argparse
import subprocess

import opentuner
from opentuner.measurement import MeasurementInterface
from opentuner.search.manipulator import ConfigurationManipulator
from opentuner.search.manipulator import FloatParameter

parser = argparse.ArgumentParser(parents=opentuner.argparsers())
bounds = [(-10,10), (0, 10), (0, 3.14159), (-10,10), (0, 10), (0, 3.14159), (-10,10), (0, 10), (0, 3.14159),(-10,10), (0, 10), (0, 3.14159)]

# python otuner.py --technique PSO_GA_Bandit --test-limit=1000 --parallelism 6
# ./Build/gmake2/bin/Release/Testbed 1 3.14 3.21 1.08 -4.90 5.46 2.37 -0.46 2.13 0.26 4.05 4.14 2.45


class Simulator(MeasurementInterface):
    def __init__(self, args):
        super(Simulator, self).__init__(args)
        self.parallel_compile = True

    def run_precompiled(self, desired_result, input, limit, compile_result, id):
        return compile_result

    def run(self, desired_result, input, limit):
        pass

    #def run(self, desired_result, input, limit):
    def compile(self, cfg, id):
        #cfg = desired_result.configuration.data
        x = [v for k,v in sorted([(int(k),v) for k,v in cfg.items()])]
        res = subprocess.check_output(['./Build/gmake2/bin/Release/Testbed', '0'] + [str(_) for _ in x])
        numerical_res = float(res.strip())
        return opentuner.resultsdb.models.Result(time=numerical_res)

    def manipulator(self):
        manipulator = ConfigurationManipulator()
        for idx,b in enumerate(bounds):
            manipulator.add_parameter(FloatParameter(str(idx),
                                                    b[0],
                                                    b[1]))
        return manipulator

    def save_final_config(self, configuration):
        print("Final configuration", configuration.data)
        cfg = configuration.data
        x = [v for k,v in sorted([(int(k),v) for k,v in cfg.items()])]
        print(' '.join(["{:.2f}".format(_) for _ in x]))

if __name__ == '__main__':
    args = parser.parse_args()
    Simulator.main(args)

