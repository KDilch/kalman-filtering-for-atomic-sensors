import unittest
from atomic_sensor_simulation.tests import test_noise
from atomic_sensor_simulation.tests import linear_kf_tests

def main():
    # initialize the test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # add tests to the test suite
    suite.addTests(loader.loadTestsFromModule(test_noise))
    suite.addTests(loader.loadTestsFromModule(linear_kf_tests))


    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)

if __name__ == '__main__':
    unittest.main()
