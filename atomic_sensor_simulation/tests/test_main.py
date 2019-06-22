import unittest
from logging import getLogger
from atomic_sensor_simulation.tests import test_noise
from atomic_sensor_simulation.tests import test_smoke_tests

logger = getLogger(__name__)

# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(test_noise))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)

if __name__ == '__main__':
    logger.info("Starting tests...")
    logger.info("Starting smoke tests...")
    test_smoke_tests.run_smoke_tests()
    logger.info("Starting unit tests...")
    unittest.main()
    logger.info("Tests finished!")
