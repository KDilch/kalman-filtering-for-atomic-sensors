import unittest
from tests import test_noise, test_smoke_tests
# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(test_noise))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)

if __name__ == '__main__':
    unittest.main()
    test_smoke_tests.run_smoke_tests()
