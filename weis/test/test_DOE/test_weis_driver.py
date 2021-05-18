import unittest
from weis.test.utils import execute_script


class TestOC3(unittest.TestCase):
    def test_run(self):
        fscript = "09_design_of_experiments/weis_driver"

        execute_script(fscript)


if __name__ == "__main__":
    unittest.main()
