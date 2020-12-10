import unittest
from weis.test.utils import execute_script


class TestOC3(unittest.TestCase):
    def test_run(self):
        fscript = "03_NREL5MW_OC3_spar/weis_driver"

        execute_script(fscript)


if __name__ == "__main__":
    unittest.main()
