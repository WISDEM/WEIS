import unittest
from weis.test.utils import execute_script


class TestOC3(unittest.TestCase):
    def test_run(self):
        fscript = "06_IEA-15-240-RWT/weis_driver"

        execute_script(fscript)
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
