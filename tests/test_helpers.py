import unittest

from aequitas.plot.commons.helpers import prettify


class TestPrettify(unittest.TestCase):
    def test_prettify(self):
        self.assertEqual(prettify(1234567), "1,234,567")
        self.assertEqual(prettify(-1234.5, separator=" "), "-1 234.5")
        self.assertEqual(prettify(123), "123")
