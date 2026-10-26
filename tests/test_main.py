import unittest
from main import parse_recording_name, validate_epochs

class AudioHelpersTests(unittest.TestCase):
    def test_parse_filename(self): self.assertEqual(parse_recording_name("7_Jackson_32.wav"), (7, "jackson"))
    def test_reject_invalid_filename(self):
        with self.assertRaises(ValueError): parse_recording_name("readme.txt")
    def test_validate_epochs(self): self.assertEqual(validate_epochs(2), 2)
    def test_reject_zero_epochs(self):
        with self.assertRaises(ValueError): validate_epochs(0)

if __name__ == "__main__": unittest.main()
