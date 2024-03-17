import unittest
from BOWaves.cmmn.utils import load_tensor

# this is the base Pycharm template for unit tests.
# Use this to test the dataloaders.
# Also, set these up to run automatically during PR for CI/CD purposes.

class cmmn_tests(unittest.TestCase):
    def test_load_tensor_single(self):
        # test the load_tensor function.
        reshaped_tensor = load_tensor(
            directory_path='../tests/',
            file_list=['cue_subj_01_test.npz'],
            srate=256,
            segment_length=256*300 # 5 minutes
        )

        # check the shape of the tensor.
        # test data has 63 ICs, and we want 30 seconds as T.
        # num of centered signals is 10, as we give segment length of 5 min / 30 seconds
        self.assertEqual(reshaped_tensor.shape, (63, 10, 256*30))

    def test_load_tensor_multiple(self):
        # test the load_tensor function.
        reshaped_tensor = load_tensor(
            directory_path='../tests/',
            file_list=['cue_subj_01_test.npz', 'cue_subj_02_test.npz'],
            srate=256,
            segment_length=256*300 # 5 minutes
        )

        # check the shape of the tensor.
        # test data has 63 ICs, and we want 30 seconds as T.
        # num of centered signals is 10, as we give segment length of 5 min / 30 seconds
        self.assertEqual(reshaped_tensor.shape, (63*2, 10, 256*30))

    def test_load_tensor_correct_reshaping(self):
        pass #not yet implemented


if __name__ == '__main__':
    unittest.main()
