import unittest
import numpy as np
import torch
import transformers
from transformers import EvalPrediction
from src.tokenizer.rt_tokenizer import RtTokenizer
from src.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer
from src.tokenizer.xval_tokenizer import XvalTokenizer
from src.evaluation import CustomMetrics

class TestEvaluationMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize your tokenizers with different encoding schemes
        cls.tokenizer_none = transformers.AutoTokenizer.from_pretrained("t5-small")
        cls.tokenizer_none_custom = T5Custom_Tokenizer.from_pretrained("t5-small")
        cls.tokenizer_xval = XvalTokenizer.from_pretrained("t5-small")
        cls.tokenizer_rt = RtTokenizer.from_pretrained("t5-small")

        # Create instances of the CustomMetrics class for each encoding type
        cls.metrics_none = CustomMetrics(cls.tokenizer_none, number_encoding="none", output_dir="tests/test_output")
        cls.metrics_none_custom = CustomMetrics(cls.tokenizer_none_custom, number_encoding="none", output_dir="tests/test_output")
        cls.metrics_xval = CustomMetrics(cls.tokenizer_xval, number_encoding="xval", output_dir="tests/test_output")
        cls.metrics_rt = CustomMetrics(cls.tokenizer_rt, number_encoding="rt", output_dir="tests/test_output")

    def test_calculate_result_mse(self):
        #Important make sure that if you remove ### again that the model matches the last number and completely.
        labels = [
            "Simple #### 5", #Parse positive 
            "First test 23.0 and -4.0 #### -4.0", #Parse negative
            "First test 23.0 and -4.0 #### -4.0", #Parse negative and positiv
            "Is 29.0 - 478.2 = 34.452 correct? #### 34.452", #Parse decimal
            "Test text -34*65=80 #### 78", #Focusses on last
            "Test 12-12 = 0 wrong? #### 0", #Handles missing value gracefully
            "Calculation: 12 + 12 = 24 #### 78", #Handels missing marker gracefully
            "Testing long 19,202 #### 19,202", #Handels , 
            "Testing long 19,203,023 #### 19,203,023", #Can handle two ,,
            #Current logic doesn't handle 123,21 correctly makes it 12321 instead of throwing error
        ]
        predictions = [
            "Simple #### 7",
            "First test 23.0 and -1.0 #### -1.0",
            "First test 23.0 and -1.0 #### 1.0",
            "Is 29.0 - 478.2 = 34.452 correct? #### 34.452",
            "Test text -34*65=80 #### 80",
            "Test 12-12 = -1 wrong? #### ",
            "Calculation: calculation calculation",
            "Testing long 19,203 #### 19,203",
            "Testing long 19,203,923 #### 19,203,923"
        ]

        mse = self.metrics_xval.calculate_result_mse(predictions, labels)
        expected_mse = np.array([(5 - 7)**2,(4 - 1)**2, (-4 - 1)**2, 0, (78 - 80)**2, np.nan, np.nan, 1, (19203023-19203923)**2])
        np.testing.assert_almost_equal(mse, expected_mse)


if __name__ == "__main__":
    unittest.main()
