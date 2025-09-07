import tensorflow as tf
import numpy as np
import unittest
import os


class TestExampleOpCPU(unittest.TestCase):
    """Test cases for the Example custom TensorFlow operation on CPU."""
    
    @classmethod
    def setUpClass(cls):
        """Load the custom operation from the shared library."""
        # Assuming the compiled shared library is named 'kernel_example.so'
        # Adjust the path as needed based on your build setup
        so_file_path = './kernel_example.so'  # or the actual path to your .so file
        
        if not os.path.exists(so_file_path):
            raise FileNotFoundError(f"Shared library not found at {so_file_path}")
        
        # Load the custom operation
        cls.example_module = tf.load_op_library(so_file_path)
        cls.device = '/CPU:0'
    
    def test_example_op_2d_tensor(self):
        """Test the Example operation with 2D tensor input on CPU."""
        # Create test input
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected_output = input_data * 2
        
        # Run the operation using eager execution on CPU
        with tf.device(self.device):
            input_tensor = tf.constant(input_data)
            output_tensor = self.example_module.example(input_tensor)
        
        # Verify the result
        np.testing.assert_array_almost_equal(output_tensor.numpy(), expected_output)


class TestExampleOpGPU(unittest.TestCase):
    """Test cases for the Example custom TensorFlow operation on GPU."""
    
    @classmethod
    def setUpClass(cls):
        """Load the custom operation and check GPU availability."""
        # Check if GPU is available
        if not tf.config.list_physical_devices('GPU'):
            raise unittest.SkipTest("GPU not available, skipping GPU tests")
        
        # Assuming the compiled shared library is named 'kernel_example.so'
        so_file_path = './kernel_example.so'  # or the actual path to your .so file
        
        if not os.path.exists(so_file_path):
            raise FileNotFoundError(f"Shared library not found at {so_file_path}")
        
        # Load the custom operation
        cls.example_module = tf.load_op_library(so_file_path)
        cls.device = '/GPU:0'
    
    def test_example_op_2d_tensor_gpu(self):
        """Test the Example operation with 2D tensor input on GPU."""
        # Create test input
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected_output = input_data * 2
        
        # Run the operation using eager execution on GPU
        with tf.device(self.device):
            input_tensor = tf.constant(input_data)
            output_tensor = self.example_module.example(input_tensor)
        
        # Verify the result
        np.testing.assert_array_almost_equal(output_tensor.numpy(), expected_output)


class TestExampleOpGeneral(unittest.TestCase):
    """General test cases that work on both CPU and GPU."""
    
    @classmethod
    def setUpClass(cls):
        """Load the custom operation from the shared library."""
        so_file_path = './kernel_example.so'
        
        if not os.path.exists(so_file_path):
            raise FileNotFoundError(f"Shared library not found at {so_file_path}")
        
        cls.example_module = tf.load_op_library(so_file_path)
    
    def test_empty_tensor(self):
        """Test the Example operation with empty tensor."""
        # Create empty test input
        input_data = np.array([], dtype=np.float32).reshape(0, 0)
        expected_output = input_data * 2
        
        # Run the operation using eager execution
        input_tensor = tf.constant(input_data)
        output_tensor = self.example_module.example(input_tensor)
        
        # Verify the result
        np.testing.assert_array_almost_equal(output_tensor.numpy(), expected_output)
    
    def test_large_tensor(self):
        """Test the Example operation with a larger 2D tensor."""
        # Create larger test input
        input_data = np.random.rand(100, 50).astype(np.float32)
        expected_output = input_data * 2
        
        # Run the operation using eager execution
        input_tensor = tf.constant(input_data)
        output_tensor = self.example_module.example(input_tensor)
        
        # Verify the result
        np.testing.assert_array_almost_equal(output_tensor.numpy(), expected_output)
    
    def test_output_shape_preservation(self):
        """Test that the output shape matches the input shape."""
        # Test with different 2D shapes
        test_shapes = [(2, 3), (5, 4), (10, 1), (1, 20), (3, 3, 2)]  # Including 3D
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                input_data = np.random.rand(*shape).astype(np.float32)
                
                # Run the operation using eager execution
                input_tensor = tf.constant(input_data)
                output_tensor = self.example_module.example(input_tensor)
                result = output_tensor.numpy()
                
                # Check that shapes match
                self.assertEqual(result.shape, input_data.shape)
                # Check that values are doubled
                np.testing.assert_array_almost_equal(result, input_data * 2)
    
    def test_with_tf_function(self):
        """Test the Example operation inside a tf.function."""
        @tf.function
        def test_function(x):
            return self.example_module.example(x)
        
        # Test with 2D float32
        input_data = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        result = test_function(input_data)
        expected = input_data * 2
        
        tf.debugging.assert_near(result, expected)
    
    def test_zero_values(self):
        """Test the Example operation with zero values."""
        # Test with zeros in 2D
        input_data = tf.constant([[0.0, 0.0], [0.0, 0.0]], dtype=tf.float32)
        result = self.example_module.example(input_data)
        expected = tf.constant([[0.0, 0.0], [0.0, 0.0]], dtype=tf.float32)
        
        tf.debugging.assert_near(result, expected)
    
    def test_negative_values(self):
        """Test the Example operation with negative values."""
        # Test with negative values in 2D
        input_data = tf.constant([[-1.0, -2.5], [-3.7, -4.2]], dtype=tf.float32)
        result = self.example_module.example(input_data)
        expected = input_data * 2
        
        tf.debugging.assert_near(result, expected)


class TestExampleOpDeviceComparison(unittest.TestCase):
    """Test to compare CPU vs GPU results."""
    
    @classmethod
    def setUpClass(cls):
        """Load the custom operation from the shared library."""
        so_file_path = './kernel_example.so'
        
        if not os.path.exists(so_file_path):
            raise FileNotFoundError(f"Shared library not found at {so_file_path}")
        
        cls.example_module = tf.load_op_library(so_file_path)
        cls.has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    
    def test_cpu_gpu_consistency(self):
        """Test that CPU and GPU produce the same results."""
        if not self.has_gpu:
            self.skipTest("GPU not available, skipping CPU vs GPU consistency test")
        
        # Create 2D test input
        input_data = np.random.rand(10, 20).astype(np.float32)
        
        # Run on CPU
        with tf.device('/CPU:0'):
            input_tensor_cpu = tf.constant(input_data)
            result_cpu = self.example_module.example(input_tensor_cpu)
        
        # Run on GPU
        with tf.device('/GPU:0'):
            input_tensor_gpu = tf.constant(input_data)
            result_gpu = self.example_module.example(input_tensor_gpu)
        
        # Compare results
        np.testing.assert_array_almost_equal(result_cpu.numpy(), result_gpu.numpy(),
                                           err_msg="CPU and GPU results differ")


def main():
    """Run the tests."""
    # Check TensorFlow version and available devices
    tf_version = tf.__version__
    print(f"Running tests with TensorFlow version: {tf_version}")
    
    # Print available devices
    print("Available devices:")
    for device in tf.config.list_physical_devices():
        print(f"  {device}")
    
    # Configure GPU memory growth to avoid allocation issues
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU configuration warning: {e}")
    
    # Set the environment variable for GPU multiprocessor count if needed
    os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '4'
    
    # Determine which test suites to run
    test_suites = []
    
    # Always run CPU tests
    test_suites.append(unittest.TestLoader().loadTestsFromTestCase(TestExampleOpCPU))
    
    # Run GPU tests if GPU is available
    if tf.config.list_physical_devices('GPU'):
        print("GPU detected - including GPU tests")
        test_suites.append(unittest.TestLoader().loadTestsFromTestCase(TestExampleOpGPU))
        test_suites.append(unittest.TestLoader().loadTestsFromTestCase(TestExampleOpDeviceComparison))
    else:
        print("No GPU detected - running CPU tests only")
    
    # Always run general tests
    test_suites.append(unittest.TestLoader().loadTestsFromTestCase(TestExampleOpGeneral))
    
    # Combine all test suites
    combined_suite = unittest.TestSuite(test_suites)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    if result.wasSuccessful():
        print("\nAll tests passed!")
    else:
        print(f"\nTests failed: {len(result.failures)} failures, {len(result.errors)} errors")


if __name__ == '__main__':
    main()
