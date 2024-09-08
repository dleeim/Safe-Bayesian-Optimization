import jax.numpy as jnp

def check_1d_array(array):
    if array.ndim != 1:
        raise ValueError("The input array must be 1-dimensional.")

# Example usage
input_array = jnp.array([1,2,3])  # 2D array

# Check if the array is 1D
check_1d_array(input_array)
