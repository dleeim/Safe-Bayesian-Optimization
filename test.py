import jax.numpy as jnp
from jax import jit, vmap

def seuclidean_distance(x, y, V):
    """
    Calculate the SEuclidean distance between two points x and y.
    
    Parameters:
    x (array): First n-dimensional point.
    y (array): Second n-dimensional point.
    V (array): Variance of each dimension.
    
    Returns:
    float: The SEuclidean distance between x and y.
    """
    return jnp.sum(((x - y) ** 2) / V)

# Vectorize the seuclidean_distance function to work on arrays of points
vectorized_distance = vmap(vmap(seuclidean_distance, (None, 0, None)), (0, None, None))

# Example usage:
points = jnp.array([[ 0.22637079,  0.95896278],
                    [ 0.31399522, -1.63230866],
                    [ 1.09072948,  0.64429404],
                    [-1.63109548,  0.02905184]])
V = jnp.array([[1.,1.]])  # Variance along each dimension

# JIT compile the vectorized function for better performance
vectorized_distance_jit = jit(vectorized_distance)

# Calculate the distance matrix
distance_matrix = vectorized_distance_jit(points, points, V)

print("SEuclidean distance matrix:\n", distance_matrix)

def squared_seuclidean_jax(X, Y, V):
    '''
    Description:
        Compute standardized euclidean distance between 
        each pair of the two collections of inputs.
    Arguments:
        X                       : input data in shape (nA, n)
        Y                       : input data in shape (nB, n)
        V                       : Variance vector
    Returns:
        dis_mat                 : matrix with elements as standardized euclidean distance 
                                    between two input data X and Y
    '''
    V_sqrt = V**-0.5
    X_adjusted = X * V_sqrt
    Y_adjusted = Y * V_sqrt

    # Compute pairwise squared SEuclidean distances
    dist_mat = jnp.sum((X_adjusted[:, None, :] - Y_adjusted[None, :, :])**2, axis=-1)
    
    return dist_mat

print(squared_seuclidean_jax(points,points,V))
