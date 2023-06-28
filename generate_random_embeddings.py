import numpy as np

def generate_random_vectors(k, d, file_path, min_val, max_val):
    vectors = np.random.uniform(min_val, max_val, (k, d))
    np.save(file_path, vectors)

# Usage example
k = 5  # Number of vectors
d = 3  # Dimension of vectors
file_path = "vectors.npy"  # File path to save the vectors
min_val = -100  # Minimum value for the range
max_val = 100  # Maximum value for the range
generate_random_vectors(k, d, file_path, min_val, max_val)
