import numpy as np

# Create a NumPy array of type uint8
arr = np.array([250, 10], dtype=np.uint8)

# Cast the array to uint16
arr = arr.astype(np.int16)

# Add a value to the array that will not result in a wrap-around
arr = arr + 1000

# Check if any values in the array are greater than 255
mask = arr > 255

# Modify any values that are greater than 255 to be 255
arr[mask] = 255

# The value at index 0 should be 255 after the modification
print(arr[0])
