import ctypes

# Load the shared library
# On Linux/macOS, use './libsquare.so'
# On Windows, use './square.dll'
lib = ctypes.CDLL('./square.so')

# Specify the argument and return types
lib.square.argtypes = [ctypes.c_int]
lib.square.restype = ctypes.c_int

# Call the function
result = lib.square(10)
print(f"The square of 10 is: {result}")
