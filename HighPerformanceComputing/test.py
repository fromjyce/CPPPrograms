import ctypes
import json
import time

# Load the shared library
# On Linux/macOS, use './libdijkstra_serial.so' or './libdijkstra_parallel.so'
# On Windows, use './dijkstra_serial.dll' or './dijkstra_parallel.dll'
lib_serial = ctypes.CDLL('./libdijkstra_serial.so')
lib_parallel = ctypes.CDLL('./libdijkstra_parallel.so')

# Define the argument and return types for the functions
lib_serial.parseGraph.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
lib_serial.dijkstra.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_longlong)]
lib_serial.dijkstra.restype = None

lib_parallel.dijkstra.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_longlong)]
lib_parallel.dijkstra.restype = None

def dijkstra(lib, graph_str, start_node):
    result = ctypes.create_string_buffer(1024)
    time_taken = ctypes.c_longlong()

    lib.dijkstra(graph_str.encode('utf-8'), start_node.encode('utf-8'), result, ctypes.byref(time_taken))
    
    return result.value.decode('utf-8'), time_taken.value

# Example usage
graph = {
    'A': [('B', 4), ('C', 5)],
    'B': [('A', 4), ('C', 11), ('D', 7)],
    'C': [('A', 5), ('B', 11), ('D', 3)],
    'D': [('B', 7), ('C', 3)]
}