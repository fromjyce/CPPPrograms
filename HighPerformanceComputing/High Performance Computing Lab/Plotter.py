import pandas as pd
import matplotlib.pyplot as plt

# Read factorial times
factorial_data = pd.read_csv("factorial_times.csv")
plt.figure(figsize=(10, 6))
plt.plot(factorial_data['N'], factorial_data['Time (milliseconds)'], marker='o', linestyle='-', color='b', label='Factorial')
plt.xlabel('N')
plt.ylabel('Execution Time (milliseconds)')
plt.title('Execution Time vs. N for Factorial Calculation')
plt.legend()
plt.grid(True)
plt.show()

# Read Fibonacci times
fibonacci_data = pd.read_csv("fibonacci_times.csv")
plt.figure(figsize=(10, 6))
plt.plot(fibonacci_data['N'], fibonacci_data['Time (milliseconds)'], marker='o', linestyle='-', color='r', label='Fibonacci')
plt.xlabel('N')
plt.ylabel('Execution Time (milliseconds)')
plt.title('Execution Time vs. N for Fibonacci Calculation')
plt.legend()
plt.grid(True)
plt.show()

# Read matrix multiplication times
matrix_data = pd.read_csv("matrix_times.csv")
plt.figure(figsize=(10, 6))
plt.plot(matrix_data['Matrix Size'], matrix_data['Time (milliseconds)'], marker='o', linestyle='-', color='g', label='Matrix Multiplication')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Execution Time (milliseconds)')
plt.title('Execution Time vs. Matrix Size for Matrix Multiplication')
plt.legend()
plt.grid(True)
plt.show()
