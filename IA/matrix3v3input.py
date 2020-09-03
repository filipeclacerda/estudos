import numpy as np

print("Insira os números com espaço entre eles: ")

# User input of entries in a
# single line separated by space
valores = list(map(int, input().split()))

# For printing the matrix
matrix = np.array(valores).reshape(3, 3)
print(matrix)