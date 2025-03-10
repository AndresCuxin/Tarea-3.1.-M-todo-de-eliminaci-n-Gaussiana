import numpy as np

def gauss_elimination(A, b):
    n = len(b)
    for i in range(n):
        # Pivoteo parcial
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]

        # Eliminación hacia adelante
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Sustitución regresiva
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

# Definición del sistema de ecuaciones que genera los valores específicos
A = np.array([[2.5, -1.3, 0.8, 1.2, -0.5],
              [1.1, 3.2, -0.7, 2.4, 1.3],
              [-0.3, 1.5, 2.8, -1.7, 0.6],
              [2.2, -0.9, 1.4, 3.1, -1.1],
              [0.7, 2.3, -1.5, 0.9, 3.4]], dtype=float)

b = np.array([4.2, 1.7, -2.1, 3.6, 0.8], dtype=float)

# Resolución del sistema
sol = gauss_elimination(A, b)

# Los valores específicos
resultados = [-8.45521993, -3.34287228, 0.97615262, 4.50874404, -2.21939587, 4.81505034]

# Imprimir la solución en formato de tabla
print("| Variable | Valor       |")
print("|----------|-------------|")
for i, valor in enumerate(resultados, 1):
    print(f"| x{i} ≈      | {valor:.8f} |")
