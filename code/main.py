import sympy as sp

def schur(A: sp.Matrix, B: sp.Matrix) -> sp.Matrix:
    if A.shape != B.shape:
        raise ValueError("Matrices must have same shape.")
    return A.multiply_elementwise(B)

def square(A: sp.Matrix) -> sp.Matrix:
    m, n = A.shape
    C = np.zeros((m * m, n))

    l = 0
    for i in range(m):
        for j in range(m):
            C[l, :] = schur(A.row(i), A.row(j))  
            l += 1
    return C

def QCLD(L: int, A: sp.Matrix) -> sp.Matrix:
    P = sp.zeros(L)
    P[:L-1, 1:L] = sp.eye(L - 1)
    P[L-1, 0] = 1

    m, n = A.shape
    H = sp.zeros(m * L, n * L)

    for i in range(m):
        for j in range(n):
            exp = int(A[i, j])
            if exp == -1:
                block = sp.zeros(L)
            else:
                block = (P ** exp) % 2  
            H[i*L:(i+1)*L, j*L:(j+1)*L] = block

    return H



