import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad, dblquad
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm

def solve(N, show_matrix=False):
    dx, dy = 1, 10
    u = lambda x, y: np.cos(np.pi*x)*np.cos(np.pi*y)
    f = lambda x, y: (dx+dy)*np.pi**2*u(x, y)
    h = 1/N
    A = np.zeros((N*N, N*N))
    b = np.zeros((N*N))

    for j in range(0, N):
        for i in range(0, N):
            row = i*N+j
            b[row] = h**2 * f((i+0.5)*h, (j+0.5)*h)
            
            u_g_top = - np.cos(np.pi*(i+0.5)*h) 
            u_g_right = - np.cos(np.pi*(j+0.5)*h) 
            
            q_left = 0 
            q_bottom = 0 
            
            if i == 0 and j == 0:   #left N + bottom N
                A[row][(i+1)+j*N] = -dx
                A[row][i+j*N] = dx + dy
                A[row][i+(j+1)*N] = -dy
                b[row] += -q_left - q_bottom

            elif i == N-1 and j == N-1: #right D + top D
                A[row][(i-1)+j*N] = -dx
                A[row][i+j*N] = 3*dx + 3*dy
                A[row][i+(j-1)*N] = -dy
                b[row] += 2*dx*u_g_right + 2*dy*u_g_top
                
            elif i == 0 and j == N-1: #left N + top D
                A[row][(i+1)+j*N] = -dx
                A[row][i+j*N] = dx + 3*dy
                A[row][i+(j-1)*N] = -dy
                b[row] += -q_left + 2*dy*u_g_top

            elif i == N-1 and j == 0: #right D + bottom N
                A[row][(i-1)+j*N] = -dx
                A[row][i+j*N] = 3*dx + dy
                A[row][i+(j+1)*N] = -dy
                b[row] += 2*dx*u_g_right - q_bottom

            elif i == 0:            #left N
                A[row][(i+1)+j*N] = -dx
                A[row][i+j*N] = dx + 2*dy
                A[row][i+(j-1)*N] = -dy
                A[row][i+(j+1)*N] = -dy
                b[row] -= q_left

            elif j == 0:            #bottom N
                A[row][(i-1)+j*N] = -dx
                A[row][(i+1)+j*N] = -dx
                A[row][i+j*N] = 2*dx + dy
                A[row][i+(j+1)*N] = -dy
                b[row] -= q_bottom

            elif i == N-1:            #right D
                A[row][(i-1)+j*N] = -dx
                A[row][i+j*N] = 3*dx + 2*dy
                A[row][i+(j-1)*N] = -dy
                A[row][i+(j+1)*N] = -dy
                b[row] += 2*dx*u_g_right

            elif j == N-1:            #top D
                A[row][(i-1)+j*N] = -dx
                A[row][(i+1)+j*N] = -dx
                A[row][i+j*N] = 2*dx + 3*dy
                A[row][i+(j-1)*N] = -dy
                b[row] += 2*dy*u_g_top

            else:                   #other
                A[row][(i-1)+j*N] = -dx
                A[row][(i+1)+j*N] = -dx
                A[row][i+j*N] = 2*dx + 2*dy
                A[row][i+(j-1)*N] = -dy
                A[row][i+(j+1)*N] = -dy

    solution = spsolve(A, b)
   
    _, ax = plt.subplots(1, 3, figsize=(12, 4))
    x = y = np.arange(0+h/2, 1, h)
    xx, yy = np.meshgrid(x, y)
    ax[0].contourf(x, y, u(xx, yy), levels=15)
    ax[0].set_title(r'Solution $u(x,y)=\cos(\pi x)\cos(\pi y)$')
    
    ax[1].imshow(u(xx, yy), origin='lower')
    ax[1].set_title(r'Discretization of $u(x,y)$')
    
    ax[2].imshow(solution.reshape(N, N), origin='lower')
    ax[2].set_title('Approximate solution')
    plt.tight_layout()
    plt.show()
    print(f"C_err: {norm(u(xx, yy).flatten() - solution, ord=np.inf)}") 
    
    L_err = 0
    for i in range(N):
        for j in range(N):
            diff_vol = lambda x, y: (u(x,y) - solution[i + N * j]) ** 2
            L_err += dblquad(diff_vol, j * h, (j + 1) * h, i * h, (i + 1) * h)[0]
    L_err = np.sqrt(L_err)
    print("L_2_err: {}".format(L_err))
