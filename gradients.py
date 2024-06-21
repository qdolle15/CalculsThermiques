import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, simps, trapz
import cProfile
import pstats
from joblib import Parallel, delayed

fac=24/640  # 640 pixels voient 24mm
raw_arr=np.load("./data/DL.npy")  # 640 colonnes, 512 lignes
thermo=np.load("./data/thermo.npy")
T_data=raw_arr
T_data[0,:4]= 0

Nx, Ny = 30, 30  # Nombres d'harmoniques

Ni, Nj = T_data.shape
Ly, Lx = np.asarray(T_data.shape)*fac  # Champs de vue de la caméra en [mm]

# Symétrisation des données
T_sym = np.vstack((
    np.hstack((np.fliplr(T_data), T_data)),
    np.flipud(np.hstack((np.fliplr(T_data), T_data)))
    ))
plt.imshow(T_sym);plt.colorbar();plt.show()

def coefficients_Fourier(i, j):

    x = np.linspace(-Lx, Lx, T_sym.shape[1])
    y = np.linspace(-Ly, Ly, T_sym.shape[0])
    X, Y = np.meshgrid(x, y)
    # Coefficient de symétrie
    eta_ij = 1 
    if i == 0 and j == 0:
        eta_ij = 1 / 4
    elif i != j and i * j == 0:
        eta_ij = 1 / 2

    # Matrice des intégrandes
    cos_i = np.cos(i * np.pi * X / Lx)
    cos_j = np.cos(j * np.pi * Y / Ly)
    sin_i = np.sin(i * np.pi * X / Lx)
    sin_j = np.sin(j * np.pi * Y / Ly)
    
    Aij = simps(simps(T_sym * cos_j * cos_i, x), y) * (eta_ij / (Lx * Ly))
    Bij = simps(simps(T_sym * cos_j * sin_i, x), y) * (eta_ij / (Lx * Ly))
    Cij = simps(simps(T_sym * sin_j * cos_i, x), y) * (eta_ij / (Lx * Ly))
    Dij = simps(simps(T_sym * sin_j * sin_i, x), y) * (eta_ij / (Lx * Ly))
    
    return Aij, Bij, Cij, Dij


def Fourier_map(raw_map):

    x = np.linspace(-Lx, Lx, raw_map.shape[1])
    y = np.linspace(-Ly, Ly, raw_map.shape[0])
    X, Y = np.meshgrid(x, y)
    sol = np.zeros_like(raw_map, dtype=np.float64)
    cpt=0 

    for hi in range(Nx):
        for hj in range(Ny):

            Aij, Bij, Cij, Dij = coefficients_Fourier(hi, hj)

            print(np.round(100*cpt/(Nx*Ny),1))
            cpt+=1
        
            term1 = Aij*np.cos(hi*X*np.pi/Lx)*np.cos(hj*Y*np.pi/Ly)
            term2 = Bij*np.cos(hi*X*np.pi/Lx)*np.sin(hj*Y*np.pi/Ly)
            term3 = Cij*np.cos(hi*X*np.pi/Lx)*np.sin(hj*Y*np.pi/Ly)
            term4 = Dij*np.sin(hi*X*np.pi/Lx)*np.sin(hj*Y*np.pi/Ly)
            
            sol+=term1+term2+term3+term4
    
    return sol


raw_map=np.copy(T_sym)
cProfile.run('Fourier_map(raw_map)', 'profil_stats')
# Charger les statistiques du fichier de profil
stats = pstats.Stats('profil_stats')

# Afficher les statistiques triées par temps cumulé
stats.sort_stats('cumulative').print_stats()

raw_map=np.copy(T_sym)
a = Fourier_map(raw_map)
plt.imshow(a);plt.colorbar();plt.show()



def solve_regul_strong_equation(i, j, alpha):

    Aij, Bij, Cij, Dij = coefficients_Fourier(i,j)
    M = np.array([
        [1 + alpha * (i * np.pi/Lx)**2, 0, 0, 0, 0, 0, 0, -alpha*i*j*np.pi**2/(Lx*Ly)],
        [0, 1 + alpha * (i * np.pi/Lx)**2, 0, 0, 0, 0, -alpha*i*j*np.pi**2/(Lx*Ly), 0],
        [0, 0, 1 + alpha * (i * np.pi/Lx)**2, 0, 0, -alpha*i*j*np.pi**2/(Lx*Ly), 0, 0],
        [0, 0, 0, 1 + alpha * (i * np.pi/Lx)**2, -alpha*i*j*np.pi**2/(Lx*Ly), 0, 0, 0],
        [0, 0, 0, -alpha*i*j*np.pi**2/(Lx*Ly), 1 + alpha * (j * np.pi/Ly)**2, 0, 0, 0],
        [0, 0, -alpha*i*j*np.pi**2/(Lx*Ly), 0, 0, 1 + alpha * (j * np.pi/Ly)**2, 0, 0],
        [0, -alpha*i*j*np.pi**2/(Lx*Ly), 0, 0, 0, 0, 1 + alpha * (j * np.pi/Ly)**2, 0],
        [-alpha*i*j*np.pi**2/(Lx*Ly), 0, 0, 0, 0, 0, 0, 1 + alpha * (j * np.pi/Ly)**2],
    ])  

    right_side = np.array([
        i*np.pi/Lx*Cij,
        i*np.pi/Lx*Dij,
        -i*np.pi/Lx*Aij,
        -i*np.pi/Lx*Bij,
        j*np.pi/Ly*Bij,
        -j*np.pi/Ly*Aij,
        j*np.pi/Ly*Dij,
        -j*np.pi/Ly*Cij,
    ])

    # Résoudre le système d'équations M.x = right_side
    x = np.linalg.solve(M, right_side)

    return x


