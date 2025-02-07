import numpy as np
from scipy.linalg import svd, eig

def DynamicModeDecomp(X, dt=1, r=15, nstacks=5):
    """
    Computes the Dynamic Mode Decomposition of data X.
    
    Parameters:
    X (numpy.ndarray): State snapshots; columns are state snapshots, rows are measurements.
    dt (float, optional): Time step between snapshots. Default is 1.
    r (float, optional): Truncate to rank-r. Default is 10 (no truncation).
    nstacks (int, optional): Number of stacks of the raw data. Default is 5.

    Returns:
    tuple: (Phi, omega, lambda_, b, freq, Xdmd, r)
        Phi (numpy.ndarray): The DMD modes.
        omega (numpy.ndarray): The continuous-time DMD eigenvalues.
        lambda_ (numpy.ndarray): The discrete-time DMD eigenvalues.
        b (numpy.ndarray): A vector of magnitudes of modes Phi.
        freq (numpy.ndarray): The estimated frequencies in Hertz.
        Xdmd (numpy.ndarray): The data matrix reconstructed by Phi, omega, b.
        r (int): The truncation rank used.

    References:
    [1] Kutz, J. N., Brunton, S. L., Brunton, B. W., & Proctor, J. L. (2016). 
    "Dynamic Mode Decomposition." Society for Industrial and Applied 
    Mathematics. https://doi.org/10.1137/1.9781611974508

    [2] Ilicak, E, Ozdemir, S, Zapp, J, Schad, LR, Zöllner, FG. (2023).
    "Dynamic mode decomposition of dynamic MRI for assessment of 
    pulmonary ventilation and perfusion." Magnetic Resonance in Medicine.
    https://doi.org/10.1002/mrm.29656

    Note: This code is based on DMDfull.m provided by the authors in [1]. 
    Changes are made to include the addition of frequency output and time
    range change in the calculation of Xdmd. Default parameters are also 
    altered for functional lung imaging.

    Efe Ilicak, 08/07/2024
    """
    
    # Stacking the data matrix
    if nstacks > 1:

        Xaug = []
        for st in range(nstacks):
            Xaug.append(X[:, st:X.shape[1]-nstacks+st+1])
        Xaug = np.vstack(Xaug)
        np.shape(Xaug)

        X1 = Xaug[:, :-1]
        X2 = Xaug[:, 1:]
    else:
        X1 = X[:, :-1]
        X2 = X[:, 1:]
    
    M = X1.shape[1]

    # DMD
    U, S, Vh = svd(X1, full_matrices=False) # Economy SVD
    V = Vh.T

    if r is None or r > len(S):
        r = len(S)
    elif r < len(S):
        U = U[:, :r]
        S = S[:r]
        V = V[:, :r]
    S_r = np.diag(S)

    Atilde = U.T @ X2 @ V @ np.linalg.inv(S_r)
    D, W = eig(Atilde) # Eigenvalues and eigenvectors (Matlab gives outputs in reverse order)
    Phi = X2 @ V @ np.linalg.inv(S_r) @ W
    

    lambda_ = D
    omega = np.log(lambda_) / dt

    # Compute DMD mode amplitudes
    x1 = X1[:, 0]
    b = np.linalg.lstsq(Phi, x1, rcond=None)[0]

    # Compute the frequencies
    freq = np.angle(lambda_) / (2 * np.pi * dt)

    # DMD reconstructions
    time_dynamics = np.zeros((r, M), dtype=complex)
    t = np.arange(M) * dt
    for iter in range(M):
        time_dynamics[:, iter] = b * np.exp(omega * t[iter])
    
    Xdmd = Phi @ time_dynamics

    return Phi, omega, lambda_, b, freq, Xdmd, r