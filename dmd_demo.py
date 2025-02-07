import mat73 # Load MatLab v7.3 files
import cmasher as cmr # Colormaps
import numpy as np
import matplotlib.pyplot as plt
from Utils.DynamicModeDecomp import DynamicModeDecomp
from Utils.reconstructFreqImage import reconstructFreqImage
from Utils.play_dicom_animation import play_dicom_animation

# Load the mat file
mat_contents = mat73.loadmat("LungPhantom.mat")
backGround = mat_contents['backGround']
largeVesselsCor = mat_contents['largeVesselsCor']
lungParencCor = mat_contents['lungParencCor']
sx = mat_contents['sx']
sy = mat_contents['sy']

# Access freeze and sunburst colormaps
ventColorMap = cmr.freeze      
perfColorMap = cmr.sunburst  

# Simulate phantom for m=480 and add noise
# Set random seed for repeatability
np.random.seed(160123)

# Parameters for modified Lujan formula from Bauman & Bieri (10.1002/mrm.26096)
A_DC = 80
A_R = 16
A_C = 4
phase_r = np.pi / 4
phase_c = np.pi / 2

dt = 0.25
T = 120
m = int(T / dt)

t = np.arange(dt, T + dt, dt)
Fs = 1 / dt

f_Resp_start = 0.2
f_Resp_end = 0.18
f_Card_start = 1.0
f_Card_end = 0.9

tt = int(5 / dt)
tstr = (m - tt) // 2
tend = tstr + tt


omega_r = 2 * np.pi * np.concatenate((
    np.full(tstr, f_Resp_start),
    np.linspace(f_Resp_start, f_Resp_end, tt),
    np.full(m - tend, f_Resp_end)
))
omega_c = 2 * np.pi * np.concatenate((
    np.full(tstr, f_Card_start),
    np.linspace(f_Card_start, f_Card_end, tt),
    np.full(m - tend, f_Card_end)
))

backDens = 1
lungDens = 0.2
vessDens = 0.9


# Create a combined segmentation map
segMap = backGround + 2 * lungParencCor + 3 * largeVesselsCor
sx, sy = segMap.shape

# Simulate noiseless phantom for m=480
noiselessPhantom = np.zeros((sx, sy, m))
# Loop through each voxel (xIdx, yIdx)
for xIdx in range(sx):
    for yIdx in range(sy):
        if segMap[xIdx, yIdx] == 1:  # Background. Only DC
            for nIdx in range(m):
                noiselessPhantom[xIdx, yIdx, nIdx] = backDens * A_DC
        elif segMap[xIdx, yIdx] == 2:  # Lung Parenchyma. DC + Vent + Perf
            for nIdx in range(m):
                term_r = A_R * np.cos(nIdx * (omega_r[nIdx] * dt / 2) + phase_r)**2
                term_c = A_C * np.cos(nIdx * (omega_c[nIdx] * dt / 2) + phase_c)**2
                noiselessPhantom[xIdx, yIdx, nIdx] = lungDens * (A_DC - term_r + term_c)
        elif segMap[xIdx, yIdx] == 3:  # Large Vessels. DC + Perf
            for nIdx in range(m):
                term_c = A_C * np.cos(nIdx * (omega_c[nIdx] * dt / 2) + phase_c)**2
                noiselessPhantom[xIdx, yIdx, nIdx] = vessDens * (A_DC + term_c)


# Visualize noiseless phantom (uncomment the next line if you want to display the video)
# plt.imshow(noiselessPhantom[:, :, 0] / 80, cmap='gray'); plt.show()

# Create noise patterns and noisy phantom
snr = 50
noiseLevel = np.max(np.abs(noiselessPhantom[94, 94, :])) / snr / np.sqrt(2)

# Generate complex Gaussian noise
noisePattern = noiseLevel * (np.random.randn(sx, sy, m) + 1j * np.random.randn(sx, sy, m))
phantom = noiselessPhantom + np.abs(noisePattern)

# Visualize phantom (uncomment the next line if you want to display the video)
# plt.imshow(phantom[:, :, 0] / 80, cmap='gray'); plt.show()

# Dynamic Mode Decomposition (DMD)
X = np.reshape(phantom, (sx * sy, m))

stackNum = 5
DMDvarin = {'dt': dt, 'nstacks': stackNum}
Phi_DMD, omega_DMD, lambda_DMD, b_DMD, freq_DMD, Xdmd_DMD, rDMD = DynamicModeDecomp(X, **DMDvarin)

res_DMD = np.reshape(Phi_DMD[:sx*sy, :], (sx, sy, rDMD))

for idx, (b_val,abs_val, freq_val) in enumerate(zip(np.abs(b_DMD),np.abs(lambda_DMD), np.abs(freq_DMD))):
    print(f"Index:\t {idx},\t b: {b_val:.4f},\t lambda: {abs_val:.4f},\t freq: {freq_val:.4f}")

# Find ventilation and perfusion related signal changes via their frequencies
ventRange = [0.05, 0.35]
perfRange = [0.75, 1.25]
idxDC_DMD = np.where(np.abs(freq_DMD) < 5e-4 )[0]
vent_DMD_idx = np.where((np.abs(freq_DMD) > ventRange[0]) & (np.abs(freq_DMD) < ventRange[1]))[0]
perf_DMD_idx = np.where((np.abs(freq_DMD) > perfRange[0]) & (np.abs(freq_DMD) < perfRange[1]) & (np.abs(lambda_DMD) > 0.80))[0]


dc_DMD = reconstructFreqImage(b_DMD, res_DMD, idxDC_DMD)
vent_DMD = reconstructFreqImage(b_DMD, res_DMD, vent_DMD_idx)
perf_DMD = reconstructFreqImage(b_DMD, res_DMD, perf_DMD_idx)

# Generate functional maps
BGr = dc_DMD[:30, :30]
BG = np.std(BGr)
# BG = np.percentile(dc_DMD, 3)
ventMap = np.abs((vent_DMD) / ((vent_DMD / 2) + dc_DMD - BG))

perfp = np.percentile(perf_DMD, 99)
perf_DMD[perf_DMD > perfp] = perfp
perfMap = perf_DMD / perfp

# Display results
ventColorMap = cmr.freeze  # Ventilation colormap
perfColorMap = cmr.sunburst  # Perfusion colormap

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

im = axs[0, 0].imshow(phantom[:, :, 0], cmap='gray')
axs[0, 0].set_title('Phantom [a.u.]')
plt.colorbar(im, ax=axs[0, 0])

im = axs[0, 1].imshow(dc_DMD, cmap='gray')
axs[0, 1].set_title('DC Component [a.u.]')
plt.colorbar(im, ax=axs[0, 1])

im = axs[1, 0].imshow(ventMap, cmap=ventColorMap, vmin=0, vmax=0.2)
axs[1, 0].set_title('Fractional Ventilation [ml/ml]')
plt.colorbar(im, ax=axs[1, 0])

im = axs[1, 1].imshow(perfMap, cmap=perfColorMap, vmin=0, vmax=0.4)
axs[1, 1].set_title('Perfusion [normalized]')
plt.colorbar(im, ax=axs[1, 1])

plt.show()