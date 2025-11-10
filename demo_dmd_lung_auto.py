## Script to do DMD automatically. 
# It selects breathing and cardiac frequencies using stable and large modes of DMD. 
# Still there are some things to do - especially for background offset calculating. As such it might not work well for slices
# where a large vessel is not visible. Also, 0.65 Hz (39 bpm) is used as the threshold frequency between respiration and cardiac pulsation.
#
# Efe Ilicak, 
# LKEB, LUMC.
# 10.11.2025.

import numpy as np
import cmasher as cmr # Colormaps
from matplotlib import pyplot as plt
from Utils.DynamicModeDecomp import DynamicModeDecomp
from Utils.reconstructFreqImage import reconstructFreqImage
from Utils.reg_lung_elastix import reg_lung_elastix_noSave
from skimage.measure import label   

def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC


SLIM = 0.83 # Eigenvalue stability limit for DMD
DCF = 0.05 # Setting frequencies below 0.05 Hz as DC term
VQT = 0.65 # Setting 0.65Hz as the transition frequency from ventilation to perfusion  (= 39 bpm and higher are considered cardiac frequencies)
MMAX = 1 # Magnitude max
VMAX = 0.4 # Ventilation max
QMAX = 0.4 # Perfusion max

### Load DICOM folder and Register Images via Elastix ###
img_reg, Fs, dicom_folder_path, dicom_metadata, cropped_image = reg_lung_elastix_noSave()

dt = 1 / Fs
img_reg = img_reg.transpose(1,2,0) # Set the temporal dimension to last

img_reg = img_reg[10:-10,10:-10,:] # Usually the corners are errorenous. We don't need them anyways

sx = np.shape(img_reg)[0]
sy = np.shape(img_reg)[1]
m = np.shape(img_reg)[2]

# Set NaNs to zero (if available)
img_reg[np.isnan(img_reg)] = 0

img_reg = (img_reg - np.min(img_reg)) / (np.max(img_reg) - np.min(img_reg))

# Remove background offset 
flat_mm = img_reg.flatten()
# plt.hist(flat_mm, bins=256)
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Count')
# plt.title('Histogram of Image')
# plt.show()
bg = np.percentile(flat_mm,1)
img_reg = img_reg - bg 

# Reshape the acquisitions
X = np.reshape(img_reg, (sx * sy, m))


# ## FFT Based peak detection ###
# t_signal = np.mean(X,axis=0)
# f_signal = np.fft.fftshift(np.fft.fft(t_signal))  # Fourier transform of the time signal (shifted)
# pwr = np.abs(f_signal)**2 / m  # Power (zero-centered)
# fshift = np.arange(-m/2, m/2) * (Fs/m)  # Frequency range (zero-centered)
# pp = pwr[m//2:]**(0.5)  # Calculate the magnitude
# freq = fshift[m//2:]
# pp[freq<0.05] = 0 # Ignore the very low frequency contents
# # plt.figure()
# # plt.plot(freq,pp)

# # Find ventilation frequency (< VQT Hz)
# mask_vent = freq < VQT
# ventFreq = freq[mask_vent][np.argmax(pp[mask_vent])]

# # Find perfusion frequency (>= VQT Hz)
# mask_perf = freq >= VQT
# perfFreq = freq[mask_perf][np.argmax(pp[mask_perf])]


### DMD ###
DMDvarin = {'r':15, 'dt': dt, 'nstacks': 5}
Phi_DMD, omega_DMD, lambda_DMD, b_DMD, freq_DMD, Xdmd_DMD, rDMD = DynamicModeDecomp(X, **DMDvarin)


## Find peak respiration and cardiac pulsation frequencies ##

# Get the filtered frequencies
stableFreqs = freq_DMD[(np.abs(lambda_DMD) > SLIM) & ((freq_DMD) > DCF)]

# Get the filtered b values (not sorted yet)
b_filtered = np.abs(b_DMD)[(np.abs(lambda_DMD) > SLIM) & ((freq_DMD) > DCF)]

# Get the indices that would sort b_filtered in ascending order
sort_indices = np.argsort(b_filtered)

# Sort stableFreqs using these indices
stableFreqs_sorted = stableFreqs[sort_indices]

# Cut-off at VQT Hertz, select the stable frequency with highest amplitude as the main peak
ventFreq = stableFreqs_sorted[stableFreqs_sorted<VQT][-1] 
perfFreq = stableFreqs_sorted[stableFreqs_sorted>=VQT][-1]

ventRange = [ventFreq-(0.35*ventFreq), ventFreq+(0.35*ventFreq)] # Allowing for plus/minus %35 deviation around the ventilation frequency
perfRange = [perfFreq-(0.2*perfFreq), perfFreq+(0.2*perfFreq)] # Allowing for plus/minus %20 deviation around the perfusion frequency
# Check for overlap and adjust if necessary
if ventRange[1] > perfRange[0] and perfRange[1] > ventRange[0]:  # Ranges overlap
    midpoint = (max(ventRange[0], perfRange[0]) + min(ventRange[1], perfRange[1])) / 2
    if ventFreq < perfFreq:
        ventRange[1] = midpoint
        perfRange[0] = midpoint
    else:
        perfRange[1] = midpoint
        ventRange[0] = midpoint

print(f"Respiration Range: {ventRange[0]:.3f}-{ventRange[1]:.3f} Hz // Cardiac Pulsation Range: {perfRange[0]:.3f}-{perfRange[1]:.3f} Hz ")

## Generate Results ##
res_DMD = np.reshape(Phi_DMD[:sx*sy, :], (sx, sy, rDMD))

for idx, (b_val,abs_val, freq_val) in enumerate(zip(np.abs(b_DMD),np.abs(lambda_DMD), np.abs(freq_DMD))):
    print(f"Index:\t {idx},\t b: {b_val:.4f},\t lambda: {abs_val:.4f},\t freq: {freq_val:.4f}")

# Find ventilation and perfusion indices
idxDC_DMD = np.where(np.abs(freq_DMD) < DCF )[0]
vent_DMD_idx = np.where((np.abs(freq_DMD) > ventRange[0]) & (np.abs(freq_DMD) < ventRange[1]) & (np.abs(lambda_DMD) > SLIM))[0]
perf_DMD_idx = np.where((np.abs(freq_DMD) > perfRange[0]) & (np.abs(freq_DMD) < perfRange[1]) & (np.abs(lambda_DMD) > SLIM))[0]

# Print the indices of the DC, ventilation and perfusion components
print(f"DC indices: {idxDC_DMD}")
print(f"Ventilation indices: {vent_DMD_idx}")
print(f"Perfusion indices: {perf_DMD_idx}")

dc_DMD = reconstructFreqImage(b_DMD, res_DMD, idxDC_DMD) 
vent_DMD = reconstructFreqImage(b_DMD, res_DMD, vent_DMD_idx)
perf_DMD = reconstructFreqImage(b_DMD, res_DMD, perf_DMD_idx)

ventp = np.percentile(vent_DMD, 99) 
vent_DMD[vent_DMD > ventp] = ventp

SImid = dc_DMD
SIexp = SImid + (vent_DMD/2)
SIinp = SImid - (vent_DMD/2)

ventMap = (SIexp - SIinp) / SIexp # Fractional Ventilation definition  Klimeš et al., https://doi.org/10.1002/nbm.4088 & Emami et al. https://doi.org/10.1002/mrm.22186 &  by Bauman et al., https://doi.org/10.1002/mrm.26096 (without BG) 

# Calculate weighted average ventilation frequency
# fVent = np.sum(np.abs(b_DMD[vent_DMD_idx])/np.sum(np.abs(b_DMD[vent_DMD_idx])) * np.abs(freq_DMD[vent_DMD_idx]))

perfMap = perf_DMD / dc_DMD # Correct the perfusion results for mean signal intensity
perfMap = perfMap / np.percentile(perfMap, 99) # Normalize the results based on 99th percentile of the maximum perfusion

# Calculate weighted average perfusion frequency
# fPerf = np.sum(np.abs(b_DMD[perf_DMD_idx])/np.sum(np.abs(b_DMD[perf_DMD_idx])) * np.abs(freq_DMD[perf_DMD_idx]))


### Display results ###
fig, axs = plt.subplots(1, 3, figsize=(19, 4))
im = axs[0].imshow(dc_DMD, cmap='gray', vmin=0, vmax=MMAX)
axs[0].set_title('DC Component [a.u.]')
axs[0].axes.get_xaxis().set_ticks([])
axs[0].axes.get_yaxis().set_ticks([])
cax = fig.add_axes([axs[0].get_position().x1+0.01,axs[0].get_position().y0,0.02,axs[0].get_position().height])
tick_values = np.linspace(0, MMAX, 6) 
colorbar = plt.colorbar(im, cax=cax)
colorbar.set_ticks(tick_values)

im = axs[1].imshow(ventMap, cmap=cmr.freeze, vmin=0, vmax=VMAX)
axs[1].set_title('Fractional Ventilation [a.u.]')
axs[1].axes.get_xaxis().set_ticks([])
axs[1].axes.get_yaxis().set_ticks([])
cax = fig.add_axes([axs[1].get_position().x1+0.01,axs[1].get_position().y0,0.02,axs[1].get_position().height])
tick_values = np.linspace(0, VMAX, 6) 
colorbar = plt.colorbar(im, cax=cax)
colorbar.set_ticks(tick_values)

im = axs[2].imshow(perfMap, cmap=cmr.sunburst, vmin=0, vmax=QMAX)
axs[2].set_title('Normalized Perfusion [a.u.]')
axs[2].axes.get_xaxis().set_ticks([])
axs[2].axes.get_yaxis().set_ticks([])
cax = fig.add_axes([axs[2].get_position().x1+0.01,axs[2].get_position().y0,0.02,axs[2].get_position().height])
# plt.colorbar(im, cax=cax)
tick_values = np.linspace(0, QMAX, 6) 
colorbar = plt.colorbar(im, cax=cax)
colorbar.set_ticks(tick_values)
plt.show()
