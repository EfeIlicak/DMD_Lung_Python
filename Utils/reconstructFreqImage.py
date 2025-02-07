import numpy as np

def reconstructFreqImage(modeAmp, FreqComps, indxToUse):
    """
    Computes the reconstruction from given indices and their weights.
    This is Equation 6 in the manuscript.
    
    Parameters:
    modeAmp : array-like
        The mode amplitudes.
    FreqComps : array-like
        The frequency components.
    indxToUse : array-like
        Indices to use for reconstruction.
    
    Returns:
    img : array-like
        The reconstructed image.
    """
    sx, sy, _ = FreqComps.shape
    img_Components = np.zeros((sx, sy, len(indxToUse)))

    for idx, indx in enumerate(indxToUse):
        img_Components[:, :, idx] = np.abs(modeAmp[indx]) * np.abs(FreqComps[:, :, indx])

    img = (np.sum(img_Components, axis=2))
    return img
