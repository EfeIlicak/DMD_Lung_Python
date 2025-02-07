import os
import tempfile
import pydicom
import subprocess
import numpy as np
import read_dicom_folder

def normalize_images(dicom_array):
    """
    Normalize the image intensities for each time step for Siemens registration software
    """
    max_val = np.max(np.abs(dicom_array)) * 25 # Calculate the max of dicom_array and scale by 25   
    norm_array = np.abs(dicom_array) # This will be the normalized array
    print(type(norm_array)) # Print the type of norm_array
    print(type(max_val)) # Print the type of max_val
    print(max_val)
        
    for time_idx in range(dicom_array.shape[-1]):
        # Normalize each 3D matrix along the 4th dimension
        norm_array[:, :, :, time_idx] = norm_array[:, :, :, time_idx] / max_val

    return norm_array
    

def create_temp_folder():
    """
    Create a temporary folder and return its path.
    """
    temp_folder = tempfile.mkdtemp()
    return temp_folder

def write_dicom_files(moving_imgs, dicom_metadata, dicom_alldata, temp_folder_path):
    """
    Write the DICOM files to the temporary folder.
    """
    # Create a list of file names
    file_names = [f"Image_{idx}.IMA" for idx in range(moving_imgs.shape[-1])]

    # Create a list of file paths
    file_paths = [os.path.join(temp_folder_path, file_name) for file_name in file_names]

    for idx in range(moving_imgs.shape[-1]):
        
        # Get the original DICOM dataset
        dd = dicom_alldata[idx]      
        
        # Change the images with the normalized ones
        arr = moving_imgs[:, :, 0, idx]
        
        dd.PixelData = arr.tobytes()
        dd.save_as("temp.dcm")
        
        
        # Write the new DICOM files
        dd.save_as(file_paths[idx])
    return   

def siemens_reg(img_array):
    """
    Call the Siemens registration software to register the images
    """

    img_reg = img_array
    return img_reg


def main():
    # Call the read_dicom_folder function to read a folder containing DICOM files
    _, _, dicom_array, dicom_metadata, dicom_alldata = read_dicom_folder.main()
    
    # Normalize the images
    norm_img = normalize_images(dicom_array)
    
    # Create a temporary folder
    temp_folder_path = create_temp_folder()
    print(f"Temporary folder created at: {temp_folder_path}")
    
    # Write the normalized images to the temporary folder as DICOM files
    write_dicom_files(norm_img, dicom_metadata, dicom_alldata, temp_folder_path)
    print("Normalized images written to the temporary folder as DICOM files.")
    
    subprocess.run(r'C:\Users\efe_i\Documents\MATLAB\fMRLung\bin\fMRLung.exe', shell=True)

    # Register the images using Siemens registration software
    img_reg = siemens_reg(norm_img)
    
    return img_reg #return the registered image
if __name__ == "__main__":
    main()
