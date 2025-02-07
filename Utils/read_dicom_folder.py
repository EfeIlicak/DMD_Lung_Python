import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pydicom
# import Utils.play_dicom_animation as pda

def find_tag(dataset, tag):
    """
    Recursively search for a tag in the DICOM dataset.
    
    Parameters:
    - dataset: pydicom Dataset object
    - tag: tuple of (group, element) of the tag
    
    Returns:
    - value of the tag if found, else None
    """
    if tag in dataset:
        return dataset[tag].value
    for elem in dataset:
        if elem.VR == 'SQ':  # If the element is a sequence
            for item in elem.value:
                result = find_tag(item, tag)
                if result is not None:
                    return result
    return None

def find_all_tags_list(datasets, tag):
    """
    Recursively search for all occurrences of a tag in a list of DICOM datasets.
    
    Parameters:
    - datasets: list of pydicom Dataset objects
    - tag: tuple of (group, element) of the tag
    
    Returns:
    - list of values of the tag if found, else an empty list
    """
    results = []
    
    def recursive_search(dataset, tag):
        if tag in dataset:
            results.append(dataset[tag].value)
        for elem in dataset:
            if elem.VR == 'SQ':  # If the element is a sequence
                for item in elem.value:
                    recursive_search(item, tag)
    
    for dataset in datasets:
        recursive_search(dataset, tag)
    
    return results
# Example usage: allScales = find_all_tags_list(dicom_alldata, (0x2005, 0x100B)) # Find all Scale Slopes


def convert_time_to_seconds(time_float):
    """
    Convert a float given as a string representing time in the format "HourHourMinuteMinuteSecondSecond" to seconds.

    Parameters:
    - time_float (float): The input time in the specified format.

    Returns:
    int: The total seconds.
    """
    # Convert to float
    time_float = float(time_float)
    
    # Extract hours, minutes, and seconds
    hours = int(time_float // 10000)
    minutes = int((time_float % 10000) // 100)
    seconds = int(time_float % 100)

    # Convert to seconds
    total_seconds = (hours * 3600) + (minutes * 60) + seconds

    return total_seconds

def read_dicom_files(dicom_folder_path):
    """ Read .IMA files in the given folder path & return a NumPy array and metadata """
   # dicom_files = [file for file in os.listdir(dicom_folder_path) if file.endswith('.IMA') ] 
    dicom_files = [file for file in os.listdir(dicom_folder_path) if file.endswith(('.IMA','.dcm'))]
    dicom_data = []
    dicom_metadata = []
    dicom_alldata = []

    for file_name in dicom_files:
        file_path = os.path.join(dicom_folder_path, file_name)
        dicom_dataset = pydicom.dcmread(file_path)

        # Extract pixel data
        dicom_array = dicom_dataset.pixel_array
        dicom_data.append(dicom_array)
        # dicom_data.append(dicom_dataset.pixel_array)
        dicom_alldata.append(dicom_dataset)
        
        # Extract metadata  
        metadata = {
            'FileName': file_name,
            'PixelSpacing': find_tag(dicom_dataset, (0x0028, 0x0030)), # Voxel size in mm
            'ImageRow': dicom_dataset[0x0028,0x0010].value,  # Rows in the image
            'ImageCol': dicom_dataset[0x0028,0x0011].value,  # Columns in the image
            'Slice Thickness': dicom_dataset[0x0018,0x0050].value, # Slice thickness in mm
            'Number of Frames' : dicom_dataset[0x0028,0x0008].value, # Number of Frames in the dynamic series 
            'Acquisition Duration': dicom_dataset[0x0018,0x9073].value, # Acquisition duration in seconds
            'Acquisition Number': find_tag(dicom_dataset, (0x0020, 0x0105)), # Acquisition number for the slice
            'Rescale Slope': find_tag(dicom_dataset, (0x0028, 0x1053)), # Rescale slope 
            'Rescale Intercept': find_tag(dicom_dataset, (0x0028, 0x1052)), # Rescale intercept
            'Scale Slope': find_tag(dicom_dataset, (0x2005, 0x100B)), # Scale slope 
            'Scale Intercept': find_tag(dicom_dataset, (0x2005, 0x100C)) # Scale intercept
        }
        dicom_metadata.append(metadata)

    # Change the shape of the array to (ImageRow, ImageCol, Coils, NumberOfImages)
    dicom_array = np.array(dicom_data)
    dicom_array = np.expand_dims(dicom_array, axis=-1)  # Add an extra axis at the end
    dicom_array = np.moveaxis(dicom_array, 0, -1)  # Move the first axis to the end

    return dicom_array, dicom_metadata, dicom_alldata

def select_folder():
    """ Select folder using GUI & return the path"""
    root = tk.Tk()
    root.withdraw()

    dicom_folder_path = filedialog.askdirectory(title="Select Folder Containing DICOM Files")
    if not dicom_folder_path:
        print("No folder selected. Exiting...")
        return None

    # Manually filter files with .IMA extension
    dicom_files = [file for file in os.listdir(dicom_folder_path) if file.endswith(('.IMA','.dcm'))]

    # Print the files
    for file in dicom_files:
        print(file)


    if not dicom_files:
        print("No DICOM files found in the selected folder. Exiting...")
        return None

    return dicom_folder_path

def main():
    dicom_folder_path = select_folder()

    if dicom_folder_path:
        dicom_array, dicom_metadata, dicom_alldata = read_dicom_files(dicom_folder_path)
        # print("Shape of the resulting NumPy array:", dicom_array.shape)

        #total_acquisition_time = convert_time_to_seconds(dicom_metadata[-1]["Acquisition Time"]) - convert_time_to_seconds(dicom_metadata[0]["Acquisition Time"])
        #Fs = float(dicom_metadata[-1]["Acquisition Number"]) / total_acquisition_time
        Fs = float(dicom_metadata[-1]["Acquisition Number"]) / dicom_metadata[-1]["Acquisition Duration"]
        print("Total Number of Images:", dicom_metadata[-1]["Acquisition Number"])
        #print("Total Acquisition Time:", total_acquisition_time, "s")
        print("Total Acquisition Time:", dicom_metadata[-1]["Acquisition Duration"], "s")
        total_acquisition_time = dicom_metadata[-1]["Acquisition Duration"]
        print("Fs:", Fs, "Images/s" )

        # Squeeze dicom_array to remove extra dimensions
        dicom_array = np.squeeze(dicom_array)
        print("Shape of the squeezed NumPy array:", dicom_array.shape)

        # # Permute dicom_array to (ImageRow, ImageCol, NumberOfImages)
        # dicom_array = np.transpose(dicom_array, (1, 2, 0))

        # print("Shape of the squeezed and reshaped NumPy array:", dicom_array.shape)

        # pda.play_dicom_animation(dicom_array)
        return Fs, total_acquisition_time, dicom_array, dicom_metadata, dicom_alldata, dicom_folder_path

if __name__ == "__main__":
    Fs, total_acquisition_time, dicom_array, dicom_metadata, dicom_alldata, dicom_folder_path = main()
    # Now you can use Fs, total_acquisition_time, dicom_array, and dicom_metadata as needed.
