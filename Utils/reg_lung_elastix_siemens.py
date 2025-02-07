import os
import numpy as np
import Utils.read_dicom_folder_siemens as read_dicom_folder_siemens
import Utils.play_dicom_animation as pda
import itk
from itkwidgets import compare, checkerboard, view
from matplotlib import pyplot as plt
import cv2
import datetime
import h5py


def normalize_images(dicom_array):
    """
    Normalize the image intensities for each time step for Siemens registration software
    """
    
    max_val = float(np.max(np.abs(dicom_array)))  # Calculate the max of dicom_array 
    norm_array = np.float32(np.abs(dicom_array)) # This will be the normalized array. It has to be float type
    # print(type(norm_array)) # Print the type of norm_array
    # print(type(max_val)) # Print the type of max_val
    print("Normalizing series by max value of:", max_val)
        
    for time_idx in range(dicom_array.shape[-1]):
        # Normalize each 2D+t matrix along the 3rd (time) dimension
        norm_array[:, :, time_idx] = norm_array[:, :, time_idx] / max_val

    return norm_array
    

def elastix_reg(images):
    """
    Call the Elastix registration to register the images
    """
    # Create Groupwise Parameter Object
    parameter_object = itk.ParameterObject.New()
    groupwise_parameter_map = parameter_object.GetDefaultParameterMap('groupwise')
    #groupwise_parameter_map['Metric'] = ['PCAMetric2'] # Default VarianceOverLastDimensionMetric works better
    groupwise_parameter_map['GridSpacingSchedule'] = ('5.0', '3.0', '2.0', '1.0')
    groupwise_parameter_map['MaximumNumberOfIterations'] = ('100',)
    groupwise_parameter_map['MovingImageDerivativeScales'] = ('1', '1', '0')
    groupwise_parameter_map['ImagePyramidSchedule'] = ('8', '8', '0', '4', '4', '0', '2', '2', '0', '1', '1', '0')

    parameter_object.AddParameterMap(groupwise_parameter_map)

    print(parameter_object)

    # Call registration function
    # both fixed and moving image should be set with the vector_itk to prevent elastix from throwing errors

    result_image, result_transform_parameters = itk.elastix_registration_method(
    images, images, parameter_object=parameter_object,
    log_to_console=True)


    img_reg = result_image
    return img_reg

def generate_unique_filename(dicom_folder_path):
    """	
    Function to generate a unique file name
    """
    # Get the base name of the DICOM folder path
    base_name = os.path.basename(dicom_folder_path.rstrip(os.sep))
    # Get the current date and time
    current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    # Create a unique file name
    unique_filename = f"{base_name}_{current_datetime}.h5"
    return unique_filename

def reg_lung_elastix_siemens():
    # Call the read_dicom_folder function to read a folder containing DICOM files
    Fs, _, dicom_array, dicom_metadata, dicom_alldata, dicom_folder_path = read_dicom_folder_siemens.main()
    
    dicom_array = dicom_array.transpose(2,0,1)
    # Normalize the images
    norm_img = normalize_images(dicom_array)
    # pda.play_dicom_animation(norm_img)

    # Create a minimum intensity projection image
    mIP = np.min(norm_img,0)
    # plt.imshow(mIP, cmap='gray') # Display the minimum intensity projection image

    r = cv2.selectROI("select the area", mIP)     

    # Crop images
    cropped_image = norm_img[:,int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])] 
    # plt.imshow(cropped_image[0,:,:], cmap='gray')


    # Convert the normalized image to an ITK image
    images = itk.GetImageFromArray(cropped_image[:280,:,:]) # Get the constant rf phase part
   #  masks = itk.GetImageFromArray(np.tile(mask,(norm_img.shape[0],1,1)))
    cv2.destroyAllWindows()

    print("Starting Elastix registration...")
    # Register the images using Elastix
    img_reg = elastix_reg(images)
    img_reg_np = itk.GetArrayFromImage(img_reg) 
    truncated_img_reg = img_reg_np[10:,:,:] # Discard the first 10 frames due to transient effects

    # pda.play_dicom_animation(img_reg)
    
    h5_file_path = generate_unique_filename(dicom_folder_path)

    with h5py.File(h5_file_path, 'w') as h5f:
        # Save the truncated_img_reg array
        h5f.create_dataset('img_reg', data=truncated_img_reg)
        
        # Save the dicom_folder_path variable
        h5f.create_dataset('dicom_path', data=dicom_folder_path)

        # Save the Fs variable
        h5f.create_dataset('Fs', data=Fs)
                
    return truncated_img_reg, Fs, dicom_folder_path, dicom_metadata #return the registered image
