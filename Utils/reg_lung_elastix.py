import os
import numpy as np
import Utils.read_dicom_folder as read_dicom_folder
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
    Call the Elastix registration to register the images using groupwise registration
    """
    # Create Groupwise Parameter Object
    parameter_object = itk.ParameterObject.New()
    groupwise_parameter_map = parameter_object.GetDefaultParameterMap('groupwise')
    #groupwise_parameter_map['Metric'] = ['PCAMetric2'] # Default VarianceOverLastDimensionMetric works better
    groupwise_parameter_map['GridSpacingSchedule'] = ('5.0', '3.0', '2.0', '1.0')
    groupwise_parameter_map['MaximumNumberOfIterations'] = ('250',)
    # groupwise_parameter_map['MovingImageDerivativeScales'] = ('1', '1', '0')
    groupwise_parameter_map['NumberOfResolutions'] = ('4',)
    groupwise_parameter_map['ImagePyramidSchedule'] = ('8', '8', '0', '4', '4', '0', '2', '2', '0', '1', '1', '0')

    parameter_object.AddParameterMap(groupwise_parameter_map)

    print(parameter_object)

    # Call registration function
    # both fixed and moving image should be set with the vector_itk to prevent elastix from throwing errors

    result_image, result_transform_parameters = itk.elastix_registration_method(
    images, images, parameter_object=parameter_object,
    log_to_console=True)

    # deformation_field = itk.transformix_deformation_field(images, result_transform_parameters)
    # plt.figure(),plt.imshow(deformation_field_a[0,:,:,0],cmaps["RdBu"]),plt.colorbar()
    
    img_reg = result_image
    return img_reg

def elastix_reg_slice(moving_images):
    """
    Call the Elastix registration to register the images using a target slice registration
    """
    # Create Parameter Object
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile('par000_mod.txt')
    print(parameter_object)

    # Calculate the average signal intensity of the moving images
    avg_intensity = np.mean(moving_images)
    # Find the slice that is closest to the average signal intensity
    slice_idx = np.argmin(np.abs(np.mean(moving_images, axis=(1,2)) - avg_intensity))
    # Set the target slice as the slice that is closest to the average signal intensity
    fixed_image = moving_images[slice_idx,:,:]

    # Create fixed images by repeating the fixed image along the time axis
    fixed_images = np.tile(fixed_image, (moving_images.shape[0],1,1))

    # print(f"Target slice index: {slice_idx}")
    # print(f"Fixed image shape: {fixed_image.shape}")
    # print(f"Fixed images shape: {fixed_images.shape}")
    # print(f"Moving image shape: {moving_images.shape}")
    # Call registration function
    # both fixed and moving image should be set with the vector_itk to prevent elastix from throwing errors
    result_image, result_transform_parameters = itk.elastix_registration_method(
    fixed_images, moving_images, parameter_object=parameter_object,
    log_to_console=True)

    # deformation_field = itk.transformix_deformation_field(images, result_transform_parameters)
    # plt.figure(),plt.imshow(deformation_field_a[0,:,:,0],cmaps["RdBu"]),plt.colorbar()
    
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

def reg_lung_elastix():
    # Call the read_dicom_folder function to read a folder containing DICOM files
    Fs, _, dicom_array, dicom_metadata, dicom_alldata, dicom_folder_path = read_dicom_folder.main()
    
    # Normalize the images
    norm_img = normalize_images(dicom_array[10:,:,:]) # Discard the first 10 frames due to transient effects and normalize the set of images
    # norm_img = normalize_images(dicom_array[:,:,:]) # Normalize the set of images
    # pda.play_dicom_animation(norm_img)

    # norm_img = norm_img ** 0.5 # Apply a square root to the normalized image to change contrast
    # Downsample the image by 2   
    # tmp_norm_img = np.zeros((norm_img.shape[0], norm_img.shape[1]*2, norm_img.shape[2]*2))
    # for i in range(norm_img.shape[0]):
    #     tmp_norm_img[i,:,:] = cv2.resize(norm_img[i,:,:], (norm_img.shape[2]*2, norm_img.shape[1]*2), interpolation = cv2.INTER_CUBIC)
    # norm_img = tmp_norm_img


    # Create a minimum intensity projection image
    mIP = np.min(norm_img,0)
    # plt.imshow(mIP, cmap='gray') # Display the minimum intensity projection image

    r = cv2.selectROI("select the area", mIP)     

    # Crop images
    cropped_image = norm_img[:,int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])] 
    # plt.imshow(cropped_image[0,:,:], cmap='gray')


    # Convert the normalized image to an ITK image
    images = itk.GetImageFromArray(cropped_image) 
   #  masks = itk.GetImageFromArray(np.tile(mask,(norm_img.shape[0],1,1)))
    cv2.destroyAllWindows()

    print("Starting Elastix registration...")
    # Register the images using Elastix
    # img_reg = elastix_reg(images) # Groupwise registration
    # truncated_img_reg = itk.GetArrayFromImage(img_reg) # Convert the registered image to a numpy array
    img_reg = elastix_reg_slice(images) # Target slice registration
    truncated_img_reg = np.asarray(img_reg).astype(np.float64)

    #truncated_img_reg = truncated_img_reg ** 2 # Apply a square to the registered image to change contrast
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


def reg_lung_elastix_numpy(img):
    # Call the read_dicom_folder function to read a folder containing DICOM files
    #Fs, _, dicom_array, dicom_metadata, dicom_alldata, dicom_folder_path = read_dicom_folder.main()
    
    # Normalize the images
    # norm_img = normalize_images(img[10:,:,:]) # Discard the first 10 frames due to transient effects and normalize the set of images
    norm_img = normalize_images(img[:,:,:]) # Normalize the set of images
    # pda.play_dicom_animation(norm_img)

    # norm_img = norm_img ** 0.5 # Apply a square root to the normalized image to change contrast
    # Downsample the image by 2   
    # tmp_norm_img = np.zeros((norm_img.shape[0], norm_img.shape[1]*2, norm_img.shape[2]*2))
    # for i in range(norm_img.shape[0]):
    #     tmp_norm_img[i,:,:] = cv2.resize(norm_img[i,:,:], (norm_img.shape[2]*2, norm_img.shape[1]*2), interpolation = cv2.INTER_CUBIC)
    # norm_img = tmp_norm_img


    # Create a minimum intensity projection image
    mIP = np.min(norm_img,0)
    # plt.imshow(mIP, cmap='gray') # Display the minimum intensity projection image

    r = cv2.selectROI("select the area", mIP)     

    # Crop images
    cropped_image = norm_img[:,int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])] 
    # plt.imshow(cropped_image[0,:,:], cmap='gray')


    # Convert the normalized image to an ITK image
    images = itk.GetImageFromArray(cropped_image) 
   #  masks = itk.GetImageFromArray(np.tile(mask,(norm_img.shape[0],1,1)))
    cv2.destroyAllWindows()

    print("Starting Elastix registration...")
    # Register the images using Elastix
    img_reg = elastix_reg(images)
    truncated_img_reg = itk.GetArrayFromImage(img_reg) # Convert the registered image to a numpy array

    #truncated_img_reg = truncated_img_reg ** 2 # Apply a square to the registered image to change contrast
    # pda.play_dicom_animation(img_reg)
    
    # h5_file_path = generate_unique_filename(dicom_folder_path)

    # with h5py.File(h5_file_path, 'w') as h5f:
    #     # Save the truncated_img_reg array
    #     h5f.create_dataset('img_reg', data=truncated_img_reg)
        
    #     # Save the dicom_folder_path variable
    #     h5f.create_dataset('dicom_path', data=dicom_folder_path)

    #     # Save the Fs variable
    #     h5f.create_dataset('Fs', data=Fs)
                
    return truncated_img_reg #return the registered image
