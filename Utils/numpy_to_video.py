import cv2
import numpy as np

# Assuming dicom_array is already loaded with shape (230, 256, 256)
# Normalize the array to range 0-255 (required for saving as video)
dicom_array_normalized = ((dicom_array - dicom_array.min()) / (dicom_array.max() - dicom_array.min()) * 255).astype(np.uint8)

# Define the video file output and settings
output_filename = 'dicom_video.mp4'
fps = Fs * 2 # 2x the sampling frequency
frame_size = (dicom_array.shape[1], dicom_array.shape[2])  # 256x256

# Initialize VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify codec for mp4
video_writer = cv2.VideoWriter(output_filename, fourcc, fps, frame_size, isColor=False)

# Write frames to the video
for i in range(dicom_array.shape[0]):
    frame = dicom_array_normalized[i]
    video_writer.write(frame)

# Release the video writer object
video_writer.release()

print(f"Video saved as {output_filename}")
