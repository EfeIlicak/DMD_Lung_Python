import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from IPython.display import display, HTML

matplotlib.rcParams['animation.embed_limit'] = 2**10  # Set to a larger value in MB if needed

def play_dicom_animation(dicom_array):
    """
    Play a DICOM animation and navigate frames interactively using buttons.

    Parameters:
    - dicom_array (numpy.ndarray): The NumPy array containing DICOM frames with frames x rows x columns.

    Returns:
    None
    """
    # Initialize variables for frame index
    current_frame = 0

    # Function to update the animation frames
    def update(frame):
        plt.clf()
        plt.imshow(dicom_array[frame, :, :], cmap='gray')
        plt.title(f"Frame {frame}")
        plt.axis('off')

    # Function to handle button click for moving forward
    def next_frame(event):
        nonlocal current_frame
        current_frame = (current_frame + 1) % dicom_array.shape[0]
        update(current_frame)

    # Function to handle button click for moving backward
    def prev_frame(event):
        nonlocal current_frame
        current_frame = (current_frame - 1) % dicom_array.shape[0]
        update(current_frame)

    # Create the animation
    fig, ax = plt.subplots()
    animation = FuncAnimation(fig, update, frames=dicom_array.shape[0], interval=50)

    # Create navigation buttons
    ax_next = plt.axes([0.8, 0.01, 0.1, 0.05])
    ax_prev = plt.axes([0.1, 0.01, 0.1, 0.05])

    btn_next = Button(ax_next, 'Next')
    btn_prev = Button(ax_prev, 'Previous')

    # Connect button clicks to functions
    btn_next.on_clicked(next_frame)
    btn_prev.on_clicked(prev_frame)

    # Display the animation in the notebook
    display(HTML(animation.to_jshtml()))

# Example usage:
# play_dicom_animation(your_dicom_array)
