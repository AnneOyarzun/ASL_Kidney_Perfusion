import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize, ListedColormap

def plot_rbf(masked_rbf, output_path, filename): 

    # # Rotate the image by -90 degrees and flip vertically
    # # Assuming img_data is a 3D array (x, y, z), we can select a slice
    # rotated_flipped_image = np.rot90(masked_rbf)  # Rotate by -90 degrees (k=1)
    # flipped_image = np.flipud(rotated_flipped_image)  # Flip vertically
    # flipped_image[flipped_image < 0] = 0

    masked_rbf = sitk.GetArrayFromImage(masked_rbf)

    masked_rbf[masked_rbf < 0] = 0
    masked_rbf[masked_rbf > 600] = 0

    # Create a custom colormap with black for 0 values
    # Get the inferno colormap and set 0 to black
    inferno = plt.get_cmap('inferno')
    colors = inferno(np.linspace(0, 1, 256))  # Get the full inferno colormap
    colors[0] = [0, 0, 0, 1]  # Set the first color (0 value) to black
    custom_cmap = ListedColormap(colors)

    # Plot the processed image using the inferno colormap
    plt.figure(figsize=(6, 6))
    plt.imshow(masked_rbf, cmap=custom_cmap, origin='lower')  # Use origin='lower' for medical images
    plt.colorbar()  # Add colorbar to show intensity range
    # Remove ticks from both axes
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis tick
    # Save the figure as EPS and PNG
    plt.savefig(output_path + '/' + filename + '.eps', format='eps')
    plt.savefig(output_path + '/' + filename + '.png', format='png')

    plt.show()