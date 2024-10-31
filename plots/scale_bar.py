import matplotlib.pyplot as plt
import numpy as np
# Function to plot an image with a signal intensity scale bar

def plot_image_with_scale_bar(image, save_path=None):
    # label_fontsize=14
    tick_fontsize = 20
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray', aspect='auto', vmin=0, vmax=200)  # Set scale from 0 to 200
    
    cbar = plt.colorbar()
    # cbar.set_label('Signal intensity (a.u.)', fontsize=label_fontsize)
    
    # Set the font size for the color bar ticks
    cbar.ax.tick_params(labelsize=tick_fontsize)
    
    # plt.title(title)
    plt.axis('off')  # Hide the axes
    
    if save_path:
        plt.savefig(save_path, format='eps', bbox_inches='tight', pad_inches=0.1)
    
    plt.show()


import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

colors = [
    (0.5, 0.0, 0.5),    # Purple
    (1.0, 0.0, 1.0),    # Magenta
    (1.0, 0.5, 0.0),    # Orange
    (1.0, 1.0, 0.0),    # Yellow
    (1.0, 1.0, 1.0)     # White
]
purple_white_colormap = LinearSegmentedColormap.from_list('purple_white', colors, N=256)

# Function to plot an image with a signal intensity scale bar using the NIH fire colormap
def plot_image_with_scale_bar_nihfire(image, save_path=None):
    tick_fontsize = 20
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='inferno', aspect='auto', vmin=0, vmax=350)
    # 100 para medula
    # 350 para cortezza
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=tick_fontsize)
    
    plt.axis('off')  # Hide the axes
    
    if save_path:
        plt.savefig(save_path, format='eps', bbox_inches='tight', pad_inches=0.1)
    
    plt.show()