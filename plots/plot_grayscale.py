import SimpleITK as sitk
import matplotlib.pyplot as plt


def plot_grayscale(images, output_path, filename): 
    # Convert the image to an array
    images = sitk.GetArrayFromImage(images)

    # Plot the processed image using the grayscale colormap
    plt.figure(figsize=(6, 6))
    img = plt.imshow(images, cmap='gray', origin='lower', vmin=0, vmax=150)  # Use grayscale colormap and origin='lower' for medical images
    cbar = plt.colorbar(img)  # Add colorbar to show intensity range

    # Adjust the fontsize of the colorbar
    cbar.ax.tick_params(labelsize=22)  # Set fontsize for colorbar ticks
    
    # plt.colorbar()  # Add colorbar to show intensity range

    # Remove ticks from both axes
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks

    # Save the figure as EPS and PNG
    plt.savefig(output_path + '/' + filename + '.eps', format='eps')
    plt.savefig(output_path + '/' + filename + '.png', format='png')

    plt.show()
