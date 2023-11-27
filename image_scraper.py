import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = 512

def linear_to_srgb(value):
    # Convert linear RGB values to sRGB format.
    mask = value <= 0.0031308
    value[mask] = value[mask] * 12.92
    value[~mask] = 1.055 * (value[~mask] ** (1.0 / 2.4)) - 0.055
    value = np.clip(value, 0.0, 1.0) * 255.0
    return value.astype(np.uint8)

def crop_center(img, cropx, cropy):
    # Crop the center of the image to the specified size.
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    return img[starty:starty+cropy, startx:startx+cropx]

def load_satellite_image(file_name):
    # Load and process a satellite image from a NetCDF file.
    data = xr.open_dataset(file_name)

    # Extract and scale RGB channels
    red = crop_center(np.array(data['B04'][0].as_numpy()), IMAGE_SIZE, IMAGE_SIZE) / 10000
    green = crop_center(np.array(data['B03'][0].as_numpy()), IMAGE_SIZE, IMAGE_SIZE) / 10000
    blue = crop_center(np.array(data['B02'][0].as_numpy()), IMAGE_SIZE, IMAGE_SIZE) / 10000
    
    # Stack channels and check for NaNs
    rgb = np.stack([red, green, blue], axis=-1)
    has_nans = np.isnan(rgb).any()     

    # Display the image
    plt.imshow(linear_to_srgb(rgb))
    plt.show()

    return rgb, has_nans

# Main script
if __name__ == "__main__":
    nc_files_path = 'C:/Users/vinicius23000/env-data-retrieval/**/*.nc'
    nc_files = glob.glob(nc_files_path, recursive=True)

    for file_name in nc_files:
        print("Processing:", file_name)
        rgb, has_nans = load_satellite_image(file_name)
        print("Contains NaNs:", has_nans)
