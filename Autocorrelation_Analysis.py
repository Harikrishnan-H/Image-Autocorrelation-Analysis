import cv2 as cv                   
import numpy as np                 
import os                          
import time                        
import matplotlib.pyplot as plt # for plotting and saving autocorrelation plots
import pandas as pd    # for creating  dataframes and saving runtime data

def load_images_from_folder(folder, image_size=(256, 256)):
    """
    Load, convert to grayscale, and resize images from a specified folder.
    Only files with .jpg, .jpeg, or .png extensions are considered.
    Args:
        folder (str): The path to the directory containing the images.
        image_size (tuple): The target size (width, height) to resize images to.

    Returns:
        list: A list of NumPy arrays, where each array represents a loaded, grayscaled, and resized image.

    Raises:
        FileNotFoundError: If the specified folder does not exist.
        ValueError: If no valid images (.jpg, .jpeg, .png) are found or loaded from the folder.
    """

    # Initialize an empty list to store the loaded images
    images = []  
    
    # Check if the provided folder path exists if not raise an error
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Error: The folder '{folder}' does not exist.")


    # Get all image filenames
    filenames = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Sort filenames numerically
    filenames = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))

    # Iterate through images
    for filename in filenames:
        filepath = os.path.join(folder, filename)
        img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)

        # Check if the image was successfully loaded
        if img is not None:
            try:
                    # Resize the image to the specified dimensions
                    img = cv.resize(img, image_size)
                    # Append the successfully loaded and resized image as a NumPy array to the list
                    images.append(img)
            except Exception as e:
                    # If resizing fails, print an error message
                    print(f"Error while resizing image: {filename}")
                    print(f"Reason: {e}")
        else:
                # If cv.imread failed, print a warning message
                print(f"Warning: Could not read image: {filename}")

    # check if any images were actually loaded
    if len(images) == 0:
        #If the list is empty, raise an error
        raise ValueError("Error: No valid images were loaded from the folder.")

    # Return the list of loaded image arrays
    return images

def calculate_mean_variance(image):
    """
    Calculates the mean pixel value and variance of an image.
    Args:
        image (np.ndarray): The input image as a NumPy array.
    Returns:
        tuple: A tuple containing the mean and variance (mean, variance).
    """
    mean = np.mean(image)
    variance = np.var(image)
    # Return the calculated mean and variance
    return mean, variance

def autocorrelation_using_loops(image, max_dist):
    """
    Compute normalized autocorrelation for x and y directions using nested loops.
    Handles boundaries using mirror padding or the reflect mode).

    Args:
        image (np.ndarray):The grayscale input image.
        max_dist (int): The maximum distance (lag) to compute autocorrelation for.

    Returns:
        np.ndarray: A (max_dist, 2) array where column 0 is autocorrelation in x-direction
                    and column 1 is autocorrelation in y-direction for distances 1 to max_dist.
                    Returns NaNs if calculation is not possible.
    """
    # Convert image data type to float32 to prevent potential overflows during calculations
    image = image.astype(np.float32)
    # Get the dimensions (height, width) of the image
    height, width = image.shape

    # Pad the image boundaries using reflection
    #The padding width is 'max_dist' on all sides to accommodate shifts up to max_dist
    padded = np.pad(image, max_dist, mode='reflect')
    # Calculate the mean and variance
    mean, variance = calculate_mean_variance(image)

    autocorrelation = np.zeros((max_dist, 2))

    # Check for zero variance. Autocorrelation is undefined in case of that.
    if variance == 0:
        return np.ones((max_dist, 2))

    # Loop through distances or lags from 1 up to max_dist
    for d in range(1, max_dist + 1):
        sum_x = 0
        sum_y = 0
        #Counters for the number of pixel pairs involved in the sum for normalization
        count_x = 0
        count_y = 0

        #Calculate autocorrelation in X-direction
        # Iterate over each row or y-coordinate of the original image
        for y in range(height):
            # Iterate over each column or the x-coordinate
            # stopping d pixels early to avoid going out of bounds for the shifted pixel
            for x in range(width - d):
                #(pixel1 - mean) * (pixel2 - mean)
                sum_x += (image[y, x] - mean) * (padded[y + max_dist, x + max_dist + d] - mean)
                # Increment the counter for the number of pairs summed
                count_x += 1

        # Calculate autocorrelation in Y-direction
        # Iterate over each column or the x-coordinate
        for x in range(width):
            # Iterate over each row or y-coordinate and stopping d pixels early
            for y in range(height - d):
                #(pixel1 - mean) * (pixel2 - mean)
                sum_y += (image[y, x] - mean) * (padded[y + max_dist + d, x + max_dist] - mean)
                # Increment the counter for the number of pairs summed
                count_y += 1

        # Normalize the sums to get the autocorrelation
        # Check if count_x is positive and variance is positive
        autocorrelation[d - 1, 0] = sum_x / (count_x * variance) if count_x > 0 and variance > 0 else np.nan
        autocorrelation[d - 1, 1] = sum_y / (count_y * variance) if count_y > 0 and variance > 0 else np.nan

    #Return the autocorrelation values for distances 1 to max_dist
    return autocorrelation

def autocorrelation_using_numpy(image, max_dist):
    """
    Compute normalized autocorrelation using NumPy.
    Handles boundaries using mirror padding or reflect mode.

    Args:
        image (np.ndarray): The grayscale input image.
        max_dist (int): The maximum distance (lag) to compute autocorrelation for.

    Returns:
        np.ndarray: A (max_dist, 2) array where column 0 is autocorrelation in x-direction
                    and column 1 is autocorrelation in y-direction for distances 1 to max_dist.
                    Returns NaNs if calculation is not possible.

    """
    # Convert image data type to float32 to prevent potential overflows during calculations
    image = image.astype(np.float32)
    height, width = image.shape
    # Pad the image boundaries using reflection
    #The padding width is 'max_dist' on all sides to accommodate shifts up to max_dist
    padded = np.pad(image, max_dist, mode='reflect')
    # Calculate the mean and variance
    mean, variance = calculate_mean_variance(image)
    autocorrelation = np.zeros((max_dist, 2))

    # Check for zero variance. Autocorrelation is undefined in case of that.
    if variance == 0:
         # Return an array of ones, indicating perfect correlation
        return np.ones((max_dist, 2))

    # Pre-subtract the mean from the padded image for efficiency in the loop
    # This creates a 'centered' version of the padded image
    padded_centered = padded - mean

    # Loop through distances or lags from 1 up to max_dist
    for d in range(1, max_dist + 1):
         # Define the base region in the centered padded image corresponding to the original image's location
        base = padded_centered[max_dist:max_dist + height, max_dist:max_dist + width]
        # Define the region shifted by d pixels in the x-direction
        # It starts d columns to the right of the base region's start
        shifted_x = padded_centered[max_dist:max_dist + height, max_dist + d:max_dist + d + width]
        # Define the region shifted by d pixels in the y-direction
        # It starts d rows below the base region's start
        shifted_y = padded_centered[max_dist + d:max_dist + d + height, max_dist:max_dist + width]

        # Calculate the autocorrelation in X-direction using NumPy
        # Select the overlapping parts: base excluding the last d columns, and shifted_x excluding the last d columns
        # shifted_x already starts 'd' columns later, so its first width-d columns overlap with base's first width-d columns
        base_overlap_x = base[:, :-d]     # Shape: (height, width - d)
        shifted_overlap_x = shifted_x[:, :-d]     # Shape: (height, width - d)
        # Calculate the number of overlapping pixels for normalization
        valid_pixels_x = base_overlap_x.size    # height * (width - d)


        # Calculate the sum of the element-wise product of the overlapping centered regions
        sum_prod_x = np.sum(base_overlap_x * shifted_overlap_x)
        # Normalize the sum by the number of valid pixels and the variance
        autocorrelation[d - 1, 0] = sum_prod_x / (valid_pixels_x * variance) if valid_pixels_x > 0 and variance > 0 else np.nan

        #Calculate autocorrelation in Y-direction using NumPy
        # Select the overlapping parts: base excluding the last d rows, and shifted_y excluding the last d rows
        base_overlap_y = base[:-d, :]   # Shape: (height - d, width)
        shifted_overlap_y = shifted_y[:-d, :]   # Shape: (height - d, width)
        valid_pixels_y = base_overlap_y.size    # (height - d) * width

        # Calculate the sum of the element-wise product of the overlapping centered regions
        sum_prod_y = np.sum(base_overlap_y * shifted_overlap_y)
        # Normalize the sum by the number of valid pixels and the variance
        autocorrelation[d - 1, 1] = sum_prod_y / (valid_pixels_y * variance) if valid_pixels_y > 0 and variance > 0 else np.nan

    # Return autocorrelation values
    return autocorrelation

def measure_runtime(image,max_dist):
    """
    Measures the execution time of both autocorrelation methods (loops vs NumPy) for a given image.

    Args:
        image (np.ndarray): The input image.
        max_dist (int): The maximum distance (lag) to compute autocorrelation for.

    Returns:
        tuple: Contains:
               - autocorr_loops (np.ndarray): Result from the loop-based method.
               - autocorr_numpy (np.ndarray): Result from the NumPy-based method.
               - time_loops (float): Execution time for the loop-based method in seconds.
               - time_numpy (float): Execution time for the NumPy-based method in seconds.
    """
    #Record the time before starting the loop based calculation
    start = time.time()
    autocorr_loops = autocorrelation_using_loops(image,max_dist)
    time_loops = time.time() - start

    # Record the time before starting the NumPy based calculation
    start = time.time()
    autocorr_numpy = autocorrelation_using_numpy(image,max_dist)
    time_numpy = time.time() - start

    # Return the results and the measured times
    return autocorr_loops, autocorr_numpy, time_loops, time_numpy

def plot_autocorrelation(autocorr, image_index, save_dir):
    """
    Plots the normalized autocorrelation values against distance for a single image.

    Args:
        autocorr (np.ndarray): The (max_dist, 2) array of autocorrelation values.
        image_index (int): The index of the image .
        save_dir (str): Directory to save the plot.
    """

    # Create an array of distances (lags) from 1 to max_dist
    distances = np.arange(1, autocorr.shape[0] + 1)
    # Plot autocorrelation values for both x and y directions
    plt.figure()
    plt.plot(distances, autocorr[:, 0], label='X-direction')
    plt.plot(distances, autocorr[:, 1], label='Y-direction')
    # Set the labels and title for the plot
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Normalized Autocorrelation')
    plt.title(f'Normalized Autocorrelation - Image {image_index + 1}')
    plt.legend()
    plt.grid(True)

    # Create the directory to save the plots
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"normalized_autocorr_image_{image_index + 1}.png")
    plt.savefig(save_path)
    plt.close()


def plot_runtime_comparison(image_names, time_loops, time_numpy, save_dir):
    """
    Plots a bar graph comparing the runtimes of the loop-based and NumPy methods.

    Args:
        image_names (list): List of names or indices of the images.
        time_loops (list): List of runtimes for the loop-based method.
        time_numpy (list): List of runtimes for the NumPy method.
        save_dir (str): Directory to save the plot.
    """

    x = np.arange(len(image_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, time_loops, width, label='Loops')
    rects2 = ax.bar(x + width/2, time_numpy, width, label='NumPy')

    # Add some text for labels, title and axes ticks
    ax.set_xlabel('Image')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison: Loops vs. NumPy')
    ax.set_xticks(x)
    ax.set_xticklabels(image_names)
    y_ticks = np.arange(0, max(max(time_loops), max(time_numpy)) + 0.5, 0.09)
    ax.set_yticks(y_ticks)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    # Create the directory to save the plots
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "runtime_comparison.png")
    plt.savefig(save_path)
    plt.close()

def main():

    """
    Main logic to load images, specifying maximum distance for autocorrelation calculation, compute autocorrelation, and plot results.
    Loads images from a specified folder, computes normalized autocorrelation using both loops and Numpy methods,
    measures execution time for both methods, and plots the results for each image and saves runtime data to .csv file for comparison.
    The results are saved in the specified directory.

    Change the max_dist variable to set different maximum distance for autocorrelation calculation.
    """
    #Folfer containing the images to be processed
    #The folder should contain images in .jpg, .jpeg, or .png format
    folder = r".\Dataset"
    #Directory to save the plots and results
    save_dir = "autocorr_plots"
    #Maximum distance or lag for autocorrelation calculation
    max_dist = 20

    #Image Loading
    try:
        images = load_images_from_folder(folder)
        print(f"Found {len(images)} images in the folder.")
    except (FileNotFoundError, ValueError) as e:
        print(e)
        exit()
    
    total_images = len(images) #Total number of images loaded
    image_names = [i+1 for i in range(5)] #list of first 5 images to plot the run-time data
    all_time_loops = [] #list for storing the run-time data of the loop based method
    all_time_numpy = [] #list for storing the run-time data of the numpy based method
    run_timedata=[] #create a list for storing run-time data

    #Iterate through the loaded images
    for i, image in enumerate(images):
        print(f"\nProcessing Image {i + 1}/{total_images}...")

        # Calculate autocorrelation using both methods and measure their execution times
        autocorr_loops, autocorr_numpy, time_loops, time_numpy = measure_runtime(image,max_dist)
        print(f"Loops time: {time_loops:.4f}s | NumPy time: {time_numpy:.4f}s")
        
        #storing time data into run_time_data for evaluation 
        run_timedata.append({
        'Image Index': i + 1,
        'Loops Runtime in seconds': time_loops,
        'NumPy Runtime in seconds': time_numpy })
                                
        #storing the runtimes for the first 5 images to plot it
        if i < 5:
            all_time_loops.append(time_loops)
            all_time_numpy.append(time_numpy)

        # Plot the autocorrelation for the current image
        plot_autocorrelation(autocorr_loops, i, save_dir)
    
    #Add run time data into a dataframe
    run_timedf=pd.DataFrame(run_timedata)

    #Saving run time data into a csv in the save_dir
    run_timedf.to_csv(os.path.join(save_dir, 'runtime_data.csv'), index=False)

    # Plot the runtime comparison for the first 5 images
    plot_runtime_comparison(image_names, all_time_loops, all_time_numpy, save_dir)
    
    print(f"\nProcessing Completed. Plots and runtime data saved in '{save_dir}' directory.")

if __name__ == "__main__":
    main()