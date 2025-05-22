# Image Autocorrelation Analysis

## Overview

This Python script calculates the autocorrelation of images in a specified folder. It provides two methods for calculating autocorrelation:
one using nested loops and the other using NumPy. The script loads images, computes the autocorrelation, generates plots of the autocorrelation values,
measures the execution time of each method, saves it to a .csv file, and also creates runtime comparison bar graph of first five images.

## Features

* **Image Loading:** Loads images from a specified folder.  Supports `.jpg`, `.jpeg`, and `.png` formats.  Resizes images to a specified size.
* **Grayscale Conversion:** Converts all loaded images to grayscale.
* **Autocorrelation Calculation:**
    * Calculates normalized autocorrelation in the x and y directions.
    * Provides two methods:
        * Loop-based implementation.
        * NumPy-based implementation.
* **Boundary Handling:** Uses reflection (mirror padding) to handle image boundaries during autocorrelation calculation.
* **Limit or lag:** Limit for autocorrelation calculation can be set by changing the value of max_dist variable.
* **Runtime Measurement:** Measures and compares the execution time of the loop-based and NumPy-based methods.
* **Visualization:**
    * Generates plots of the autocorrelation values for each image.
    * Generates a bar graph comparing the runtimes of the two methods.
* **Results Saving:** Saves the runtime data to a CSV file.
* **Error Handling:** Includes error handling for file operations, image loading, and calculations.

## Requirements

* Python 3.x
* OpenCV
* NumPy
* Matplotlib (for creating plots)
* Pandas (for saving runtime values to .csv file)

## Installation

1.  **Install Python:** Ensure Python 3.x is installed.
2.  **Install the required packages:**
    ```bash
    pip install opencv-python numpy matplotlib pandas
    ```

## Usage

1.  **Prepare the images:**
    * The images to process are placed in a folder named `Dataset` in the same directory as the script .  
	* The images should be named numerically.
    * The images are in `.jpg`, `.jpeg`, or `.png` format.

2.  **Run the script:**
    * Open a terminal or command prompt.
    * Navigate to the directory where the script is saved.
    * Run the script.

3.  **View the results:**
    * The script will create a folder named `autocorr_plots` in the same directory as the script.
    * This folder will contain:
        * Plots of the autocorrelation values for each image (e.g. `normalized_autocorr_image_1.png`).
        * A bar graph comparing the runtimes of the loop-based and NumPy methods (`runtime_comparison.png`).
        * A CSV file (`runtime_data.csv`) containing the runtime data for each image.

## Script Details

### `load_images_from_folder(folder, image_size=(256, 256))`

* Loads images from the specified folder.
* Converts images to grayscale using `cv.imread(filepath, cv.IMREAD_GRAYSCALE)`.
* Resizes images to the specified `image_size` using `cv.resize`.
* Handles file not found errors and invalid image file errors.
* Sorts the image filenames numerically before loading.

### `calculate_mean_variance(image)`

* Calculates the mean and variance of the pixel values in the given image using `np.mean` and `np.var`.

### `autocorrelation_using_loops(image, max_dist)`

* Calculates the normalized autocorrelation using nested loops.
* Implements the autocorrelation formula directly.
* Pads the image using `np.pad` with the `reflect` mode to handle boundaries.
* Calculates autocorrelation for both the x and y directions.

### `autocorrelation_using_numpy(image, max_dist)`

* Calculates the normalized autocorrelation using NumPy functions.
* Pads the image using `np.pad` with the `reflect` mode.
* Calculates autocorrelation for both the x and y directions.

### `measure_runtime(image, max_dist)`

* Measures the execution time of the `autocorrelation_using_loops` and `autocorrelation_using_numpy` functions using `time.time()`.
* Returns the autocorrelation results from both methods, along with their execution times.

### `plot_autocorrelation(autocorr, image_index, save_dir)`

* Generates a plot of the normalized autocorrelation values against distance for a single image using `matplotlib.pyplot`.
* Plots the autocorrelation in the x and y directions on the same graph.
* Saves the plot to a PNG file in the specified directory.

### `plot_runtime_comparison(image_names, time_loops, time_numpy, save_dir)`

* Generates a bar graph comparing the runtimes of the loop-based and NumPy-based autocorrelation methods using `matplotlib.pyplot`.
* Displays the runtime for each method for the first 5 images.
* Saves the plot to a PNG file in the specified directory.

### `main()`

* The main function of the script.
* Sets the input folder, output directory, and maximum distance for autocorrelation calculation.
* Loads the images using `load_images_from_folder`.
* Iterates through the images, calculating autocorrelation and measuring runtime for each.
* Calls `plot_autocorrelation` to generate the autocorrelation plots.
* Calls `plot_runtime_comparison` to generate the runtime comparison plot.
* Saves the runtime data to a CSV file using `pandas.DataFrame.to_csv`.
* Prints a completion message.

##  Notes

* The `max_dist` variable in the `main()` function controls the maximum distance (lag) for which autocorrelation is calculated.  You can change this value to adjust the range of the calculation.
