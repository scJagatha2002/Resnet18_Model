import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from matplotlib import colors
from skimage import color

# Load the trained model
model = tf.keras.models.load_model('resnet_18_color_model.h5')

# Define broad color categories with RGB ranges (optional if you want to keep it)
COLOR_CATEGORIES = {
    "lightblue":[173,216,230],
    "blue":[0,0,255],
    "darkblue":[0,0,139],
    "turquoise":[64,224,208],
    "teal":[0,128,128],
    "beige":[245,245,220],
    "peach":[255,218,185],
    "khaki":[240,230,140],
    "olive":[128,128,0],
    "yellow":[255,255,0],
    "lavender":[230,230,250],
    "parrotgreen":[124,252,0],
    "green":[0,128,0],
    "seagreen":[46,139,87],
    "lightgreen":[144,238,144],
    "grey":[128,128,128],
    "black":[0,0,0],
    "orange":[255,69,0],
    "red":[255,0,0],
    "white":[255,255,255],
    "purple":[128,0,128],
    "magenta":[255,0,255],
    "lightpink":[255,192,203],
    "darkpink":[255,20,147],
    "Brown": [139, 69, 19],


    
}

# Function to convert RGB to Lab color space
def rgb_to_lab(rgb):
    # Normalize RGB to [0, 1]
    rgb_normalized = np.array(rgb) / 255.0
    # Convert to Lab color space using skimage
    rgb_reshaped = rgb_normalized.reshape(1, 1, 3)  # Reshape for skimage
    lab_color = color.rgb2lab(rgb_reshaped)  # Convert RGB to Lab
    return lab_color.flatten()  # Flatten the output to 1D

# Function to find the closest broad color category based on RGB
def map_to_broad_color(rgb):
    min_distance = float('inf')
    closest_color = None
    for color_name, color_rgb in COLOR_CATEGORIES.items():
        dist = distance.euclidean(rgb, color_rgb)
        if dist < min_distance:
            min_distance = dist
            closest_color = color_name
    return closest_color

# Function to find the closest color name using Lab color space
def get_color_name(rgb):
    # Convert input RGB to Lab
    lab_rgb = rgb_to_lab(rgb)

    # Predefined color names from matplotlib (in Lab color space)
    color_names = list(colors.CSS4_COLORS.keys())
    
    min_distance = float('inf')
    closest_name = None
    for color_name in color_names:
        color_rgb = np.array(colors.to_rgb(color_name)) * 255  # Convert color name to RGB
        color_lab = rgb_to_lab(color_rgb)  # Convert RGB to Lab
        dist = distance.euclidean(lab_rgb, color_lab)  # Calculate Euclidean distance in Lab space
        
        if dist < min_distance:
            min_distance = dist
            closest_name = color_name

    return closest_name

# Function to predict color combinations based on an input RGB color
def predict_color_combinations(input_rgb):
    # Normalize the input RGB color to [0, 1]
    normalized_rgb = np.array(input_rgb) / 255.0
    normalized_rgb = normalized_rgb.reshape(1, 1, 1, 3)  # Reshape for the model input

    # Make the prediction
    prediction = model.predict(normalized_rgb)

    # Scale back the predictions to [0, 255]
    predicted_colors = (prediction * 255).astype(int)

    # Reshape to get the colors in groups of RGB
    predicted_colors = predicted_colors.reshape(-1, 3).tolist()  # Reshape for RGB format

    # Map each predicted color to a broad color category and actual color name
    results = []
    for color in predicted_colors:
        broad_category = map_to_broad_color(color)
        color_name = get_color_name(color)
        results.append((color, broad_category, color_name))

    return results


# Main function to run the prediction
if __name__ == "__main__":
    # Example input RGB color (you can change this to any color you want)
    input_color = [67,29,21]  # Example color (R, G, B)

    try:
        # Get the color combinations prediction
        color_combinations = predict_color_combinations(input_color)

        # Write the output to a log file
        with open('color_prediction_output.txt', 'w', encoding='utf-8') as output_file:
            output_file.write(f"Input Color (RGB): {input_color}\n")
            output_file.write("Predicted Color Combinations (RGB), Broad Categories, and Names:\n")
            for color, broad_category, color_name in color_combinations:
                output_file.write(f"{color} - {broad_category} - {color_name}\n")

        print("Prediction complete! Check 'color_prediction_output.txt' for results.")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
