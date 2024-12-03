import numpy as np
import json
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data from JSON
with open('balanced_colour.json', 'r') as file:
    data = json.load(file)

# Access color data
color_data = data.get('colors', [])
input_data, target_data = [], []

for color in color_data:
    if isinstance(color, dict):
        try:
            # Extract input RGB
            input_rgb = color['input_color']['rgb']
            input_data.append(input_rgb)

            # Collect target colors
            target_rgb = [color['complementary_color']['rgb']]
            target_rgb.extend([a['rgb'] for a in color.get('analogous_colors', [])])
            target_rgb.extend([t['rgb'] for t in color.get('triadic_colors', [])])
            target_rgb.extend([m['rgb'] for m in color.get('monochromatic_colors', [])])
            
            # Flatten target RGB values
            target_data.append(np.array(target_rgb).flatten())
        except KeyError:
            continue

# Convert to numpy arrays and normalize
input_data = np.array(input_data) / 255.0
target_data = np.array(target_data) / 255.0

# Load the model
model = tf.keras.models.load_model('resnet_18_color_model.h5')

# Predict on evaluation data
predictions = model.predict(input_data.reshape(-1, 1, 1, 3))

# Map RGB to color categories
def rgb_to_category(rgb):
    if rgb[0] > 200 and rgb[1] < 100 and rgb[2] < 100:
        return "red"
    elif rgb[1] > 200 and rgb[0] < 100 and rgb[2] < 100:
        return "green"
    elif rgb[2] > 200 and rgb[0] < 100 and rgb[1] < 100:
        return "blue"
    else:
        return "other"

# Convert predictions and targets to categories
predicted_categories = [rgb_to_category((pred * 255).astype(int)) for pred in predictions]
target_categories = [rgb_to_category((target * 255).astype(int)) for target in target_data]

# Compute accuracy and confusion matrix
accuracy = accuracy_score(target_categories, predicted_categories)
conf_matrix = confusion_matrix(target_categories, predicted_categories, labels=["red", "green", "blue", "other"])
classification_rep = classification_report(target_categories, predicted_categories, target_names=["red", "green", "blue", "other"])

# Write results to a file
with open('evaluation_results.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix) + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_rep)

print("Evaluation results saved to evaluation_results.txt")
