import json

# Load the JSON data
with open('balanced_colour.json', 'r') as file:
    data = json.load(file)

# Initialize counters
total_input_colors = 0
total_complementary_colors = 0
total_analogous_colors = 0
total_triadic_colors = 0
total_monochromatic_colors = 0

# Iterate through the dataset
if 'colors' in data:
    for color_entry in data['colors']:
        total_input_colors += 1
        total_complementary_colors += 1  # Each entry has one complementary color
        total_analogous_colors += len(color_entry.get('analogous_colors', []))
        total_triadic_colors += len(color_entry.get('triadic_colors', []))
        total_monochromatic_colors += len(color_entry.get('monochromatic_colors', []))
else:
    print("Error: 'colors' key not found in the dataset.")

# Calculate total size
total_dataset_size = (
    total_input_colors +
    total_complementary_colors +
    total_analogous_colors +
    total_triadic_colors +
    total_monochromatic_colors
)

# Print the results
print(f"Total Input Colors: {total_input_colors}")
print(f"Total Complementary Colors: {total_complementary_colors}")
print(f"Total Analogous Colors: {total_analogous_colors}")
print(f"Total Triadic Colors: {total_triadic_colors}")
print(f"Total Monochromatic Colors: {total_monochromatic_colors}")
print(f"Total Dataset Size: {total_dataset_size}")
