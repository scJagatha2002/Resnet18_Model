import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import sys

# Load your data
with open('balanced_colour.json', 'r') as file:
    data = json.load(file)


# Access the 'colors' list from the data
if 'colors' in data:
    color_data = data['colors']
else:
    print("Error: 'colors' key not found in JSON data.")
    sys.exit()

# Prepare lists to hold input and target data
input_data = []
target_data = []

# Iterate through the list of colors
for index, color in enumerate(color_data):
    # Ensure color is a dictionary
    if isinstance(color, dict):
        try:
            # Extract the input color (RGB)
            input_rgb = color['input_color']['rgb']
            input_data.append(input_rgb)

            # Extract the target colors
            target_complementary = color['complementary_color']['rgb']

            # Collect all target colors into a single list
            targets = [target_complementary]  # Start with complementary color
            targets.extend([a['rgb'] for a in color.get('analogous_colors', [])])  # Add analogous colors
            targets.extend([t['rgb'] for t in color.get('triadic_colors', [])])  # Add triadic colors
            targets.extend([m['rgb'] for m in color.get('monochromatic_colors', [])])  # Add monochromatic colors

            # Flatten the target RGB values into a single array
            target_rgb = np.array(targets).flatten().tolist()
            
            target_data.append(target_rgb)
        except KeyError as e:
            print(f"Key error for index {index}: {e}")
        except Exception as e:
            print(f"Error processing color at index {index}: {e}")
    else:
        print(f"Unexpected item in JSON data at index {index}: {color}")

# Check if we have collected input and target data
if not input_data or not target_data:
    print("Error: No valid input or target data collected.")
else:
    # Convert lists to NumPy arrays
    input_data = np.array(input_data)
    target_data = np.array(target_data)

    # Normalize the RGB values to [0, 1]
    input_data = input_data / 255.0
    target_data = target_data / 255.0

    # Split data into training and evaluation sets (e.g., 80% train, 20% evaluation)
    X_train, X_eval, y_train, y_eval = train_test_split(input_data, target_data, test_size=0.2, random_state=42)

    # ResNet block for 2D data
    def conv_block(x, filters, kernel_size=3, stride=1, activation='relu'):
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x

    def resnet_block(x, filters):
        shortcut = x
        x = conv_block(x, filters)
        x = conv_block(x, filters)
        
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, kernel_size=1)(shortcut)
        
        x = layers.add([x, shortcut])
        return layers.Activation('relu')(x)

    def build_resnet_18(input_shape):
        inputs = layers.Input(shape=input_shape)
        x = conv_block(inputs, 64, kernel_size=7, stride=2)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
        
        for _ in range(2):
            x = resnet_block(x, 64)
        for _ in range(2):
            x = resnet_block(x, 128)
        for _ in range(2):
            x = resnet_block(x, 256)
        for _ in range(2):
            x = resnet_block(x, 512)

        x = layers.GlobalAveragePooling2D()(x)
        # Output layer with a dynamic number of outputs based on your needs
        outputs = layers.Dense(len(target_data[0]), activation='sigmoid')(x)
        model = models.Model(inputs, outputs)
        return model

    # Reshape input data for the model
    input_shape = (1, 1, 3)  # Input shape for a single RGB color
    model = build_resnet_18(input_shape)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Redirect output to a log file for training
    with open('training_log.txt', 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = f  # Change the standard output to the log file

        # Train the model with early stopping
        model.fit(X_train.reshape(-1, 1, 1, 3), y_train, epochs=100, batch_size=32, 
                  validation_data=(X_eval.reshape(-1, 1, 1, 3), y_eval), 
                  callbacks=[early_stopping], verbose=1)

        sys.stdout = original_stdout  # Reset the standard output to its original value

    # Save the model
    model.save('resnet_18_color_model.h5')

    # Evaluate the model and handle output encoding
with open('evaluation_log.txt', 'w', encoding='utf-8') as eval_file:
    # Evaluate the model without verbose output
    evaluation_loss = model.evaluate(X_eval.reshape(-1, 1, 1, 3), y_eval, verbose=0)

    # Write the evaluation loss to the file
    eval_file.write(f"Evaluation loss: {evaluation_loss}\n")
