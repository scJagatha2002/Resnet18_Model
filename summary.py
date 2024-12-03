from tensorflow.keras.models import load_model

# Load the model
loaded_model = load_model('resnet_18_color_model.h5')

# Check the model summary to confirm it's loaded correctly
loaded_model.summary()
