import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
# Load the SavedModel
model = tf.saved_model.load('saved_model/')
# Load and preprocess the input image
image_path = r'teja.jpeg'
img = Image.open(image_path)
img = img.resize((320, 320))  # Resize to match input shape of the model
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Normalize pixel values

# Make a prediction using the loaded model
preds = model(img_array, training=False)
preds = tf.squeeze(preds)  # Remove batch dimension
class_idx = tf.math.argmax(preds, axis=-1)
class_names = ['class1', 'class2', 'class3']  # List of class names in the order of model's output
class_name = class_names[class_idx]
#confidence = tf.math.exp(preds[class_idx]) / tf.reduce_sum(tf.math.exp(preds))
confidence = preds[class_idx]
#print(f'Predicted class: {class_name}, Confidence: {confidence.numpy()}')

# Display the input image and the predicted class with its confidence score
plt.imshow(img)
plt.title(f'{class_name}, {confidence * 100:.2f}%')
plt.show()


# Plot the image and predicted probabilities
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#ax1.imshow(img)
#ax1.axis('off')
#ax1.set_title('Input Image')
#ax2.bar(class_names, preds.numpy())
#ax2.set_title('Predicted Class Probabilities')
#ax2.set_xlabel('Class')
#ax2.set_ylabel('Probability')
#plt.show()