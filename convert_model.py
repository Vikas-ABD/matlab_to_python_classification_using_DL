import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import myModel
model = myModel.load_model()
# Save the model in SavedModel format
model.save('saved_model/')