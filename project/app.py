# way to upload image : endpoint
# way to save the image
# function to make prediction on the image
# show the results

from flask import Flask
from flask import render_template
import os
from flask import request
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER =r"\project\static\uploads"
## In Python, backslashes are treated as escape characters, and you need to either double them (\\) or use a raw string prefix (r"...") 
# to avoid the "truncated \UXXXXXXXX escape" error.

model_path=r"saved_model"

# Define the class names
class_names = ["apple_red_delicios_1", "apple_red_yellow_1", "apple_rotten_1"]

# Load the trained model
model = tf.keras.models.load_model(model_path)

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER,image_file.filename)
            image_file.save(image_location)
     
             # Preprocess the image
            img = Image.open(image_location)
            img = img.resize((320, 320))  # Resize the image

                # Ensure the image has three channels (RGB)
            if img.mode != "RGB":
                img = img.convert("RGB")

            img_array = np.array(img)
            img_array = img_array[np.newaxis, ...]
            img_array = img_array / 255.0  # Normalize the image
            # Make predictions using the loaded model
            predictions = model.predict(img_array)
            class_index = np.argmax(predictions)
            print(class_index)
            class_name = class_names[class_index]
            
            # Return the predicted class
            return render_template("index.html", prediction=class_name)
    return render_template('index.html', prediction=0)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
