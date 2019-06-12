from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16

#CIFAR labels from the training data
class_labels = [
    "Plane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Boat",
    "Truck"
]

#load the json file that contains the model's structure
f = Path("model_structure.json")
model_structure = f.read_text()

#Recreate the keras model object from the json file
model = model_from_json(model_structure)

#Reload the models trained weights
model.load_weights("model_weights.h5")

#Load an image file to test
img = image.load_img("dog.png", target_size=(64,64))

#Convert image to numpy array
image_to_test = image.img_to_array(img)

#Add a fourth dimension to the image
list_of_images = np.expand_dims(image_to_test, axis=0)

#Normalize the data
images = vgg16.preprocess_input(list_of_images)

feature_extraction_model = vgg16.VGG16(weights="imagenet", include_top=False)


#Make predictions
results = model.predict(list_of_images)
#Just one image
single_result = results[0]

most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]

class_label = class_labels[most_likely_class_index]

print("This is a {} - likelihood : {:2f}".format(class_label, class_likelihood, ))
