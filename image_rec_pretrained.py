import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16

#Load the Keras VGG16 model pre-trained against ImageNet Database
model = vgg16.VGG16()

#Load the image file and resizing it to 224x224 pixels(Model requirement)
img = image.load_img("bay.jpg", target_size=(224,224))

#Convert Image to array
x = image.img_to_array(img)

#Add fourth dimension(since keras expects a lists of images)
x = np.expand_dims(x, axis=0)

#Normalize the input image's pixel values to the range used when training the neural network
x = vgg16.preprocess_input(x)

#Run the image through the model to make a prediction
predictions = model.predict(x)

#Look up the names of the predicted classes. Index zero is the results for the first
predicted_classes = vgg16.decode_predictions(predictions)

print("Top predictions for this image:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print("Prediction: {} - {:2f}".format(name, likelihood))
