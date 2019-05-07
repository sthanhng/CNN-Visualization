# Convolutional Neural Network Visualization

## Overview

- Visualizing Convolutional Layers
- Pre-trained VGG Model
- How to visualize filters
- How to visualize feature maps

## Visualizing Convolutional Layers

Convolutional neural networks are designed to work with image data, and their structure and function suggest that should be less inscrutable than other types of neural networks.

Both filters and feature maps can be visualized.

## Pre-trained VGG Model

We can load and summarize the VGG16 model with just a few lines of code:

```python
# Import the VGG16 model
from keras.applications.vgg16 import VGG16

# Load the model
model = VGG16()
# Summarize the model
model.summary()
```

## How to visualize filters

The first step is to review the filters in the model, to see what we have to work with.

```
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
```

The model summary printed above summarizes the output shape of each layer, e.g. the shape of the resulting feature maps. It does not five any idea of the shape of the filters (weights) in the network, only the total number of weights per layer. We can access all of the layers of the model via the `model.layers` property.

Each layer has a `layer.name` property, where the convolutional layers have a naming convolution like `block#_conv#`, where the '#' is an integer.

Each convolutional layer has two sets of weights:

- One is the block of filters, and
- The other is the block of bias values.

These are accessible via the `layer.get_weights()` function. We can retrieve these weights and then summarize their shape.

```python
# Summarize filters in each convolutional layer
from keras.applications.vgg16 import VGG16

# Load the model
model = VGG16()
# Summarize filter shapes
for layer in model.layers:
	# Check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# Get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)
```

A list of layer details

```
block1_conv1 (3, 3, 3, 64)
block1_conv2 (3, 3, 64, 64)
block2_conv1 (3, 3, 64, 128)
block2_conv2 (3, 3, 128, 128)
block3_conv1 (3, 3, 128, 256)
block3_conv2 (3, 3, 256, 256)
block3_conv3 (3, 3, 256, 256)
block4_conv1 (3, 3, 256, 512)
block4_conv2 (3, 3, 512, 512)
block4_conv3 (3, 3, 512, 512)
block5_conv1 (3, 3, 512, 512)
block5_conv2 (3, 3, 512, 512)
block5_conv3 (3, 3, 512, 512)
```

We can retrieve the filters from the first layer:

```python
filters, biases = model.layers[1].get_weights()
```

We can normalize their values to the range 0-1 to make them easy to visualize.

```python
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
```

Now we can enumerate the first six filters out of the 64 in the block and plot each of the three channels of each filter.

```python
from matplotlib import pyplot

# Plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# Get the filter
	f = filters[:, :, :, i]
	# Plot each channel separately
	for j in range(3):
		# Specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# Plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# Show the figure
pyplot.show()
```

## How to visualize feature maps

The activation maps (feature maps) capture the result of applying the filters to input, such as the input image or another feature map.

The idea of visualizing a feature map for a specific input image would be to understand what features of the input are detected or preserved in the feature maps. The expectation would be that the feature maps close to the input detect small or fine-grained detail, whereas feature maps close to the output of the model capture more general features.

The example below will enumerate all layers in the model and print the output size or feature map size for each convolutional layer as well as the layer index in the model.

```python
# Summarize feature map size for each conv layer
from keras.applications.vgg16 import VGG16

# Load the model
model = VGG16()
# Summarize feature map shapes
for i in range(len(model.layers)):
	layer = model.layers[i]
	# Check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# Summarize output shape
	print(i, layer.name, layer.output.shape)
```

Output:

```
1 block1_conv1 (?, 224, 224, 64)
2 block1_conv2 (?, 224, 224, 64)
4 block2_conv1 (?, 112, 112, 128)
5 block2_conv2 (?, 112, 112, 128)
7 block3_conv1 (?, 56, 56, 256)
8 block3_conv2 (?, 56, 56, 256)
9 block3_conv3 (?, 56, 56, 256)
11 block4_conv1 (?, 28, 28, 512)
12 block4_conv2 (?, 28, 28, 512)
13 block4_conv3 (?, 28, 28, 512)
15 block5_conv1 (?, 14, 14, 512)
16 block5_conv2 (?, 14, 14, 512)
17 block5_conv3 (?, 14, 14, 512)
```

We can use this information and design a new model that is subset of the layers in the full VGG16 model. For example, after loading the VGG model, we can define a new model that outputs a feature map from the first convolutional layer (index 1) as follows.

```python
# Redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
```

After defining the model, we need to load the bird image with the size expected by the model, in this case, `224×224`.

```python
# Load the image with the required shape
img = load_img('test_img.jpg', target_size=(224, 224))

# The image PIL object needs to be converted to a NumPy array of pixel data
# and expanded from a 3D array to a 4D array with the dimensions of
# [samples, rows, cols, channels], where we only have one sample.

# Convert the image to an array
img = img_to_array(img)

# Expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)

# Prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
```

We are now ready to get the feature map. We can do this easy by calling the `model.predict()` function and passing in the prepared single image.

```python
# Get feature maps for the first hidden layer
feature_maps = model.predict(img)
```

## Reference

- [How to Visualize Filters and Feature Maps in Convolutional Neural Networks](https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/?fbclid=IwAR3SdRsa8Esc_VyjvjASkwQvh5VO4gr_KSxb7xALWwBWEEck59AIlee8baE)
