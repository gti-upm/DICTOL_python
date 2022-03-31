from skimage.io import imread
from skimage import data, io
from matplotlib import pyplot as plt

image = imread("./dictol/data/yaleB01_P00A+000E+00.pgm")
io.imshow(image)
plt.show()
print(f'Image size: {image.shape}')
