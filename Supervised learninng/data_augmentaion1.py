import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

df = pd.read_csv('fashion-mnist_train.csv')

image_data = df.iloc[0, 1:].values.astype('uint8').reshape(28, 28)


img = Image.fromarray(image_data)

img_rotate = img.rotate(15)

img_shift = ImageChops.offset(img, 3, 0)

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

axes[0].imshow(img, cmap='gray')
axes[0].set_title("1. Original Image")

axes[1].imshow(img_rotate, cmap='gray')
axes[1].set_title("3. Rotated (15°)")

axes[2].imshow(img_shift, cmap='gray')
axes[2].set_title("4. Shifted")

plt.show()