import cv2
import matplotlib.pyplot as plt
img_file = '/usr/vision/data/articulate_human_data/bouncing_new/f_0000/Image1_0000.png'
img = cv2.imread(img_file)
plt.imshow(img)
plt.show()
print("Success!")