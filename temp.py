import matplotlib.image as mpimg
import matplotlib.pyplot as plt
 
# Read Images
img = mpimg.imread('./images/2008_000950_jpg.rf.bada94e801d90dd06f7e1337a2dc2825.jpg')
img2 = mpimg.imread('./images/2008_000971_jpg.rf.f8b9efbc430e763e9ad2da5fb2e1f18e.jpg')
 
# Output Images
plt.subplot(221),plt.imshow(img)
plt.subplot(224),plt.imshow(img2)
plt.show()