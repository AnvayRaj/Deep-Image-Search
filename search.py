from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# Read image features
fe=FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./features").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./images")/(feature_path.stem+".jpg"))
features = np.array(features)

print(features.shape)

def search(img):
    query = fe.extract(img)
    dists = np.linalg.norm(features - query, axis=1) # L2 distances to the features
    ids = np.argsort(dists)[:9] # Top 9 results
    scores = [img_paths[id] for id in ids]
    return scores, dists

# Search & Plot results
img=Image.open("Test Images/baby1.jpg")
results,numerical_scores = search(img)
for i in range(len(results)):
    im = mpimg.imread(results[i])
    plt.subplot(331+i),plt.imshow(im)

# numerical_scores.sort(reverse=True)
print(sorted(numerical_scores.tolist()))
plt.show()
