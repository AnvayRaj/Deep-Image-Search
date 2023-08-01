from PIL import Image
from pathlib import Path
import numpy as np

from feature_extractor import FeatureExtractor

fe = FeatureExtractor()

for img_path in sorted(Path("./images").glob("*.jpg")):
    print(img_path)

    # Extract deep feature
    feature = fe.extract(img=Image.open(img_path))

    feature_path = Path("./features")/(img_path.stem+".npy")
    print(feature_path)

    # Save the feature
    np.save(feature_path, feature)