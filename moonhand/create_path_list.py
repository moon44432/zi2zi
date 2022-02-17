import glob
import os

paths = sorted(glob.glob("E:\\Projects\\font\\moonhand\\*.png"))

with open('path.txt', 'w') as f:
    for imgpath in paths:
        f.write(imgpath + '\n')