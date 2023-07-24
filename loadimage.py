import matplotlib.image as mpimg

def read(filename):
    image_loaded = mpimg.imread(filename)
    return image_loaded