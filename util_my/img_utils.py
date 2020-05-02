from scipy.misc import imread,imresize



def compression(image):
    image = imresize(imread(image), (224, 224)) / 255.0 - 0.5
    return image


def decompression(image):
    return