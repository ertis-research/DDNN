from PIL import Image
from tensorflow.keras import preprocessing
import numpy as np

def preprocess( labels, list_images, img_height, img_width ):
    
    # print( "Total images: {}.".format( len(list_images) ) )
    # print( list_images["daisy"][1] )
    # Image.open( str(list_images["daisy"][1]) ).show()
    # print( "Image {} shown.".format( list_images["daisy"][1] ) )
    # print( labels )

    x = []
    y = []

    for i, label in enumerate( labels ):

        for image_path in list_images[ label ]:
            image = preprocessing.image.load_img( image_path, target_size=( img_height, img_width ) )
            input_arr = preprocessing.image.img_to_array( image )
            x.append( input_arr.reshape(1, img_height, img_width, 3 ) )
            one_hot = np.zeros( len(labels) )
            one_hot[i] = 1
            y.append( one_hot )

    return x, y
