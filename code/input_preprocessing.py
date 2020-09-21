from PIL import Image

def preprocess( list_images ):
    
    print( "Total images: {}.".format( len(list_images) ) )
    
    Image.open( str(list_images[1]) ).show()
    print( "Image {} shown.".format( list_images[1] ) )
