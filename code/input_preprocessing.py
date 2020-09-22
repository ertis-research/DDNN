from PIL import Image

def preprocess( labels, list_images ):
    
    print( "Total images: {}.".format( len(list_images) ) )
    
    print( list_images["daisy"][1] )
    Image.open( str(list_images["daisy"][1]) ).show()
    print( "Image {} shown.".format( list_images["daisy"][1] ) )
    print( labels )
