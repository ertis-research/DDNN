
from kafka import KafkaConsumer, KafkaProducer
from input_preprocessing import preprocess
from output_results import output

import argparse
import pathlib

# References:
# https://www.tensorflow.org/tutorials/images/classification
# https://github.com/ertis-research/DDNN-Implementation/blob/master/EDGE/model.py
# https://github.com/ertis-research/DDNN-Implementation/blob/master/EDGE/model-coral-1.py

def read_images( i_path, i_format, label_file_path ):
    
    images = {}
    labels = []
    # Read image per label
    with open( label_file_path, "r" ) as label_file:
        
        for line in label_file.readlines():

            label = line.replace("\n", "").replace("\t", "")
            labels.append( label )
            data_dir = pathlib.Path( i_path + "/" + label )
            images[ label ] = list( data_dir.glob( '*.{}'.format( i_format ) ) )
    
    return labels, images

def error_input( p ):
    p.error( "The --input (-i) argument requires the --labels arguments." )

def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '-t', '--tensorflow', help='Flag to indicate if the file is a tensorflow model.', nargs='?', const=True, default=False, type=bool ) 
    # parser.add_argument( '--model', help='File path of the model.' )

    parser.add_argument( '-i', '--input', help="Directory path of input images." )
    parser.add_argument( '-f', '--input_format', help="Format of input images.", nargs='?', const=True, default="jpg")
    parser.add_argument( '-l', '--labels', help="File path of labels file." )

    # parser --edge
    # parser --fog
    # parser --cloud

    args = parser.parse_args()

    if args.tensorflow:
        import tensorflow as tf
        print( "\tNOTE: Using Tensorflow {}.".format( tf.__version__ ) )
    # else:
    #     https://www.tensorflow.org/lite/guide/python
    #     import tflite_runtime.interpreter as tflite

    if args.input:

        if not args.labels:
            error_input( parser )

        else:

            labels, images = read_images( args.input, args.input_format, args.labels )

            preprocess( labels, images )
            
            # model inference

            output( None )

if __name__ == '__main__':
    main()
