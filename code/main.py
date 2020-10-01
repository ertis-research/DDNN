
from kafka import KafkaConsumer, KafkaProducer
from input_preprocessing import preprocess
from output_results import output

import argparse
import pathlib
import timeit

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

def error_input( p, error ):
    error_message = {
        0: "The --input (-i) argument requires the --labels arguments.",
        1: "If no model is specified, you need to pass the direction of the device which will process the information."
    }
    p.error( error_message[ error ] )

def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '--tensorflow', help='Flag to indicate if the file is a tensorflow model.', nargs='?', const=True, default=False, type=bool ) 
    parser.add_argument( '--models', help='Each model path.', nargs="*", required=True )

    parser.add_argument( '-i', '--input', help="Directory path of input images." )
    parser.add_argument( '-f', '--input_format', help="Format of input images.", nargs='?', const=True, default="jpg")
    parser.add_argument( '-l', '--labels', help="File path of labels file." )
    parser.add_argument( '--height', help="", default=180 )
    parser.add_argument( '--width', help="", default=180 )
    
    parser.add_argument( '--edge', help="This device computes entries as an edge device.", nargs='?', const=True, default=False, type=bool )
    parser.add_argument( '--fog', help="This device computes entries as a fog device.", nargs='?', const=True, default=False, type=bool )
    parser.add_argument( '--cloud', help="This device computes entries as a cloud device.", nargs='?', const=True, default=False, type=bool )
    
    parser.add_argument( '-n', '--next-device', help="Indicates the value of the IP to send the inferance values obtained." )
    parser.add_argument( '-t', '--threshold', help="Minimum value to be accepted as a correct value.", type=float, default=0.8)
    args = parser.parse_args()

    if args.tensorflow:
        import tensorflow as tf
        print( "\tNOTE: Using Tensorflow {}.".format( tf.__version__ ) )
    # else:
    #     https://www.tensorflow.org/lite/guide/python
    #     import tflite_runtime.interpreter as tflite

    if args.input:

        if not args.labels:
            error_input( parser, 0 )

        else:

            labels, images = read_images( args.input, args.input_format, args.labels )

            x, y = preprocess( labels, images, args.height, args.width )
            
            # model inference
            if not args.models:
                # If no model is passed, this code will only pass data after preprocessing them and
                # recover the output of another model in a higher layer of the architecture.
                if not args.next_device:
                    error_input( parser, 1 )
                else:

                    from kafka import KafkaConsumer, KafkaProducer
                    print( args.next_device )

            else:
                # If a model or models are passed as argument, it will inference using them in the order they are
                # introduced and then send the result if a higher layer is specified.
                
                ## Load models
                models = []
                for model_path in args.models:
                    models.append( tf.keras.models.load_model( model_path ) )

                if not args.edge and not args.fog and not args.cloud:
                    ## Inference
                    _y = []
                    start_global_time = timeit.default_timer()
                    times = []
                    for input_i in x:

                        start_prediction_time = timeit.default_timer()
                        _x = input_i
                        for model in models:
                            _x = model.predict( _x )
                            if len( _x ) > 1:
                                if _x[-1:][0].max() >= args.threshold:
                                    _x = _x[-1:][0][0]
                                    break
                        _y.append( _x )
                        times.append( timeit.default_timer() - start_prediction_time )
                else:
                    # Here, models are loaded, but no data and so need to received and send it back.
                    if args.edge:
                        print( "EDGE" )
                    elif args.fog:
                        print( "FOG" )
                    elif args.cloud:
                        print( "CLOUD" )

                output( y, _y, timeit.default_timer() - start_global_time, times )

if __name__ == '__main__':
    main()
