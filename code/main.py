
from kafka import KafkaConsumer, KafkaProducer
from input_preprocessing import preprocess
from output_results import output

import argparse
import pathlib
import timeit
import json
import numpy as np

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
        1: "If no model is specified, you need to pass the direction of the device which will process the information.",

        3: " -- "
    }
    p.error( error_message[ error ] )

def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '--tensorflow', help='Flag to indicate if the file is a tensorflow model.', nargs='?', const=True, default=False, type=bool ) 
    parser.add_argument( '--cifar10', help='Flag to indicate if using cifar10 dataset.', nargs='?', const=True, default=False, type=bool ) 
    parser.add_argument( '--models', help='Each model path.', nargs="*" )

    parser.add_argument( '-i', '--input', help="Directory path of input images." )
    parser.add_argument( '-f', '--input_format', help="Format of input images.", nargs='?', const=True, default="jpg")
    parser.add_argument( '-l', '--labels', help="File path of labels file." )
    parser.add_argument( '--height', help="", default=180 )
    parser.add_argument( '--width', help="", default=180 )
    
    parser.add_argument( '--name', help="Name of the device.", required=True )

    parser.add_argument( '--producer-back', help="IP to send values to the previous device.", default="localhost" )
    parser.add_argument( '--producer-back-port', help="Port to send values to the previous device.", default="9092" )
    parser.add_argument( '--producer-back-topic', help="Topic to send values to the previous device." )
    parser.add_argument( '--producer-front', help="IP to send values to the next device." )
    parser.add_argument( '--producer-front-port', help="Port to send values to the next device." )
    parser.add_argument( '--producer-front-topic', help="Topic to send values to the previous device." )
    
    parser.add_argument( '--consumer-back', help="IP to receive values from the previous device.", default="localhost" )
    parser.add_argument( '--consumer-back-port', help="Port to send values to the previous device", default="9092" )
    parser.add_argument( '--consumer-back-topic', help="Topic to send values to the previous device" )
    parser.add_argument( '--consumer-front', help="IP to receive values from the next device." )
    parser.add_argument( '--consumer-front-port', help="Port to send values to the next device." )
    parser.add_argument( '--consumer-front-topic', help="Topic to send values to the previous device" )

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
                if not (args.producer_front and args.producer_front_port and \
                    args.consumer_front and args.consumer_front_port):
                    error_input( parser, 1 )
                else:

                    from kafka import KafkaConsumer, KafkaProducer

                    # At this step, this code should send values and wait for response of these values.
                    # print( args.next_device )
                    
                    # https://kafka.apache.org/quickstart
                    producer = KafkaProducer( bootstrap_servers=["{}:{}".format( args.producer_front, args.producer_front_port )], max_request_size=1024000000 )
                    consumer = KafkaConsumer( args.consumer_front_topic, bootstrap_servers=["{}:{}".format( args.consumer_front, args.consumer_front_port )],
                                            value_deserializer=lambda m: json.loads( m.decode() ),
                                            auto_offset_reset='earliest',
                                            enable_auto_commit=True,
                                            auto_commit_interval_ms=1,
                                            max_partition_fetch_bytes=1024000000,
                                            group_id=args.name
                                            )
    
                    ## Inference
                    try:
                        
                        _y = []
                        start_global_time = timeit.default_timer()
                        times = []
                        print( "Total to be sent: {}".format( len(x) ) )
                        for input_i in x:

                            start_prediction_time = timeit.default_timer()
                            producer.send( args.producer_front_topic, json.dumps( input_i.tolist() ).encode() )
                            producer.flush()
                            print( "Message sent." )
                            for msg in consumer:
                                if msg != {} or msg is not None:           
                                    _y.append( msg )
                                    break
                            times.append( timeit.default_timer() - start_prediction_time )
                            print( "result:\n\t", _y[-1:], "\ntotal time:", times[-1:] )
                        output( y, _y, timeit.default_timer() - start_global_time, times )
                    finally:
                        consumer.close()
            else:
                # If a model or models are passed as argument, it will inference using them in the order they are
                # introduced and then send the result if a higher layer is specified.

                ## Load models
                models = []
                for model_path in args.models:
                    models.append( tf.keras.models.load_model( model_path ) )
                
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
                            # print(_x[-1:][0][0] )
                            if _x[-1:][0].max() >= args.threshold:
                                _x = _x[-1:][0][0]
                                break
                    _y.append( _x )
                    times.append( timeit.default_timer() - start_prediction_time )

                output( y, _y, timeit.default_timer() - start_global_time, times )
    elif args.cifar10:
        
        _, (x, y) = tf.keras.datasets.cifar10.load_data()

        # model inference
        if not args.models:
            # If no model is passed, this code will only pass data after preprocessing them and
            # recover the output of another model in a higher layer of the architecture.
            if not (args.producer_front and args.producer_front_port and \
                args.consumer_front and args.consumer_front_port):
                error_input( parser, 1 )
            else:

                from kafka import KafkaConsumer, KafkaProducer

                # At this step, this code should send values and wait for response of these values.
                # print( args.next_device )
                
                # https://kafka.apache.org/quickstart
                producer = KafkaProducer( bootstrap_servers=["{}:{}".format( args.producer_front, args.producer_front_port )], max_request_size=1024000000 )
                consumer = KafkaConsumer( args.consumer_front_topic, bootstrap_servers=["{}:{}".format( args.consumer_front, args.consumer_front_port )],
                                        value_deserializer=lambda m: json.loads( m.decode() ),
                                        auto_offset_reset='earliest',
                                        enable_auto_commit=True,
                                        auto_commit_interval_ms=1,
                                        max_partition_fetch_bytes=1024000000,
                                        group_id=args.name
                                        )

                ## Inference
                try:
                    
                    _y = []
                    start_global_time = timeit.default_timer()
                    times = []
                    print( "Total to be sent: {}".format( len(x) ) )
                    for input_i in x:

                        start_prediction_time = timeit.default_timer()
                        producer.send( args.producer_front_topic, json.dumps( input_i.tolist() ).encode() )
                        producer.flush()
                        print( "Message sent." )
                        for msg in consumer:
                            if msg != {} or msg is not None:           
                                _y.append( msg )
                                break
                        times.append( timeit.default_timer() - start_prediction_time )
                        print( "result:\n\t", _y[-1:], "\ntotal time:", times[-1:] )
                    output( y, _y, timeit.default_timer() - start_global_time, times )
                finally:
                    consumer.close()
        else:
            # If a model or models are passed as argument, it will inference using them in the order they are
            # introduced and then send the result if a higher layer is specified.

            ## Load models
            models = []
            for model_path in args.models:
                models.append( tf.keras.models.load_model( model_path ) )
            
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
                        # print(_x[-1:][0][0] )
                        if _x[-1:][0].max() >= args.threshold:
                            _x = _x[-1:][0][0]
                            break
                _y.append( _x )
                times.append( timeit.default_timer() - start_prediction_time )

            output( y, _y, timeit.default_timer() - start_global_time, times )
    else:
        # Here, models are loaded, but no data and so need to received and send it back.
        from kafka import KafkaConsumer, KafkaProducer

        if not args.models:
            error_input( parser, 3 )
        else:

            ## Load models
            models = []
            for model_path in args.models:
                models.append( tf.keras.models.load_model( model_path ) )

            producer = KafkaProducer( bootstrap_servers=["{}:{}".format( args.producer_back, args.producer_back_port )] , max_request_size=1024000000)
            consumer = KafkaConsumer( args.consumer_back_topic, bootstrap_servers=["{}:{}".format( args.consumer_back, args.consumer_back_port )],
                                    # value_deserializer=lambda m: json.loads( m.decode() ),
                                    auto_offset_reset='earliest',
                                    enable_auto_commit=True,
                                    auto_commit_interval_ms=1,
                                    max_partition_fetch_bytes=1024000000,
                                    group_id=args.name
                                    )
            # consumer.subscribe([])
            if args.producer_front and args.producer_front_port and \
                args.consumer_front and args.consumer_front_port:
                producer_next = KafkaProducer( bootstrap_servers=["{}:{}".format( args.producer_front, args.producer_front_port )],  max_request_size=1024000000 )
                consumer_next = KafkaConsumer( args.consumer_front_topic, bootstrap_servers=["{}:{}".format( args.consumer_front, args.consumer_front_port )],
                                    # value_deserializer=lambda m: json.loads( m.decode() ),
                                    auto_offset_reset='earliest',
                                    enable_auto_commit=True,
                                    auto_commit_interval_ms=1,
                                    max_partition_fetch_bytes=1024000000,
                                    group_id=args.name
                                    )

            try:

                print( "Waiting for inputs...")
                print( consumer )
                for msg in consumer:

                    if msg != {} or msg is not None:
                        
                        input_i = np.array( json.loads( msg.value.decode() ) )
                        print( "Input Received" )
                        start_prediction_time = timeit.default_timer()
                        _x = input_i
                        result_x = _x
                        # print( models )
                        ended = False
                        for model in models:
                            # print( model )
                            _x = model.predict( _x )
                            if len( _x ) > 1:
                                if _x[-1:][0].max() >= args.threshold:
                                    result_x = _x[-1:][0]
                                    ended = True
                                    break
                            else:
                                result_x = _x[0]
                        
                        result = { "result": result_x.tolist(), "device": args.name, "execution-time": timeit.default_timer() - start_prediction_time }
                        if args.producer_front and args.producer_front_port and \
                            args.consumer_front and args.consumer_front_port and not ended:

                            producer_next.send( args.producer_front_topic, json.dumps( _x[:-1][0].tolist() ).encode() )
                            producer_next.flush()
                            print( "Sent front:\n\t", result )

                            print( consumer_next )
                            for m in consumer_next:
                                print( "Received: {}".format(m) )
                                load_message = json.loads(m.value.decode())
                                print( "Received: {}".format(load_message))
                                if load_message != {} or load_message is not None:
                                    if load_message.get("result", None) != None:
                                        print( "Received:\n\t", result )
                                        result["next"] = load_message
                                        break

                        result["total-time"] = timeit.default_timer() - start_prediction_time
                        # Return result to device
                        producer.send( args.producer_back_topic, json.dumps( result ).encode() )
                        producer.flush()
                        print( "Sent back:\n\t", result )
            finally:
                print( "Closing ")
                consumer.close()
                if args.producer_front and args.producer_front_port and \
                    args.consumer_front and args.consumer_front_port:
                    consumer_next.close()

if __name__ == '__main__':
    main()


# python code/main.py -i data/flower_photos/train -l data/labels.txt --name device --producer-front localhost --producer-front-port 9092 --producer-front-topic from_device_to_edge --consumer-front localhost --consumer-front-port 9092 --consumer-front-topic from_edge_to_device

# python code/main.py --models saved_models/edge --name local_edge --producer-back-topic from_edge_to_device --consumer-back-topic from_device_to_edge --producer-front localhost --producer-front-port 9092 --producer-front-topic from_edge_to_fog --consumer-front localhost --consumer-front-port 9092 --consumer-front-topic from_fog_to_edge --tensorflow
# python code/main.py --models saved_models/edge saved_models/fog saved_models/cloud --name local_edge --producer-back-topic from_edge_to_device --consumer-back-topic from_device_to_edge --tensorflow
# 192.168.48.159

# python code/main.py --models saved_models/fog --name local_fog --producer-back-topic from_fog_to_edge --consumer-back-topic from_edge_to_fog --producer-front localhost --producer-front-port 9092 --producer-front-topic from_fog_to_cloud --consumer-front localhost --consumer-front-port 9092 --consumer-front-topic from_cloud_to_fog --tensorflow
# python code/main.py --models saved_models/fog saved_models/cloud --name local_fog --producer-back-topic from_fog_to_edge --consumer-back-topic from_edge_to_fog --tensorflow

# python code/main.py --models saved_models/cloud --name local_cloud --producer-back-topic from_cloud_to_fog --consumer-back-topic from_fog_to_cloud --tensorflow