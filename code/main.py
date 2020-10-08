
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
        1: "If no model is specified, you need to pass the direction of the device which will process the information."
        2: "If --input is not specified, one of the followings is needed: --edge, --fog, --cloud",
        3: " -- "
    }
    p.error( error_message[ error ] )

def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '--tensorflow', help='Flag to indicate if the file is a tensorflow model.', nargs='?', const=True, default=False, type=bool ) 
    parser.add_argument( '--models', help='Each model path.', nargs="*" )

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

                    # https://kafka.apache.org/quickstart
                    producer = KafkaProducer( bootstrap_servers=['localhost:9092'], max_request_size=1024000000 )
                    consumer = KafkaConsumer( bootstrap_servers=['localhost:9092'],
                                            value_deserializer=lambda m: json.loads( m.decode() ),
                                            auto_offset_reset='earliest',
                                            enable_auto_commit=True,
                                            auto_commit_interval_ms=1000,
                                            max_partition_fetch_bytes=1024000000,
                                            group_id='device'
                                            )
                    consumer.subscribe(['from_edge_to_device'])
                    
                    producer.send( 'test', "hola".encode() )
                    try:
                        
                        _y = []
                        start_global_time = timeit.default_timer()
                        times = []
                        for input_i in x:

                            start_prediction_time = timeit.default_timer()
                            producer.send( 'from_device_to_edge', json.dumps( input_i.tolist() ).encode() )
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
                    # producer.flush()
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
            if args.edge:
                print( "EDGE" )
                        # consumer = KafkaConsumer( bootstrap_servers=['localhost:9092'],
                        #                         value_deserializer=lambda m: json.loads( m.decode() ),
                        #                         auto_offset_reset='earliest',
                        #                         enable_auto_commit=True,
                        #                         auto_commit_interval_ms=1000,
                        #                         group_id='edge'
                        #                         )
                        # consumer.subscribe(['fog_result'])

                producer = KafkaProducer( bootstrap_servers=['localhost:9092'] , max_request_size=1024000000)
                consumer = KafkaConsumer( 'from_device_to_edge', bootstrap_servers=['localhost:9092'],
                                        # value_deserializer=lambda m: json.loads( m.decode() ),
                                        auto_offset_reset='earliest',
                                        enable_auto_commit=True,
                                        auto_commit_interval_ms=1000,
                                        max_partition_fetch_bytes=1024000000,
                                        group_id='edge'
                                        )
                # consumer.subscribe([])
                if args.next_device:
                    producer_next = KafkaProducer( bootstrap_servers=['localhost:9092'],  max_request_size=1024000000 )
                    consumer_next = KafkaConsumer( bootstrap_servers=['localhost:9092'],
                                        # value_deserializer=lambda m: json.loads( m.decode() ),
                                        auto_offset_reset='earliest',
                                        enable_auto_commit=True,
                                        auto_commit_interval_ms=1000,
                                        max_partition_fetch_bytes=1024000000,
                                        group_id='edge'
                                        )
                    consumer_next.subscribe(['from_fog_to_edge'])
                try:

                    print( "Waiting for inputs...")
                    print( consumer )
                    for msg in consumer:

                        if msg != {} or msg is not None:

                            input_i = np.array( json.loads( msg.value.decode() ) )
                            # print( "Input Received: ", input_i )
                            start_prediction_time = timeit.default_timer()
                            _x = input_i
                            result_x = _x
                            # print( models )
                            for model in models:
                                # print( model )
                                _x = model.predict( _x )
                                if len( _x ) > 1:
                                    if _x[-1:][0].max() >= args.threshold:
                                        result_x = _x[-1:][0][0]
                                        break
                                else:
                                    result_x = _x[0][0]
                            
                            result = { "result": result_x.tolist(), "device": "edge", "time": timeit.default_timer() - start_prediction_time }
                            if args.next_device:
                                # Send data to FOG
                                print( "to_fog" )
                                producer_next.send( 'from_edge_to_fog', json.dumps( _x[:-1].tolist() ).encode() )
                                producer_next.flush()
                            producer.flush()

                                for m in consumer_next:
                                    load_message = json.loads(m.value.decode())
                                    if load_message != {} or load_message is not None:
                                        if load_message.get("result", None) != None:
                                            result["next"] = load_message
                                            break
                            #                         group_id='edge'
                            #                         )
                            # consumer.subscribe(['fog_result'])

                            # Return result to device
                            print( "to_device:\n\t", result )
                            producer.send( 'from_edge_to_device', json.dumps( result ).encode() )
                            producer.flush()
                finally:
                    consumer.close()
                    if args.next_device:
                        consumer_next.close()
            elif args.fog:
                
                 ## Load models
                models = []
                for model_path in args.models:
                    models.append( tf.keras.models.load_model( model_path ) )

                if args.edge:
                    print( "FOG" )

                    producer = KafkaProducer( bootstrap_servers=['localhost:9092'] , max_request_size=1024000000)
                    consumer = KafkaConsumer( 'from_edge_to_fog', bootstrap_servers=['localhost:9092'],
                                            # value_deserializer=lambda m: json.loads( m.decode() ),
                                            auto_offset_reset='earliest',
                                            enable_auto_commit=True,
                                            auto_commit_interval_ms=1000,
                                            max_partition_fetch_bytes=1024000000,
                                            group_id='fog'
                                            )
                    # consumer.subscribe([])
                    if args.next_device:
                        producer_next = KafkaProducer( bootstrap_servers=['localhost:9092'],  max_request_size=1024000000 )
                        consumer_next = KafkaConsumer( bootstrap_servers=['localhost:9092'],
                                            # value_deserializer=lambda m: json.loads( m.decode() ),
                                            auto_offset_reset='earliest',
                                            enable_auto_commit=True,
                                            auto_commit_interval_ms=1000,
                                            max_partition_fetch_bytes=1024000000,
                                            group_id='fog'
                                            )
                        consumer_next.subscribe(['from_cloud_to_fog'])
                    try:

                        print( "Waiting for inputs...")
                        print( consumer )
                        for msg in consumer:

                            if msg != {} or msg is not None:

                                input_i = np.array( json.loads( msg.value.decode() ) )
                                # print( "Input Received: ", input_i )
                                start_prediction_time = timeit.default_timer()
                                _x = input_i
                                result_x = _x
                                # print( models )
                                for model in models:
                                    # print( model )
                                    _x = model.predict( _x )
                                    if len( _x ) > 1:
                                        if _x[-1:][0].max() >= args.threshold:
                                            result_x = _x[-1:][0][0]
                                            break
                                    else:
                                        result_x = _x[0][0]
                                
                                result = { "result": result_x.tolist(), "device": "fog", "time": timeit.default_timer() - start_prediction_time }
                                if args.next_device:
                                    # Send data to CLOUD
                                    print( "to_cloud" )
                                    producer_next.send( 'from_fog_to_cloud', json.dumps( _x[:-1].tolist() ).encode() )
                                    producer_next.flush()

                                    for m in consumer_next:
                                        load_message = json.loads(m.value.decode())
                                        if load_message != {} or load_message is not None:
                                            if load_message.get("result", None) != None:
                                                result["next"] = load_message
                                                break

                                # Return result to edge
                                print( "to_edge:\n\t", result )
                                producer.send( 'from_fog_to_edge', json.dumps( result ).encode() )
                                producer.flush()
                    finally:
                        consumer.close()
                        if args.next_device:
                            consumer_next.close()

            elif args.cloud:
                 ## Load models
                models = []
                for model_path in args.models:
                    models.append( tf.keras.models.load_model( model_path ) )
                if args.edge:
                    print( "CLOUD" )

                    producer = KafkaProducer( bootstrap_servers=['localhost:9092'] , max_request_size=1024000000)
                    consumer = KafkaConsumer( 'from_fog_to_cloud', bootstrap_servers=['localhost:9092'],
                                            # value_deserializer=lambda m: json.loads( m.decode() ),
                                            auto_offset_reset='earliest',
                                            enable_auto_commit=True,
                                            auto_commit_interval_ms=1000,
                                            max_partition_fetch_bytes=1024000000,
                                            group_id='cloud'
                                            )
                    
                    try:

                        print( "Waiting for inputs...")
                        print( consumer )
                        for msg in consumer:

                            if msg != {} or msg is not None:

                                input_i = np.array( json.loads( msg.value.decode() ) )
                                # print( "Input Received: ", input_i )
                                start_prediction_time = timeit.default_timer()
                                _x = input_i
                                result_x = _x
                                # print( models )
                                for model in models:
                                    # print( model )
                                    _x = model.predict( _x )
                                    if len( _x ) > 1:
                                        if _x[-1:][0].max() >= args.threshold:
                                            result_x = _x[-1:][0][0]
                                            break
                                    else:
                                        result_x = _x[0][0]
                                
                                result = { "result": result_x.tolist(), "device": "cloud", "time": timeit.default_timer() - start_prediction_time }
                                
                                # Return result to device
                                print( "to_fog:\n\t", result )
                                producer.send( 'from_cloud_to_fog', json.dumps( result ).encode() )
                                producer.flush()
                    finally:
                        consumer.close()
                        if args.next_device:
                            consumer_next.close()
            else:
                error_input( parser, 2 )

if __name__ == '__main__':
    main()
