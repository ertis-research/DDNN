# REFERENCES:
# https://www.tensorflow.org/tutorials/images/classification
import argparse

# Required functions of external files:
from model import create_model 
from training_results import show_results

def read_data( path, vsplit, seed, image_height, image_width, batch_size):
    import pathlib
    from tensorflow import keras

    data_dir = pathlib.Path( path )
    train_ds = keras.preprocessing.image_dataset_from_directory( data_dir, validation_split=vsplit, subset="training", seed=seed, 
        image_size=( image_height, image_width ), batch_size=batch_size )

    val_ds = keras.preprocessing.image_dataset_from_directory( data_dir, validation_split=vsplit, subset="validation", seed=seed,
        image_size=( image_height, image_width ), batch_size=batch_size )

    return train_ds, val_ds, train_ds.class_names

def save_model( model, history, output ):
    import json

    model.save( output ) # tf.keras.models.load_model('saved_model/my_model')
    json.dump( history.history, open( output + ".history", "w" )) # json.load to read it
    # if tflite is selected we should add a condition here for converting the model.

def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument( '-i', '--training_set', help="File path of the training images.", required=True )
    parser.add_argument( '-o', '--output', help="File path to save the resulted model.", default="my_model" )
    parser.add_argument( '-b', '--batch_size', type=int, help="Batch size.", default=32 )
    parser.add_argument( '-e', '--epochs', type=int, help="Number of epochs.", default=10 )
    parser.add_argument( '-g', '--image_height', type=int, help="Image height for training the model.", default=180 )
    parser.add_argument( '-w', '--image_width', type=int, help="Image width for training the model.", default=180 )
    parser.add_argument( '-s', '--seed', type=int, help="Seed used in random operations.", default=123)
    parser.add_argument( '-v', '--vsplit', type=int, help="Validation split.", default=0.2)
    
    args = parser.parse_args()

    if args.training_set:

        import tensorflow as tf
        print( "\tNOTE: Using Tensorflow {}.".format( tf.__version__ ) )
        
        train_ds, val_ds, class_names = read_data( args.training_set, args.vsplit, args.seed, args.image_height, args.image_width, args.batch_size )

        model = create_model( args.image_height, args.image_width, len( class_names ))
        # model.summary()

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs= args.epochs
        )

        save_model( model, history, args.output )
        show_results( history, args.epochs )

if __name__ == '__main__': 
    main()
