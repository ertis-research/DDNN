# Run your model in an edge-fog-cloud architecture

This code has been developed to facilitate the inference of BranchyNet-based models [[1]][[2]] over different architecture levels, such as edge, fog and cloud. 

## Getting started

First, to run this code, we need to install dependencies included in the `requirements.txt`. We recommend the use of Python virtual environment for this.

```
(.venv)$ pip install -r requirements.txt
```

Once libraries are installed, we should be able to train a model as the one placed on `model.py`. We can modify the content of the function as we want, but we have to keep the first and the last value this function returns - of course, we can change their values. To train, we run:

```
(.venv)$ python code/model_training.py -i <PATH_TO_TRAINING_SET> [<TRAINING OPTIONS>]
```

After training, our model and submodels generated will be saved in the folder `saved_models` by default. If we want to modified the way we show the metrics during training, we can replace the function in `training_results.py`.

Finally, to perform inference with our model - where the input (*preprocessing*) and ouput can be modified in `input_preprocessing.py` and `output_results.py`, respectively -, we only need to call the `main.py` file:

```
python code/main.py -i <PATH_TO_TEST_SET> --labels <LABEL NAME FILE> --models saved_models/model --name  <NAME_DEVICE> --tensorflow
```

This will generate a CSV file according with our `output_results.py` function. However, it will contains a lot of blank spaces due to some of the values recorded in the CSV file is for our code when using edge, fog and cloud devices. In order to do that, we allow our code to send and receive from different Kafka servers specified in the arguments of the `main.py` program:

- `--producer-back`. IP to send values to the previous device - `localhost` by default.
- `--producer-back-port`. Port to send values to the previous device - default=`9092`.
- `--producer-back-topic`. Topic to send values to the previous device.
- `--producer-front`. IP to send values to the next device.
- `--producer-front-port`. Port to send values to the next device.
- `--producer-front-topic`. Topic to send values to the previous device.
    
- `--consumer-back`. IP to receive values from the previous device - `localhost` by default.
- `--consumer-back-port`. Port to send values to the previous device - default=`9092`.
- `--consumer-back-topic`. Topic to send values to the previous device.
- `--consumer-front`. IP to receive values from the next device.
- `--consumer-front-port`. Port to send values to the next device.
- `--consumer-front-topic`. Topic to send values to the previous device.

### Examples

> Using the flower dataset of tensorflow [[3]]

- If you want to try our code in your localhost but with different terminals:

    - Emulates a simple device that only capture and send values to the edge:
    ```
    python code/main.py -i data/flower_photos/train -l data/labels.txt --name device --producer-front localhost --producer-front-port 9092 --producer-front-topic from_device_to_edge --consumer-front localhost --consumer-front-port 9092 --consumer-front-topic from_edge_to_device
    ```

    - Emulates the edge service, receiving from the device and sending to the fog if prediction is not higher than the specified threshold (`0.8` by default), receiving from the fog and sending back the results to the device: 

    ```
    python code/main.py --models saved_models/edge --name local_edge --producer-back-topic from_edge_to_device --consumer-back-topic from_device_to_edge --producer-front localhost --producer-front-port 9092 --producer-front-topic from_edge_to_fog --consumer-front localhost --consumer-front-port 9092 --consumer-front-topic from_fog_to_edge --tensorflow
    ```

    - Emulates the fog service, receiving from the edge, making the inference,  sending to cloud if needed, and receiving from the cloud and sending back the results to the edge:
    ```
    python code/main.py --models saved_models/fog --name local_fog --producer-back-topic from_fog_to_edge --consumer-back-topic from_edge_to_fog --producer-front localhost --producer-front-port 9092 --producer-front-topic from_fog_to_cloud --consumer-front localhost --consumer-front-port 9092 --consumer-front-topic from_cloud_to_fog --tensorflow
    ```
    - Emulates the cloud service, receving from the fog, making the inference and sending the results back to the fog:
    ```
    python code/main.py --models saved_models/cloud --name local_cloud --producer-back-topic from_cloud_to_fog --consumer-back-topic from_fog_to_cloud --tensorflow
    ```

- Another example of use when we got a device and only an additional computer, either edge, fog or cloud:
    - Emulates the device:
    ```
    python code/main.py -i data/flower_photos/train -l data/labels.txt --name device --producer-front localhost --producer-front-port 9092 --producer-front-topic from_device_to_edge --consumer-front localhost --consumer-front-port 9092 --consumer-front-topic from_edge_to_device
    ```

    - Emulates the external computer with more resources, it receives from the device, computes the result and sends it back:
    ```
    python code/main.py --models saved_models/edge saved_models/fog saved_models/cloud --name local_edge --producer-back-topic from_edge_to_device --consumer-back-topic from_device_to_edge --tensorflow
    ```

## Contributions

We are still working on this code to make it more useful and practical for the scientific community. We thank any feedback and contribution to our project.

----

This a project of [ERTIS Lab](https://github.com/ertis-research) at [University of Malaga](uma.es)

[1]: https://arxiv.org/abs/1709.01686
[2]: https://github.com/kunglab/branchynet
[3]: https://www.tensorflow.org/tutorials/load_data/images
