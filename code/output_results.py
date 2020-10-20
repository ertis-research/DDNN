from statistics import mean 
import numpy as np 

def output( expected_results, results, global_time, prediction_times ):

    row = "{};{};{};{};{};{};{};{};{};{};{};{}"
    with open("results.csv", "w") as f:
        
        f.write( row.format( 
            "Test", "Expected_result", "Prediction_Time", 
            "Edge_Prediction", "Edge_Computation_Time", "Edge_Prediction_Time", 
            "Fog_Prediction", "Fog_Computation_Time", "Fog_Prediction_Time", 
            "Cloud_Prediction", "Cloud_Computation_Time", "Cloud_Prediction_Time" 
        ) )
        f.write("\n")
        for i, values in enumerate( zip(expected_results, results, prediction_times) ):
            
            result = values[1].value
            next_device = result.get("next", {})
            next_next_device = next_device.get("next", {})

            f.write( row.format(
                i, values[0].argmax(), values[2],
                np.array( result.get("result", []) ).argmax() if result.get("result", []) else "", result.get("execution-time", ""), result.get("total-time", ""),
                np.array( next_device.get("result", []) ).argmax() if next_device.get("result", []) else "", next_device.get("execution-time", ""), next_device.get("total-time", ""),
                np.array( next_next_device.get("result", []) ).argmax() if next_next_device.get("result", []) else "", next_next_device.get("execution-time", ""), next_next_device.get("total-time", "") 
            ))        
            f.write("\n")
    # print( "Expected_results:\n\t{}\n\nResults:\n\t{}\n\nGlobal_time:\n\t{}\n\nPrediction_times:\n\t{}".format(
    #     expected_results[0],
    #     results[0].value,
    #     global_time,
    #     prediction_times[0]
    # ))

def output( expected_results, results, global_time, prediction_times ):

    accuracy = 0
    loss = 0
    for expected_result, result in zip( expected_results, results ):

        result_i = result.argmax()
        expected_result_i = expected_result.argmax()
        accuracy += 1 if expected_result_i == result_i else 0
        loss += ( expected_result[ expected_result_i  ] - result[ result_i ] ) ** 2

    print( "\nTotal number of predictions: {}\nAccuracy: {}\nLoss: {}\nGlobal time: {}\nAverage Time per Prediction: {}".format( 
        len( results ),
        accuracy / len( results ), 
        loss, 
        global_time, 
        mean( prediction_times ) 
    ))