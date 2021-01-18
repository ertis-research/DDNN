from statistics import mean 
import numpy as np 

def output( expected_results, results, global_time, prediction_times, output_path ):

    row = "{};{};{};{};{};{};{};{};{};{};{};{}"
    with open(output_path, "w") as f:
        
        f.write( row.format( 
            "Test", "Expected_result", "Prediction_Time", 
            "Edge_Prediction", "Edge_Computation_Time", "Edge_Prediction_Time", 
            "Fog_Prediction", "Fog_Computation_Time", "Fog_Prediction_Time", 
            "Cloud_Prediction", "Cloud_Computation_Time", "Cloud_Prediction_Time" 
        ) )
        f.write("\n")
        for i, values in enumerate( zip(expected_results, results, prediction_times) ):
            
            if not isinstance(values[1], list) and not isinstance(values[1], np.ndarray):
                result = values[1].value
            else:
                result = { "result": values[1] }
            next_device = result.get("next", {})
            next_next_device = next_device.get("next", {})

            #print( result )
            f.write( row.format(
                #i, values[0].argmax(), values[2],
                i, values[0][0], values[2],
                np.array( result.get("result", []) ).argmax() if len( result.get("result", []) ) > 0 else "", result.get("execution-time", ""), result.get("total-time", ""),
                np.array( next_device.get("result", []) ).argmax() if len( next_device.get("result", []) ) > 0 else "", next_device.get("execution-time", ""), next_device.get("total-time", ""),
                np.array( next_next_device.get("result", []) ).argmax() if len( next_next_device.get("result", []) ) > 0 else "", next_next_device.get("execution-time", ""), next_next_device.get("total-time", "") 
            ))        
            f.write("\n")