from statistics import mean 

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