import numpy as np
from sklearn.metrics import r2_score

#Forecasting metriccs 

EPSILON = 1e-10

def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))

def leastSquare(actual, predicted):
    errors  = _error(actual, predicted)
    errrors_squared  = np.square(errors)
    dataset_length = len(actual)
    return errrors_squared / dataset_length

def wape(actual: np.ndarray, predicted: np.ndarray):
    """ Weighted Absolute Percentage Error """
    return mae(actual, predicted) / (np.mean(actual) +EPSILON )

def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

#average r2 of all series(seperately)
def  r2_avr(y_true: np.ndarray, y_pred: np.ndarray):  
    #y_true & y_pred : #(num_series,batch,samples)
    num_series = len(y_true)
    avr_r2 = 0
    for i in range(num_series):
        avr_r2 +=r2_score(y_true[i], y_pred[i])
       # print(str(i) + " " + str(avr_r2) )

    return avr_r2/num_series

def  r2_all(y_true: np.ndarray, y_pred: np.ndarray):       
    return  r2_score(y_true, y_pred)
    return avr_r2/num_series
    #return  r2_score(y_true, y_pred)

def rmse(y_true: np.ndarray,y_pred: np.ndarray):
    return np.sqrt(mse(y_true,y_pred))


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error

    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)

def mape(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.abs(_percentage_error(actual, predicted)))

def displayMetrics(y_true,y_pred):
    print('mse  : {}'.format(mse(y_true, y_pred)))
    print( 'mae  : {}'.format( mae(y_true,y_pred)))
    print('rmse  : {}'.format( rmse(y_true,y_pred)))
    print('wape  : {}'.format( wape(y_true,y_pred)))
    print('mape  : {}'.format( mape(y_true,y_pred)))
    print('avg r2  : {}'.format( r2_avr(y_true,y_pred)))
    err_dic= {
        "mse": mse(y_true, y_pred),
        "mae":  mae(y_true,y_pred),
        "rmse": rmse(y_true,y_pred),
        "wape": wape(y_true,y_pred),
        "mape": mape(y_true,y_pred),
        "r2": r2_avr(y_true,y_pred)
    }
    return err_dic