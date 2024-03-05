from model import Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def make_prediction(inputs: list[float], outputs: list[float], input_value: float, plot: bool=False) -> Prediction:
    if len(inputs) != len(outputs):
        raise Exception('Length of "inputs" and "outputs" must match.')
    # create dataframe for data
    df = pd.DataFrame({'inputs':inputs, 'outputs': outputs})
    # reshape the data using numpy
    x = np.array(df['inputs']).reshape(-1, 1)
    y = np.array(df['outputs']).reshape(-1, 1)
    # split data into training data to test model
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=0, test_size=.20)
    # initialize the model and test it
    model = LinearRegression()
    model.fit(train_x, train_y)
    # prediction
    y_prediction = model.predict([[input_value]])
    y_line = model.predict(x)
    # test for accuracy
    y_test_prediction = model.predict(test_x)
    # plot
    if plot: 
        display_plot(inputs=x, outputs=y, y_line=y_line)
    
    return Prediction(value=y_prediction[0][0],
                      r2_score=r2_score(test_y, y_test_prediction),
                      slope=model.coef_[0][0],
                      intercept=model.intercept_[0],
                      mean_absolute_error=mean_absolute_error(test_y, y_test_prediction))



def display_plot(inputs: list[float], outputs: list[float], y_line):
    plt.scatter(inputs, outputs, s=12)
    plt.xlabel('Inputs')
    plt.ylabel('Outputs')
    plt.plot(inputs, y_line, color='r')
    # display the plot
    plt.show()

if __name__ == '__main__':
    years: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    earnings: list[int] = [100_000, 125_000, 175_000, 200_000, 225_000, 250_000, 275_000, 300_000, 325_000, 350_000]
    my_input: int = 20
    prediction: Prediction = make_prediction(inputs=years, outputs=earnings, input_value=my_input, plot=False)
    print('Input:', my_input)
    print(prediction)

    # print('Year 30:', prediction.slope * 30)
    # print('Year 40:', prediction.slope * 40)
    # print('Year 50:', prediction.slope * 50)
    print(f'mean absolute error:', prediction.mean_absolute_error)





