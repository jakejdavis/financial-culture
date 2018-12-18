import quandl
import numpy as np


def abstract_matrix(matrix):
    abstracted_matrix = []
    initial_date = int(matrix[0]["Date"])
    for data_form in matrix:
        abstracted_matrix.append([int(data_form["Date"])-initial_date, data_form["Close"]])
    return np.array(abstracted_matrix)

def values_from_tag(tag):
    # Get stocks
    prices = quandl.get(tag, authtoken="hqUKYvKzjJxm23narJ4x", returns="numpy")[:7500]# Quandl returns DataFrame
    """split_possible = False
    while not split_possible:
        prices = np.delete(prices, 0)
        if len(prices) % 50 == 0:
            split_possible = True"""
    prices = np.array_split(prices, 50)

    values_X = []
    values_Y = []
    for prices_matrix in prices:
        # Abstract matrix
        prices_matrix = abstract_matrix(prices_matrix)

        # Remove last item & set to future
        future, prices_matrix = prices_matrix[-1], prices_matrix[:-1]
        
        # Create y output data

        """diff = future[1]-prices_matrix[-1][1]

        y = [0, 0] # buy, sell
        if diff > diff_activation: 
            y[0] = 1
        if diff < -diff_activation: 
            y[1] = 1"""
            
        values_X.append(prices_matrix)
        values_Y.append(future[1])
        
    return np.array(values_X), np.array(values_Y)

    