import numpy as np

def predict_sales(radio, weight, bias):
    return weight*radio + bias

def cost_function(radio, sales, weight, bias):
    companies = len(radio)
    total_error = 0.0
    for i in range(companies):
        total_error += (sales[i] - (weight*radio[i] + bias))**2
    return total_error / companies

def update_weights(radio, sales, weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    companies = len(radio)

    for i in range(companies):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_deriv += 2*radio[i] * ((weight*radio[i] + bias) - sales[i])
        # -2(y - (mx + b))
        bias_deriv += 2*((weight*radio[i] + bias) - sales[i])

    # We subtract because the derivatives point in direction of steepest ascent
    weight_deriv = weight_deriv / companies
    bias_deriv = bias_deriv / companies
    print ("weight derivative: " + str(weight_deriv))
    weight -= (weight_deriv) * learning_rate
    bias -= (bias_deriv) * learning_rate

    return weight, bias

def train(radio, sales, weight, bias, learning_rate, iters):
    cost_history = []

    for i in range(iters):
        weight,bias = update_weights(radio, sales, weight, bias, learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(radio, sales, weight, bias)
        cost_history.append(cost)

        # Log Progress
        print ("iter: "+str(i) + " weight: "+str(weight) + " bias: " + str(bias)  + " cost: " + str(cost))

    return weight, bias, cost_history

def main():
    radio = [1, 2, 3, 4, 5]
    sales = [3, 8, 9, 12, 15]
    weight = 2
    bias = 0
    weight, bias, cost_history = train(radio, sales, weight, bias, 0.001, 20)
    print (weight, bias)

if __name__ == "__main__":
    main()