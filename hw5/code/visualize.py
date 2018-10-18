import numpy as np
import matplotlib.pyplot as plt

def plot_hidden_units():
    x = [5,20,50,100,200]
    y = [0.5531343616356629, 0.1279839035046264, 0.05322168613981989, 0.04615925466447202, 0.04603373366176334]
    y1 = [0.7238621783217674, 0.5835209349422685, 0.44735377719429076, 0.44815397251439854, 0.42468312752322246]

    plt.plot(x,y,color='r',marker='o')
    plt.plot(x,y1,color='b',marker='v')
    plt.xlabel('Number of hidden units')
    plt.ylabel('Mean Cross Entropy')
    plt.legend(['Train Set', 'Test Set'])
    plt.show()

def plot_learning_rate():
    fileName = '../results/modelLarge0001Metrics.txt'
    with open(fileName, 'r') as f:
        count = 0
        y = []
        y1 = []
        for line in f:
            val = float(line.split(':')[1])
            rest = line.split(':')[0]
            if "error" in rest:
                continue
            else:
                if count%2 == 0:
                    y.append(val)
                else:
                    y1.append(val)
            count+=1

    x = np.linspace(1,100,100)
    plt.plot(x,y,color='r')
    plt.plot(x,y1,color='b')
    plt.xlabel('Number of epochs')
    plt.ylabel('Mean Cross Entropy')
    plt.legend(['Train Set', 'Test Set'])
    plt.title('Learning-Rate = 0.001')
    plt.show()

if __name__ == "__main__":
    # plot_hidden_units()
    plot_learning_rate()