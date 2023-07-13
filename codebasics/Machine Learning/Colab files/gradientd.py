import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.08
    m_list = []
    b_list = []

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        m_list.append(m_curr)
        b_list.append(b_curr)

        print(f"m {m_curr}, b {b_curr}, cost {cost} iteration {i}")

    # Plot the iterations of gradient descent
    plt.plot(m_list, b_list)
    plt.xlabel('m')
    plt.ylabel('b')
    plt.title('Gradient Descent')
    plt.show()

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)
