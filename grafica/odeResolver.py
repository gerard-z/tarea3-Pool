import numpy as np

def eulerArray(h, tiempo, y0, funcion):
    " Arreglo con el método de Euler, acá el tiempo es el intervalo donde se evaluará"
    iterations = int(tiempo / h) + 1 # llegar hasta el paso final

    t_array = np.zeros(iterations)
    y_array = np.zeros(iterations)
    
    y_array[0] = y0

    for i in range(1, iterations):
        t = t_array[i-1]
        y = y_array[i-1]
        t_array[i] = t + h
        y_array[i] = y + h*funcion(t,y)
    
    return t_array, y_array

def RK4Array(h, tiempo, y0, funcion):
    " Arreglo con el método de Runge Kutta 4, acá el tiempo es el intervalo donde se evaluará"
    iterations = int(tiempo / h) + 1 # llegar hasta el paso final

    t_array = np.zeros(iterations)
    y_array = np.zeros(iterations)
    
    y_array[0] = y0

    for i in range(1, iterations):
        t = t_array[i-1]
        y = y_array[i-1]
        t_array[i] = t + h

        k1 = funcion(t, y)
        k2 = funcion(t+h/2, y + h/2 * k1)
        k3 = funcion(t+h/2, y + h/2 * k2)
        k4 = funcion(t+h, y + h * k3)

        y_array[i] = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t_array, y_array

def euler(h, tiempo, y0, funcion):
    " Resolución con el método de Euler"
    return y0 + h*funcion(tiempo, y0)

def RK4(h, tiempo, y0, funcion):
    " Resolución con el método de Runge Kutta 4"
    k1 = funcion(tiempo, y0)
    k2 = funcion(tiempo + h/2, y0 + h/2 * k1)
    k3 = funcion(tiempo + h/2, y0 + h/2 * k2)
    k4 = funcion(tiempo + h, y0 + h * k3)
    return y0 + h/6 * (k1 + 2*k2 + 2*k3 + k4)