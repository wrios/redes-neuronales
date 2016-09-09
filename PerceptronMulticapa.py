import sys
import getopt
import numpy as np


#variables con el tamanio de las capas intermedias.
m = 3
l = 2
#Matrices con los pesos desde una capa a la siguiente. (los pesos que inciden en una neurona J desde una neurona I).
#generados en matrices con numeros aleatorios entre 0 y 1.

#Agrego +1 en todas las matrices que quedan del lado derecho de las multiplicaciones para representar los umbrales
#de cada neurona (cada neurona es representada por una columna)

#Vector de entrada (1x10) que contiene los datos de una instancia
#A leer de archivo CSV
X = np.random.rand(11)

W = [np.random.rand(11, m), np.random.rand(m+1, l), np.random.rand(l+1, 1)]

Y = [np.zeros(m+1), np.zeros(l), np.zeros(1)]

#Matriz de 10xm que representa el peso de las entradas en las neuronas de la primera capa intermedia (oculta)
#W[O] = np.random.rand(10, m+1)

#Matriz de mxl que reprensenta el peso de las neuronas de la primera capa intermedia en la segunda.
#W[1] = np.random.rand(m, l+1)

#Matriz lx1 que representa el peso de las neuronas de la segunda capa en la neurona de salida que es unica por ser un problema de decision binario.
#W[2] = np.random.rand(l, 1+1)

#TODO definir Y si llega a fallar la ejecucion

def activation():
#Le aplico la funcion de activacion a todas las neuronas paso a paso haciendo F(resultado de multiplicacion matricial)

#Agrego -1 en todos los vectores que esten del lado izquierdo de una multiplicacion
#para que de esa manera al hacer el producto matricial se reste el umbral de la ultima
#fila de cada Matriz (umbral correspondiente a la neurona representada por esa columna)
	X[10] = -1

#Calculos de multiplicacion de matrices para obtener resultado
#AGREGAR -1 en el final del vector para que haga la cuenta con el umbral
	Y[0] = nonlin(np.dot(X, W[0]))
	Y[0][m+1] = -1

#AGREGAR -1 en el final del vector para que haga la cuenta con el umbral
	Y[1] = nonlin(np.dot(Y[0], W[1]))
	Y[1][l+1] = -1

	Y[2] = nonlin(np.dot(Y[1], W[2]))

	return Y[2]


def batch(dataset):
	e = 0
	for d in dataset:
#Calculo de la diferencia entre el valor output y el esperado
		if d[Z] == "B":
			esperado = 1
		else:
			esperado = 0

		activation(d[X])

		e = e + correction(esperado)

	adaptation()
	return e

def correction(esperado):
#Diferencia entre el resultado obtenido y el esperado	
	E = Y[2] - esperado
#Se usa el modulo y se lo eleva al cuadrado, porque el error cuadratico medio es buen indicador	
	e = np.power(np.absolute(E),2)

	for j in range(3, 1):
		
		D = E * nonlin( np.dot(Y[j-1], W[j]), true)

		deltaW[j] = deltaW[j] + lr * np.dot(Y[j-1].T, D)

	return e

def adaptation():
	for j in range(1, 3):
		W[j] = W[j] + deltaW[j]
		deltaW[j] = 0

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


def interpretar():
#interpretacion del resultado en letras
	if (Y[2] > 1/2):
		diagnostico = "B"
	else:
		diagnostico = "M"
	
	return diagnostico

def main():
#    # parse command line options
#    try:
#        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
#    except getopt.error, msg:
#        print msg
#        print "for help use --help"
#        sys.exit(2)
#    # process options
#    for o, a in opts:
#        if o in ("-h", "--help"):
#            print __doc__
#            sys.exit(0)
#    # process arguments
#    for arg in args:
#        process(arg) # process() is defined elsewhere
	print activation()

if __name__ == "__main__":
    main()