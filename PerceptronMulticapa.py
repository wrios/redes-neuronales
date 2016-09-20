import csv
import sys
import getopt
import numpy as np

#variables con el tamanio de las capas intermedias.
m = 11
l = 5

def activation(W, X, Y):
#Le aplico la funcion de activacion a todas las neuronas paso a paso haciendo F(resultado de multiplicacion matricial)

#Agrego -1 en todos los vectores que esten del lado izquierdo de una multiplicacion
#para que de esa manera al hacer el producto matricial se reste el umbral de la ultima
#fila de cada Matriz (umbral correspondiente a la neurona representada por esa columna)
	#X[10] = -1

#Calculos de multiplicacion de matrices para obtener resultado
#AGREGAR -1 en el final del vector para que haga la cuenta con el umbral
	Y[0] = nonlin(np.dot(X, W[0]))
	Y[0][m] = -1

#AGREGAR -1 en el final del vector para que haga la cuenta con el umbral
	Y[1] = nonlin(np.dot(Y[0], W[1]))
	Y[1][l] = -1

	Y[2] = nonlin(np.dot(Y[1], W[2]))

	return Y[2]


def batch(W, deltaW, X, Y):
	e = 0
	with open(sys.argv[1], 'rb') as csvfile:
		dataset = csv.reader(csvfile, delimiter=',', quotechar='\'')
		for row in dataset:
#print ', '.join(row)
#Calculo de la diferencia entre el valor output y el esperado
			if row[0] == "B":
				esperado = 1
			else:
				esperado = 0
			X = [float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), -1]
			X = normalizar(X)
			activation(W, X, Y)

			e = e + correction(Y, deltaW, esperado)

	adaptation(W, deltaW)
	return e

def correction(Y, deltaW, esperado):
#Diferencia entre el resultado obtenido y el esperado	
	E = Y[2] - esperado
#Se usa el modulo y se lo eleva al cuadrado, porque el error cuadratico medio es buen indicador	
	e = np.power(np.absolute(E),2)

	for j in range(3, 1):
		
		D = E * nonlin( np.dot(Y[j-1], W[j]), true)

		deltaW[j] = deltaW[j] + lr * np.dot(Y[j-1].T, D)

	return e

def adaptation(W, deltaW):
	for j in range(1, 3):
		W[j] = W[j] + deltaW[j]
		deltaW[j] = 0
	return deltaW

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


def interpretar(Y):
#interpretacion del resultado en letras
	if (Y[2] > 0.5):
		diagnostico = "B"
	else:
		diagnostico = "M"
	
	return diagnostico

def normalizar(X):
	X[0] = (X[0] - 18556.3594594595) / 4445.9064834242
	X[1] = (X[1] - 21288.2324324324) / 6164.3937442774
	X[2] = (X[2] - 113925.113513514) / 28603.5114501387
	X[3] = (X[3] - 828266.510810811) / 389121.133315752
	X[4] = (X[4] - 1789.2648648649) / 481.6933044966
	X[5] = (X[5] - 1881.4783783784) / 609.3791254842
	X[6] = (X[6] - 3667.8054054054) / 1409.4645274337
	X[7] = (X[7] - 4574.2432432433) / 2040.155060359
	X[8] = (X[8] - 734.4297297297) / 262.3439058088
	X[9] = (X[9] - 1555.4054054054) / 613.1253350794
	return X

def main():

	#Matrices con los pesos desde una capa a la siguiente. (los pesos que inciden en una neurona J desde una neurona I).
	#generados en matrices con numeros aleatorios entre 0 y 1.

	#Agrego +1 en todas las matrices que quedan del lado derecho de las multiplicaciones para representar los umbrales
	#de cada neurona (cada neurona es representada por una columna)

	#Vector de entrada (1x10) que contiene los datos de una instancia
	#A leer de archivo CSV
	X = np.zeros(11)

	W = [np.random.rand(11, m+1), np.random.rand(m+1, l+1), np.random.rand(l+1, 1)]
	#deltaW = [np.random.rand(11, m+1), np.random.rand(m+1, l+1), np.random.rand(l+1, 1)]
	deltaW = [np.zeros((11, m+1)), np.zeros((m+1, l+1)), np.zeros((l+1, 1))]

	Y = [np.zeros(m+1), np.zeros(l+1), np.zeros(1)]


	#Sin entrenar
	X = [11779, 29321, 93649, 1300312, 2469, 1071, 4459, 6299, 586, 1506, -1]
	X = normalizar(X)
	activation(W, X, Y)
	print interpretar(Y)
	#Entrenamiento
	print batch(W, deltaW, X, Y)
	X = [11779, 29321, 93649, 1300312, 2469, 1071, 4459, 6299, 586, 1506, -1]
	X = normalizar(X)
	#Post Entrenamiento
	activation(W, X, Y)
	print interpretar(Y)

if __name__ == "__main__":
    main()