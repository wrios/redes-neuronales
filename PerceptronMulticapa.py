import csv
import sys
import getopt
import numpy as np

#variables con el tamanio de las capas intermedias.
m = 13
l = 5
lr = 0.15

#Dim(deltaW[0]) = 
W = [np.random.rand(10, m), np.random.rand(m, l), np.random.rand(l, 1)]
deltaW = [np.zeros((10, m)), np.zeros((m, l)), np.zeros((l, 1))]
umbrales = [np.random.rand(1,m), np.random.rand(1, l), np.random.rand(1, 1)]
#La ultima fila de deltaW tiene que ser siempre 0's

Y = [np.zeros((11,1)), np.zeros((m+1,1)), np.zeros((l+1,1)), np.zeros((1,1))]

X = np.zeros((10,1))

def normalizar():
	global X
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

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def activation():
	global Y
	global X
	global W
	#Agrego los umbrales
	W[0] = np.vstack([W[0], umbrales[0]])
	W[1] = np.vstack([W[1], umbrales[1]])
	W[2] = np.vstack([W[2], umbrales[2]])
	#Le aplico la funcion de activacion a todas las neuronas paso a paso haciendo F(resultado de multiplicacion matricial)
	Y[0] = np.insert(X, 10,-1)
  	Y[1] = nonlin(np.dot(Y[0], W[0]))
  	#arriba Y[1] tiene que tener tamanio m, abajo tiene tamanio m+1
  	Y[1] = np.insert(Y[1], m, -1)
  	Y[2] = nonlin(np.dot(Y[1], W[1]))
  	#arriba Y[2] tiene que tener tamanio l, abajo tiene tamanio l+1
  	Y[2] = np.insert(Y[2], l, -1)
  	#Tiene tamanio 1!
  	Y[3] = np.asmatrix(nonlin(np.dot(Y[2], W[2])))
  	#Quitar los -1 de las Y
  	#Tiene que quedar con tamanio 10
	Y[0] = np.asmatrix(np.delete(Y[0], 10, -1))
    #Tiene que quedar con tamanio m
  	Y[1] = np.asmatrix(np.delete(Y[1], m, -1))
  	#Tiene que quedar con tamanio l
  	Y[2] = np.asmatrix(np.delete(Y[2], l , -1))
	#Borramos las filas l y m que tiene los umbrales
	W[0] = np.delete(W[0], (10), axis = 0)
	W[1] = np.delete(W[1], (m), axis = 0)
	W[2] = np.delete(W[2], (l), axis = 0)
  	return Y

def correction(esperado):
	global Y
	global deltaW
	global W
	#Diferencia entre el resultado obtenido y el esperado	
	E = Y[3] - esperado
	#Se usa el modulo y se lo eleva al cuadrado, porque el error cuadratico medio es buen indicador	
	e = np.power(np.absolute(E),2)
	for j in range(3, 0,-1):
		D = E * np.dot(Y[j].T,(1-Y[j])) #nonlin( np.dot(Y[j-1], W[j]), True) # Y[j](1-Y[j])
		deltaW[j-1] = deltaW[j-1] + lr * Y[j-1].T * D
	return e

def adaptation():
	global W
	global deltaW
	for j in range(1, 3):
		W[j] = W[j] + deltaW[j]
	deltaW = [np.zeros((10, m)), np.zeros((m, l)), np.zeros((l, 1))]
	return deltaW

def batch():
	global W
	global deltaW
	global X
	global Y
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
			X = [float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10])]
			normalizar()
			activation()
			e = e + correction(esperado)
	adaptation()
	return e

def interpretar():
	global Y
#interpretacion del resultado en letras
	if (Y[3] > 0.5):
		diagnostico = "B"
	else:
		diagnostico = "M"
	
	return diagnostico


def main():



	#Sin entrenar
	X = [11779, 29321, 93649, 1300312, 2469, 1071, 4459, 6299, 586, 1506]
	normalizar()
	activation()
	print interpretar()
	
	#Entrenamiento
	n = 0
	error = 1
	while error >= 0.15 or n <= 1000:
		error = batch()
	
	#Post Entrenamiento
	X = [11779, 29321, 93649, 1300312, 2469, 1071, 4459, 6299, 586, 1506]
	normalizar()
	activation()
	print interpretar()

if __name__ == "__main__":
    main()