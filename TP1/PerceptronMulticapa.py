import csv
import sys
import getopt
import numpy as np
from numpy import linalg as LA 

m = 7
lr = 0.25
W = [np.random.uniform( -0.1, 0.1, (11,m)), np.random.uniform( -0.1, 0.1, (m+1,1))]

def normalizar(row):
	X = (-1)*np.ones((11,1))
	X[0] = (row[0] - 18556.3594594595) / 4445.9064834242
	X[1] = (row[1] - 21288.2324324324) / 6164.3937442774
	X[2] = (row[2] - 113925.113513514) / 28603.5114501387
	X[3] = (row[3] - 828266.510810811) / 389121.133315752
	X[4] = (row[4] - 1789.2648648649) / 481.6933044966
	X[5] = (row[5] - 1881.4783783784) / 609.3791254842
	X[6] = (row[6] - 3667.8054054054) / 1409.4645274337
	X[7] = (row[7] - 4574.2432432433) / 2040.155060359
	X[8] = (row[8] - 734.4297297297) / 262.3439058088
	X[9] = (row[9] - 1555.4054054054) / 613.1253350794
	#X[:-1] 
	X = np.asarray(X)
	return X

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def activation(X,W):	
	K = []
	K.append(X[:-1])
	Y = (-1)*np.ones((m+1,1))
	Y[:-1] = nonlin(np.dot(X.T,W[0]).T)
	K.append(Y[:-1])
	Y2 = nonlin(np.dot(Y.T,W[1]))
	K.append(Y2)
#devuelve la lista con las salidas (sin el menos uno )	
	return K

def correction(Y,esperado):
#salida menos lo esperado
	delta2 = esperado - Y[2]
	delta1 = W[1][:-1] * delta2
	dW[1] = dW[1] - lr*np.dot(Y[1],(nonlin(Y[2].T,True)*delta2).T)
	dW[0] = dW[0] - lr*np.dot(Y[0],(nonlin(Y[1],True)*delta1).T)
	E = np.dot((nonlin(Y[1],True)*delta1).T,W[0][:-1].T)
	e = np.linalg.norm(np.absolute(E),2)
	return e



def adaptation(W):
	for j in range(0,2):
		W[j][:-1] = W[j][:-1] + dW[j]



def batch():
	global W
	global dW
	global X
	global Y
	e = 0
	dW = [np.zeros((10, m)), np.zeros((m, 1))]
	reader = csv.reader(open('/home/kennedy/TP2-master/RNA_TP1_datasets/tp1_ej1_training.csv','rb')) 	
	for row in reader:
#print ', '.join(row)
#Calculo de la diferencia entre el valor output y el esperado
		if row[0] == "B":
			esperado = 1
		else:
			esperado = 0
			#print np.shape(row)
		X = [float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10])]
		N =	normalizar(X)
		A = activation(N,W) # devuelve una lista con los resultados
		e = correction(A,esperado)
	adaptation(W)
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
	cota = 0.000001
	T = 2000
	e = 1
	t = 1

	while e > cota and t < T:
		e = batch()
		t = t + 1
		print t,e
	return e,t	
	#Entrenamiento
#	n = 0
#	error = 1
#	while error >= 0.15 or n <= 10:
#		error = batch()
	
#	normalizar()
#	activation()
	#print interpretar()

if __name__ == "__main__":
    main()