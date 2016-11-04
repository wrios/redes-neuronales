import csv
import sys
import getopt
import numpy as np
from numpy import linalg as LA 

cota = 0.000001
T = 2000
m = 30
Y = [np.random.uniform( -0.1, 0.1, (11,1)), np.random.uniform( -0.1, 0.1, (m+1,1)), np.random.uniform(-0.1, 0.1, (1,1))]
W = [np.random.uniform( -0.1, 0.1, (11,m)), np.random.uniform( -0.1, 0.1, (m+1,1))]
dW = [np.zeros((11, m)), np.zeros((m+1, 1))]
esperados = []
resultados = []
datos = []


def levantarYNormalizar():
	global datos
	reader = csv.reader(open('RNA_TP1_datasets/tp1_ej1_training.csv','rb'))
	for vector in reader:
		temp = np.zeros((12))
		if vector[0] == 'B':
			temp[0] = 1
		else:
			temp[0] = -1
		temp [1:]= normalizar(np.asarray(vector[1:]))
		datos.append(temp) 


def normalizar(row):
	X = (-1)*np.ones((11))
	X[0] = (float(row[0]) - 18556.3594594595) / 4445.9064834242
	X[1] = (float(row[1]) - 21288.2324324324) / 6164.3937442774
	X[2] = (float(row[2]) - 113925.113513514) / 28603.5114501387
	X[3] = (float(row[3]) - 828266.510810811) / 389121.133315752
	X[4] = (float(row[4]) - 1789.2648648649) / 481.6933044966
	X[5] = (float(row[5]) - 1881.4783783784) / 609.3791254842
	X[6] = (float(row[6]) - 3667.8054054054) / 1409.4645274337
	X[7] = (float(row[7]) - 4574.2432432433) / 2040.155060359
	X[8] = (float(row[8]) - 734.4297297297) / 262.3439058088
	X[9] = (float(row[9]) - 1555.4054054054) / 613.1253350794
	X = np.asarray(X)
	return X

def nonlin(x,deriv=False):
    if(deriv==True):
    	#print x
        return 1-(x*x)
    return np.tanh(x)

def activation(X,W):
	global Y
	Y = []
	Y.append(X.reshape(11,1))
	temp = (-1)*np.ones((m+1, 1))
	temp[:-1] = np.dot(Y[0].T, W[0]).T
	Y.append(nonlin(temp))
	Y.append(nonlin(np.dot(Y[1].T, W[1])))
	return Y

def correction(Y,Z,n):
	global dW
	E_2 = Z - Y[2]
#salida menos lo esperado
	delta2 = nonlin(Y[2],True)*E_2
	#print np.dot(delta2.T, W[1].T).shape
	#print 'Y[1]: ', Y[1]
	delta1 = (nonlin(Y[1],True)*(np.dot(delta2.T, W[1].T).T))[:-1]	
	#print 'delta1.shape: ', delta1.shape
	#print 'Y[1].shape: ', Y[1].shape
	dW[1] = learningRate(n)*np.dot(Y[1],delta2.T)
	#print 'Y[0].shape: ', Y[0].shape
	#print 'dW[1].shape: ', dW[1].shape
	dW[0] = learningRate(n)*np.dot(Y[0],delta1.T)
	#print 'dW[0].shape: ', dW[0].shape
	e = np.linalg.norm(E_2,2)
	return e

def adaptation(W):
	for j in range(0,2):
		#print W[j].shape, dW[j].shape
		W[j] += dW[j]

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben

def incremental(n):
	global W
	global dW
	global X
	global Y
	e = 0
	for i in np.random.permutation(410):
		row = datos[i]
		dW = [np.zeros((10, m)), np.zeros((m, 1))]
#Calculo de la diferencia entre el valor output y el esperado
		esperado = row[0]
		X = row[1:]
		np.reshape(X, (11,1))
		A = activation(X,W) # devuelve una lista con los resultados
		e += correction(A,esperado, n)
		adaptation(W)
	return e

def interpretar(Y):
#interpretacion del resultado en letras
	if (Y[2] > 0):
		diagnostico = "B"
	else:
		diagnostico = "M"
	return diagnostico

def entrenamiento():
	t = 1
	e = 1
	ePrima = 1
	ciclosSinCambios = 0
	while e > cota and t < T:
		e = incremental(t)
		if ePrima == e:
			ciclosSinCambios += 1
		else:
			ePrima = e
			ciclosSinCambios = 0
		if ciclosSinCambios > 20:
			print 'Pasaron 20 ciclos sin cambios en el error	'
			break
		progress(t, T, 'e: '+str(e))
		t = t + 1
	return e,t

def cicloCompleto():
	global W
	global X
	global esperados
	global resultados
	esperados = []
	resultados = []
	reader = csv.reader(open('RNA_TP1_datasets/tp1_ej1_training.csv','rb'))
	for row in reader:
		X = [float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10])]
		N =	normalizar(X)
		A = activation(N,W) # devuelve una lista con los resultados
		esperados.append(row[0])
		resultados.append(interpretar(A))
		#print 'resultado:', row[0], ' resultado ', interpretar(A), A[2]#interpretar(A)
	return esperados, resultados

def precision(esperados, resultados):
	correctos = 0
	totales = 0
	for i in xrange(0,len(esperados)):
		if esperados[i] == resultados[i]:
			correctos += 1
		totales += 1
	print 'Total: ', totales, ' Correctos: ', correctos
	return totales, correctos

def learningRate(n):
	lr = 0.1 #1/(np.exp(n))
	return lr

def main():
	global datos
	print 'Levantar y normalizar'
	levantarYNormalizar()
	print 'Termina de normalizar'
	print 'Pre entrenamiento'
	esperados, resultados = cicloCompleto()
	precision(esperados, resultados)
	print 'Comienza Entrenamiento'
	entrenamiento()
	print 'Finaliza Entrenamiento'
	print 'Post entrenamiento'
	esperados, resultados = cicloCompleto()
	precision(esperados, resultados)

if __name__ == "__main__":
    main()