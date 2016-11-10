import csv
import os.path
import sys
import getopt
import itertools
import numpy as np
from numpy import linalg as LA 
import matplotlib.pylab as plt

in_T = 10000
in_m = 120
in_lr = 0.05
lr = in_lr
cota = 0.000001
in_inicio_validacion = 0
in_fin_validacion = 0
tolerancia = 2
data_min1 = 99
data_max1 = 0
data_min2 = 99
data_max2 = 0
fname = 'p2_'+str(in_m)+'_'+str(in_lr)+'_'+str(in_T)
in_dataset = 'RNA_TP1_datasets/tp1_ej2_training.csv'
in_train_dataset = 'RNA_TP1_datasets/tp1_ej2_training.csv'
Y = [np.random.uniform( -0.1, 0.1, (9,1)), np.random.uniform( -0.1, 0.1, (in_m+1,1)), np.random.uniform(-0.1, 0.1, (1,2))]
W = [np.random.uniform( -0.1, 0.1, (9,in_m)), np.random.uniform( -0.1, 0.1, (in_m+1,2))]
dW = [np.zeros((9, in_m)), np.zeros((in_m+1, 2))]
esperados = []
resultados = []
datos = []

def inicializarMatrices():
	global Y
	global W
	global dW
	Y = [np.random.uniform( -0.1, 0.1, (9,1)), np.random.uniform( -0.1, 0.1, (in_m+1,1)), np.random.uniform(-0.1, 0.1, (1,2))]
	W = [np.random.uniform( -0.1, 0.1, (9,in_m)), np.random.uniform( -0.1, 0.1, (in_m+1,2))]
	dW = [np.zeros((9, in_m)), np.zeros((in_m+1, 2))]

def guardarEntrenamiento():
	if not(checkFile('training/'+fname)):
		print 'Guardando Entrenamiento'
		outfile = open('training/'+fname, 'w')
		np.save(outfile, W)
		outfile.close()
		print 'Entrenamiento Guardado'

def valorAceptable(x, y):
	if np.linalg.norm(np.fabs(x-y)) < tolerancia:
		return True
	else:
		return False

def cargarEntrenamiento():
	global W
	if checkFile('training/'+fname):
		print 'Cargando Entrenamiento'
		infile = open('training/'+fname, 'r')
		W[:] = np.load(infile)
		infile.close()
		print 'Entrenamiento Cargado'
		return True
	return False

def checkFile(fname):
	if os.path.exists(fname):
		return os.path.isfile(fname)
	else:
		return False

def levantarYNormalizar():
	global datos
	global data_min1
	global data_max1
	global data_min2
	global data_max2
	primero = True
	i = 0
	reader = csv.reader(open(in_train_dataset,'rb'))
	for vector in reader:
		if float(vector[8]) < data_min1:
			data_min1 = float(vector[8])
			primero = True
		if float(vector[9]) < data_min2:
			data_min2 = float(vector[9])
			primero = False

		if float(vector[8]) > data_max1:
			data_max1 = float(vector[8])
			primero = True
		if float(vector[9]) > data_max2:
			data_max2 = float(vector[9])
			primero = False

	reader = csv.reader(open(in_train_dataset,'rb'))
	for vector in reader:
		temp = np.zeros((11))
		temp[:-2]= normalizar(np.asarray(vector[:]))
		temp[9] = -1 + 2*((float(vector[8]) - data_min1) / (data_max1 - data_min1))
		temp[10] = -1 + 2*((float(vector[9]) - data_min2) / (data_max2 - data_min2))
		if (in_dataset != in_train_dataset):
			datos.append(temp)
		elif not(in_inicio_validacion <= i <= in_fin_validacion):
			datos.append(temp)
		i += 1

def normalizar(row):
	X = (-1)*np.ones((9))
	X[0] = (float(row[0]) - 0.76) / 0.103722197
	X[1] = (float(row[1]) - 673.41) / 86.90758141
	X[2] = (float(row[2]) - 318.35) / 44.10
	X[3] = (float(row[3]) - 177.53) / 44.61500609
	X[4] = (float(row[4]) - 5.24) / 1.751696572
	X[5] = (float(row[5]) - 3.42) / 1.106975514
	X[6] = (float(row[6]) - 0.23) / 0.133087695
	X[7] = (float(row[7]) - 2.77) / 1.56
	X = np.asarray(X)
	return X

def denormalizar(R):
	R[0][0] = (((R[0][0] + 1) / 2 ) * (data_max1-data_min1)) + data_min1
	R[0][1] = (((R[0][1] + 1) / 2 ) * (data_max2-data_min2)) + data_min2
	#print R

def nonlin(x,deriv=False):
    if(deriv==True):
    	#print x
        return 1-(x*x)
    return np.tanh(x)

def activation(X,W):
	global Y
	Y = []
	Y.append(X.reshape(9,1))	
	temp = (-1)*np.ones((in_m+1, 1))
	temp[:-1] = np.dot(Y[0].T, W[0]).T
	temp[:-1] = nonlin(temp[:-1])
	Y.append(temp)
	Y.append(nonlin(np.dot(Y[1].T, W[1])))
	return Y

def correction(Y,Z,b=False):
	global dW
	E_2 = Z - Y[2]
	delta2 = nonlin(Y[2],True)*E_2
	delta1 = (nonlin(Y[1],True)*(np.dot(delta2, W[1].T).T))[:-1]	
	dW[1] = learningRate(b)*np.dot(Y[1],delta2)
	dW[0] = learningRate(b)*np.dot(Y[0],delta1.T)
	e = np.linalg.norm(E_2,2)
	return e

def adaptation(W):
	for j in range(0,2):
		W[j] += dW[j]

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben

def graficarResultados(esperados, resultados):
	fig = plt.figure()
	plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])
	esperados.sort()
	resultados.sort(key=lambda tup: tup[1])
	plt.plot(esperados)
	plt.plot(resultados)
	plt.legend(['esperados_v1', 'esperados_v2', 'resultados_v1', 'resultados_v2'], loc='upper left')
	plt.show()
	plt.savefig('results/'+fname+'.png', format='png')

def incremental(b):
	global W
	global dW
	global X
	global Y
	e = 0
	for i in np.random.permutation(len(datos)):
		row = datos[i]
		dW = [np.zeros((10, in_m)), np.zeros((in_m, 1))]
		#Calculo de la diferencia entre el valor output y el esperado
		esperado = row[9:]
		X = row[:-2]
		np.reshape(X, (9,1))
		A = activation(X,W) # devuelve una lista con los resultados
		if ((esperado[0] > 1) or (esperado[1]>1)): 
			print 'A, esperado: ', A[2], esperado
		e += correction(A,esperado, b)
		adaptation(W)
	return e#/len(datos)

def entrenamiento():
	print 'Comienza Entrenamiento'
	t = 1
	e = 1
	ePrima = 1
	agrandarLR = False
	while e > cota and t < in_T:
		e = incremental(agrandarLR)
		if e-ePrima < 1:
			agrandarLR = True
		else:
			ePrima = e
			agrandarLR = False
		progress(t, in_T, 'e: '+str(e))
		t = t + 1
	print 'Finaliza Entrenamiento'
	return e,t

def cicloCompleto():
	global W
	global X
	global esperados
	global resultados
	esperados = []
	resultados = []
	reader = csv.reader(open(in_dataset,'rb'))
	i = 0
	for row in reader:
		X = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])]
		N =	normalizar(X)
		A = activation(N,W) # devuelve una lista con los resultados
		if (in_dataset != in_train_dataset):
			esperados.append([float(row[8]), float(row[9])])
			denormalizar(A[2])
			resultados.append(A[2][0])
		elif (in_fin_validacion - in_inicio_validacion == 0) or (in_inicio_validacion <= i < in_fin_validacion):
			esperados.append([float(row[8]), float(row[9])])
			denormalizar(A[2])
			resultados.append(A[2][0])
		i += 1
	return esperados, resultados

def precision(esperados, resultados):
	aceptables = 0.0
	totales = 0.0
	for i in xrange(0,len(esperados)):
		print esperados[i], '   =    ', resultados[i]
		if valorAceptable(esperados[i],resultados[i]):
			aceptables += 1
		totales += 1
	precision = aceptables/totales
	print 'Precision: ', precision
	return precision

def learningRate(b):
	global lr
	if b:
		#print 'crece ', lr
		lr += 0.01
		if lr > 0.1:
			lr = in_lr
	else:
		lr -= lr/2
		if lr < 0.001:
			lr = 0.001
	return lr

def main():
	global in_dataset
	global in_m
	global in_lr
	global in_T
	global in_inicio_validacion
	global in_fin_validacion
	global fname
	global datos
	if len(sys.argv) > 1:
		if sys.argv[1] == '-h' or sys.argv[1] == '--help':
			print 'los parametros disponibles aparecen en el readme junto con su explicacion'
			print 'si no se especifica ningun parametro se usan los que estan por defecto en el codigo'
			print 'el entrenamiento se guarda y carga dependiendo de los parametros tamanio_capa_oculta max_lr max_epocas inicio/fin_validacion'
			return 0
		elif len(sys.argv) == 6 and (len(sys.argv[1]) > 1):
			in_dataset = sys.argv[1]
			in_train_dataset = sys.argv[2]
			in_m = int(sys.argv[3])
			in_lr = float(sys.argv[4])
			in_T = int(sys.argv[5])
		elif len(sys.argv) == 7:
			in_dataset = sys.argv[1]
			in_train_dataset = in_dataset
			in_m = int(sys.argv[2])
			in_lr = float(sys.argv[3])
			in_T = int(sys.argv[4])
			in_inicio_validacion = int(sys.argv[5])
			in_fin_validacion = int(sys.argv[6])
		elif len(sys.argv) == 6:
			in_m = int(sys.argv[1])
			in_lr = float(sys.argv[2])
			in_T = int(sys.argv[3])
			in_inicio_validacion = int(sys.argv[4])
			in_fin_validacion = int(sys.argv[5])
		elif len(sys.argv) == 4:
			in_m = int(sys.argv[1])
			in_lr = float(sys.argv[2])
			in_T = int(sys.argv[3])
		elif len(sys.argv) == 2:
			in_dataset = sys.argv[1]
		elif len(sys.argv) == 3:
			in_dataset = sys.argv[1]
			in_train_dataset = sys.argv[2]
	fname = 'p2_'+str(in_m)+'_'+str(in_lr)+'_'+str(in_T)+'_'+str(in_inicio_validacion)+'_'+str(in_fin_validacion)
	inicializarMatrices()
	print 'Levantar y normalizar'
	levantarYNormalizar()
	print 'Termina de normalizar'
	print 'Pre entrenamiento'
	esperados, resultados = cicloCompleto()
	precision(esperados, resultados)
	entrenamientoCargado = cargarEntrenamiento()
	if not(entrenamientoCargado):
		entrenamiento()
		guardarEntrenamiento()
	print 'Post entrenamiento'
	esperados, resultados = cicloCompleto()
	precision(esperados, resultados)
	graficarResultados(esperados, resultados)

if __name__ == "__main__":
    main()