from __future__ import division
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random
import platform
import itertools
from itertools import groupby as g
import csv
import sys
import getopt
import numpy as np
import ast
import scipy.io
from numpy import linalg as LA
import os.path
import matplotlib.pylab as plt

in_T = 600
m1 = 10
m2 = 10
W = []
X = np.ones((856,1))
Y = np.zeros((856,1))
M = []
datos = []
Map = np.zeros((m1, m2))

in_inicio_validacion = 0
in_fin_validacion = 0
in_cargarEntrenamiento = False
in_entrenamiento = True
in_guardarEntrenamiento = True
in_dataset = 'tp2_training_dataset.csv'
in_train_dataset = in_dataset
fname = 'kohonen_'+'_'+str(m1)+'_'+str(m2)+'_'+str(in_T)+'_'+str(in_inicio_validacion)+'_'+str(in_fin_validacion)

def inicializarMatrices():
	armarMatriz()
	armarMatrizActivaciones()
	armarMapaActivaciones()

#def learningRate(n):
	#lr = 0.05/np.power(n+16,1.8)
	#return lr

def learningRate(n):
	lr = 0.001/(np.power(n+1,0.5))
	return lr

def levantarYNormalizar():
	global datos
	i = 0
	reader = csv.reader(open(in_train_dataset,'rb'))
	for vector in reader:
		temp = np.asarray(vector[1:]).reshape((856,1))
		if (in_dataset != in_train_dataset):
			datos.append(temp)
		elif not(in_inicio_validacion <= i < in_fin_validacion):
			datos.append(temp)
		i += 1

def armarMatrizActivaciones():
	global M
	for i in range(0,m1):
		fila = []
		for j in range(0,m2):
			fila.append([])
		M.append(fila)

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  

def armarMapaActivaciones():
	global M
	global Map
	Map = np.zeros((m1, m2))
	for i in range(0,m1):
		for j in range(0,m2):
			Map[i][j] = most_common(M[i][j])


def sigma(n):
	dr = (m1-1)/(in_T*m1)
	sigma = m1/(1+(n*m1*dr))	
	return sigma 

def armarMatriz():
	global W
	W = []
	for i in range(0,m1):
		fila = []
		for j in range(0,m2):
			fila.append(np.random.uniform( -0.1, 0.1, (856,1)))
		W.append(fila)

def activar():
	global Y
	norm = np.linalg.norm(X - W[0][0])
	Y[:] = (X - W[0][0] )
	i = 0
	j = 0
	for i_t in range(0,m1):
		for j_t in range(0,m2):
			Y_t = (X - W[i_t][j_t] )
			norm_t = np.linalg.norm(Y_t)
			if norm_t < norm:
				norm = norm_t
				Y[:] = Y_t
				i = i_t
				j = j_t
	return i, j, Y_t

def gaussiana(n, i, j, i_t ,j_t,sigma):
	return np.exp((-((np.power(i-i_t,2)+(np.power(j-j_t,2)))/(2*(np.power(sigma,2))))))

def vecindad(n, i, j, Y_t):
	r = [] 
	if sigma(n)> 1.2:
		for i_t in range(i-int(sigma(n)*0.5)-1,2+i+int(sigma(n)*0.5)):
			for j_t in range(j-int(sigma(n)*0.5)-1,2+j+int(sigma(n)*0.5)):	
				t1 = i_t
				t2 = j_t
				t1 = (t1+m1)%(m1)
				t2 = (t2+m2)%(m2)
				if not(t1 == i and t2 == j):
					x = [t1, t2, (X - W[t1][t2])*gaussiana(n, i, j, t1, t2, sigma(n))]	
					r.append(x)
				elif i_t == i and j_t == j:
					x = [i_t, j_t, Y_t] 
					r.append(x)			
	else:
		x = [i, j, Y_t] 
		r.append(x)	
	return r

def correccion(vecindades, n):
	global W
	e = 0 
	for vs in vecindades:
		for v in vs:
			i, j, delta = v	
			W[i][j] = (learningRate(n) * delta) + W[i][j] 
			e += np.linalg.norm(learningRate(n) * delta)
	print e, n, sigma(n), learningRate(n)		

def epoca(n):
	global X
	global W
	global Y
	global datos
	vecindades = []
	o = 0
	reader = csv.reader(open(in_dataset,'rb'))
	for vector in reader:
		X[:] = np.asarray(vector[1:]).reshape((856,1))
		i, j, Y_t = activar()
		v = vecindad(n, i, j, Y_t)
		vecindades.append(v)
	print 'Vecindad ', len(vecindades[0])	
	correccion(vecindades, n)
	

def entrenamiento():
	n = 1
	while n < in_T:
		progress(n, in_T)
		epoca(n)
		n += 1

def cicloCompleto():
	global X
	global W
	global Y
	global M
	l = 1
	reader = csv.reader(open(in_dataset,'rb'))
	for vector in reader:
		X[:] = np.asarray(vector[1:]).reshape((856,1))
		i, j, Y_t = activar()
		M[i][j].append(vector[0])
		progress(l, 900, 'i: '+str(i)+'j: '+str(j)+'vector[0]: '+vector[0])
		l += 1

def graficar():
	armarMapaActivaciones()
	fig = plt.figure()
	fig.suptitle('kohonen'+fname)
	print Map
	plt.matshow(Map)
	plt.savefig('training/'+fname+'.png', format = 'png')
	plt.show()

def most_common(L):
	if len(L) > 0:
		groups = itertools.groupby(sorted(L))
		def _auxfun((item, iterable)):
			return len(list(iterable)), -L.index(item)
		return max(groups, key=_auxfun)[0]
	else:
		return -1

def guardarEntrenamiento():
	if not(checkFile('training/'+fname)):
		print 'Guardando Entrenamiento'
		outfile = open('training/'+fname, 'w')
		np.save(outfile, W)
		outfile.close()
		print 'Entrenamiento Guardado'

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

def main():
	global in_dataset
	global in_metodo
	global in_T
	global in_inicio_validacion
	global in_fin_validacion
	global fname
	global datos
	global m1
	global m2
	if len(sys.argv) > 1:
		if sys.argv[1] == '-h' or sys.argv[1] == '--help':
			print 'los parametros disponibles aparecen en el readme junto con su explicacion'
			print 'si no se especifica ningun parametro se usan los que estan por defecto en el codigo'
			print 'el entrenamiento se guarda y carga dependiendo de los parametros max_epocas inicio/fin_validacion'
			return 0
		elif len(sys.argv) == 2:
			in_dataset = sys.argv[1]
			in_train_dataset = in_dataset
		elif len(sys.argv) == 3:
			in_dataset = sys.argv[1]
			in_train_dataset = sys.argv[2]
		elif len(sys.argv) == 4:
			m1 = int(sys.argv[1])
			m2 = int(sys.argv[2])
			in_T = int(sys.argv[3])
		elif len(sys.argv) == 6:
			m1 = int(sys.argv[1])
			m2 = int(sys.argv[2])
			in_T = int(sys.argv[3])
			in_inicio_validacion = int(sys.argv[4])
			in_fin_validacion = int(sys.argv[5])
		#elif len(sys.argv) == 6:
			#m1 = int(sys.argv[1])
			#m2 = int(sys.argv[2])
			#in_T = int(sys.argv[3])
			#in_dataset = sys.argv[4]
			#in_train_dataset = sys.argv[5]
		elif len(sys.argv) == 7:
			m1 = int(sys.argv[1])
			m2 = int(sys.argv[2])
			in_T = int(sys.argv[3])
			in_dataset = sys.argv[4]
			in_train_dataset = in_dataset
			in_inicio_validacion = int(sys.argv[5])
			in_fin_validacion = int(sys.argv[6])

	fname = 'kohonen_'+'_'+str(m1)+'_'+str(m2)+'_'+str(in_T)+'_'+str(in_inicio_validacion)+'_'+str(in_fin_validacion)
	inicializarMatrices()
	print 'Levantar y normalizar'
	levantarYNormalizar()
	print 'Termina de normalizar'
	print 'Pre entrenamiento'
	resultados = []
	resultados = cicloCompleto()
	#graficar()
	entrenamientoCargado = cargarEntrenamiento()
	if not(entrenamientoCargado):
		entrenamiento()
		guardarEntrenamiento()
	print 'Post entrenamiento'
	resultados = []
	resultados = cicloCompleto()
	graficar()

if __name__ == '__main__':
	main()