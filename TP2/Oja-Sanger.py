from __future__ import division
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random
import platform
import itertools
import csv
import sys
import getopt
import numpy as np
import ast
import scipy.io
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score

l = 3
Y = np.zeros((3,1))
in_inicio_validacion = 0
in_fin_validacion = 0
in_metodo = 'Sanger'
in_cargarEntrenamiento = False
in_entrenamiento = True
in_guardarEntrenamiento = True
TotalEpoca = 500
W = np.random.uniform( -0.1, 0.1, (856,l))
dW = np.zeros((856,l))
Xm = np.zeros((856,1))
X = np.zeros((856,1))
in_dataset = 'tp2_training_dataset.csv'

def epoca(n):
	global W
	global dW
	global Y
	e_actual = 0
	reader = csv.reader(open(in_dataset,'rb'))
	for vector in reader:
		if not(int(in_inicio_validacion) <= e_actual <= int(in_fin_validacion)):
			X[:] = np.asarray(vector[1:]).reshape((856,1))
			Y[:] = np.dot(X.T,W).T
			if in_metodo == 'Oja':
				oja(n)
			else:
				sanger(n)
			W += dW 
		e_actual += 1
	s =  np.linalg.norm(np.dot(W.T, W) - np.identity(3))
	return s

def activacion():
	res = np.dot(X.T,W).T
	return res

def learningRate(n):
	lr = 1/(np.exp(n))
	return lr

def cicloCompleto():
	global X
	resultados = []
	reader = csv.reader(open(in_dataset,'rb'))
	for vector in reader:
		X[:] = np.asarray(vector[1:]).reshape((856,1))
		activacion_x = activacion()
		resultado = [activacion_x[0], activacion_x[1], activacion_x[2], vector[0]]
		resultados.append(resultado)
	return resultados

def color(cat):
	colours = ["b","g","r","c","m","y","k","w", "#D8BFD8"]
	return colours[int(cat)-1]

def sanger(n):
	global dW
	dW[:] = np.zeros((856,l))
	for i in xrange(0,l):
		Y_i = np.zeros((l,1))
		Y_i[:i+1] = Y[:i+1]
		#print 'Y_i :' + str(i)
		#print Y_i
		Xm[:] = np.dot(W, Y_i)
		dW[:] += 0.01/n * np.dot((X-Xm), Y_i.T)

def oja(n):
	global dW
	Xm[:] = np.dot(W, Y)
	dW[:] = 0.01/n * np.dot((X-Xm), Y.T)


def entrenamiento():
#poner cada categoria con su salida 
	s = 1
	umbral = 0.00001
	n = 1
	while s > umbral and n < TotalEpoca:
		s = epoca(n)
		n += 1
		progress(n, TotalEpoca, "s: " + s)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def graficar(resultados):
	fig = pylab.figure()
	ax = Axes3D(fig)
	e_actual = 0
	for resultado in resultados:
		if (not(int(in_inicio_validacion) <= e_actual <= int(in_fin_validacion))):
			ax.scatter(resultado[0], resultado[1], resultado[2], color=color(resultado[3]), marker='o')
		else:
			ax.scatter(resultado[0], resultado[1], resultado[2], color=color(resultado[3]), marker='x')
		e_actual += 1
	plt.show()

def crearMatriz(resultados):
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
	y_true = []
	y_pred = []
	f = open(in_dataset,'rb')
	lines = f.readlines()
	for i in xrange(int(in_inicio_validacion),int(in_fin_validacion)):
		vector = lines[i]
		resultado = resultados[i]
		y_true.append(vector[0])
		y_pred.append(resultado[3])
	accuracy = accuracy_score(y_true, y_pred, True)
	print y_true
	print y_pred
	confMatrix = confusion_matrix(y_true,y_pred,labels)
	return accuracy, confMatrix

def graficarMatriz(confMatrix):
	labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	archivoSalida = 'Matriz_confusion_'+in_metodo+'_'+in_inicio_validacion+'_'+in_fin_validacion
	plt.figure()
	plot_confusion_matrix(confMatrix, labels, False, archivoSalida)
	plt.show()
	#plt.savefig(archivoSalida +'.jpg')
	plt.close()

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben

def guardarEntrenamiento():
	outfile = open('entrenamiento_'+in_metodo+'_'+str(TotalEpoca)+'_'+str(in_inicio_validacion)+'_'+str(in_fin_validacion), 'w')
	np.save(outfile, W)
	outfile.close()

def cargarEntrenamiento():
	global W
	infile = open('entrenamiento_'+in_metodo+'_'+str(TotalEpoca)+'_'+str(in_inicio_validacion)+'_'+str(in_fin_validacion), 'r')
	W[:] = np.load(infile)
	infile.close()

def main():
	global in_metodo
	global in_dataset
	global in_cargarEntrenamiento
	global in_entrenamiento
	global in_guardarEntrenamiento
	global in_inicio_validacion
	global in_fin_validacion
	if len(sys.argv) > 1:
		if sys.argv[1] == '-h' or sys.argv[1] == '--help':
			print 'usage: python Oja-Sanger.py metodo dataset cargarEntrenamiento? entrenar? guardarEntrenamiento? inicio_validacion fin_validacion'
			print 'los parametros bool se definen con 1 o 0, los de validacion con el inicio y final del segmento en int'
			print 'si no se especifica ningun parametro se usan los que estan por defecto en el codigo'
			print 'el entrenamiento se guarda y carga dependiendo de los parametros TotalEpoca en el codigo y metodo e inicio / validacion de parametros'
			return 0
		else:
			in_metodo = sys.argv[1]
			in_dataset = sys.argv[2]
			in_cargarEntrenamiento = sys.argv[3] == '1'
			in_entrenamiento = sys.argv[4] == '1'
			in_guardarEntrenamiento = sys.argv[5] == '1'
			in_inicio_validacion = sys.argv[6]
			in_fin_validacion = sys.argv[7]
	print "comienzo entrenamiento"
	if in_cargarEntrenamiento:
		cargarEntrenamiento()
	if in_entrenamiento:
		entrenamiento()
	if in_guardarEntrenamiento:
		guardarEntrenamiento()
	print "finaliza entrenamiento"
	print "comienzo ciclo completo"
	resultados = []
	resultados = cicloCompleto()
	print "finaliza ciclo completo"
	print "comienzo graficar scatter"
	graficar(resultados)
	print "finaliza graficar scatter"
	print "comienzo graficar matriz confusion"
	#accuracy, confMatrix = crearMatriz(resultados)
	#graficarMatriz(confMatrix)
	print "finaliza graficar matriz confusion"

if __name__ == '__main__':
	main()