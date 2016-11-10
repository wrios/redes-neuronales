python problemaX.py opciones:
	dataset_test
	dataset_train
	tam_red_oculta
	learning_rate
	cant_epocas

	dataset_test_train
	tam_red_oculta
	learning_rate
	cant_epocas
	inicio_cross_validacion
	fin_cross_validacion

	tam_red_oculta
	learning_rate
	cant_epocas
	inicio_cross_validacion
	fin_cross_validacion

	tam_red_oculta
	learning_rate
	cant_epocas

	dataset_test
	dataset_train

	dataset_test_train

donde:
	dataset_test = dataset usado como base para entrenar
	dataset_train = dataset usado como base para testear
	dataset_test_train = dataset usado como base para entrenar y testear, mejor usado en cross_validation
	tam_red_oculta = tamanio en neuronas de la red oculta
	learning_rate = coeficiente de aprendizaje
	cant_epocas = cantidad de epocas a entrenar
	inicio_cross_validacion = inicio del intervalo usado como validacion en cross validation
	fin_cross_validacion = fin del intervalo usado como validacion en cross validation