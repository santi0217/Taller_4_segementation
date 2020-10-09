#
#  Fecha: 8 de octubre del 2020
#
#  Autores: Santiago Márquez Álvarez
#           Sergio Mora Pradilla
#
#  Descripción: Este programa consiste en la segmentación de colores de una imagen de 1 a 10 colores. Lo que se busca es computar
#               la suma de distancias intra-clusters según el numero de colores en cada caso. Para realizarlo se importaron las
#               librerías necesarias para la implementación, luego en el main se obtuvo como entrada de teclado la imagen y el método
#               a realizar, se convirtió esta imagen de BGR a RGB, se normalizo y se redimensiono a 2 dimensiones, y luego se declararon
#               las variables de numero de colores y los arreglos donde se guardarían las sumas intra.clusters.
#               Siguiente a esto, se creo un bucle para realizar los 10 procesos para segmentar de 1 a 10 colores, con las instrucciones
#               que permiten  muestrear los valores de la imagen para realizar la segmentación con el método escogido y posterior a esto
#               calcular, primero la suma de las distancias entre los pixeles y el centro de su respectivo cluster, y segundo la suma de
#               aquellos acumulados, para finalmente comparar con un gráfico cuales fueron las sumas de los acumulados de los clusters en
#               cada uno de los casos del número de colores segmentados.
#

# Importación de librerias
import cv2
import numpy as np
import math

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

if __name__ == '__main__':
    # Se pide como entrada de teclado la ruta de la imagen y el metodo de segmentación
    path = input("Ingrese la ruta de la imagen:")
    method = input("Ingrese el metodo de segmentación:")

    # Se lee la imagen y se convierte a RGB
    image_init = cv2.imread(path)
    image = cv2.cvtColor(image_init, cv2.COLOR_BGR2RGB)

    # Se define los casos de clustering que se haran de 1 a 10
    n_colors = 10
    n_color = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Se hace casting a los datos de la imagen y se normaliza
    image = np.array(image, dtype=np.float64) / 255

    # Se convierte la imagen a 2 dimensiones
    rows, cols, ch = image.shape
    image_array_init = np.reshape(image, (rows * cols, ch))
    image_array = image_array_init.copy()

    # Se declaran los arreglos referentes a las sumas intra-clusters
    dist = np.zeros((10,10))
    dist_sum = np.zeros(10)

    # Se realiza el proceso de segmentación para segmentar de 1 a 10 colores
    for i in range(n_colors):

        num_colors = i + 1 # Se itera el numero de clusters de 1 a 10
        image_array_sample = shuffle(image_array, n_samples=1000, random_state=0) # Se muestrea la imagen aleatoriamente
        if method == 'gmm':
            model = GMM(n_components=num_colors).fit(image_array_sample) # Se modela el metodo de GMM
        elif method == 'kmeans':
            model = KMeans(n_clusters=num_colors, random_state=0).fit(image_array_sample) # Se modela el metodo de Kmenas

        # Get labels for all points
        if method == 'gmm':
            labels = model.predict(image_array) # Se etiqueta cada pixel segun su cluster
            centers = model.means_ # Se enceuntran los centros de cada cluster
        elif method == 'kmeans':
            labels = model.predict(image_array) # Se etiqueta cada pixel segun su cluster
            centers = model.cluster_centers_ # Se enceuntran los centros de cada cluster

        #Se recorre cada uno de los pixeles para calcular su distancia con respecto a su cluster
        for j in range(rows*cols):
            index = labels[j] # Se escoge el numero del cluster al que pertenece el pixel
            dist[i][index] += math.sqrt((image_array[j][0] - centers[index][0])** 2 + (image_array[j][1] - centers[index][1]) ** 2
                                 + (image_array[j][2] - centers[index][2]) ** 2) # Se calcula la distancia de cada pixel con respecto al centro de su cluster
                                                                                 # y se suma cada una de esas distancias

    # Se recorre la matriz de las sumas de distancias de los clusters
    for k in range(n_colors):
        for l in range(n_colors):
            dist_sum[k] += dist[k][l] # Se suma el acumulado de cada cluster segun el numero de colores que haya

    # Se hace un grafico de barras para comparar cada uno de los casos de n_colors con su suma total de distancias intra-cluster
    # Se pone titulo y nombre de los ejes en el grafico
    plt.bar(n_color,dist_sum)
    plt.title('Suma de distancias intra-cluster vs n_color, (method={})'.format(method))
    plt.xlabel('Número de colores')
    plt.ylabel('Suma total de distancias intra-clusters')
    plt.show()