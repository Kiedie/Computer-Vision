import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt  

'''
Funcion que muestra por pantalla una imagen a color o en escala de grises según el flagColor pasado
'''

def leer_imagen(filename,flagColor):
    
    #Leemos la imagen 
    img = cv.imread(filename,flagColor)
    
    #Calculamos el numero de canales
    dim = img.ndim #también vale img.shape[2]
    
    #Visualizamos la imagen
    plt.figure(1)
    plt.imshow(img[:,:,::-1] if dim == 3 else img, cmap='gray')
    cv.imwrite('original.jpg',img)

    #Si la imagen es a color la ponemos en gris
    if dim == 3: 
        img3 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        plt.figure(2)
        plt.imshow(img3,cmap='gray')
        cv.imwrite('Grises.jpg',img3)
    plt.show ()



def visualizar_matrices(mono,tri):
    
    #Normalizamos matriz monobanda y la imprimimos sin los ejes
    normalizado_mb = ( mat_mb-mat_mb.min() ) / (mat_mb-mat_mb.min()).max()
    plt.figure(1)
    plt.axis('off')
    print(normalizado_mb,"\n============================================================\n")
    plt.imshow(normalizado_mb, cmap = 'gray',aspect='equal')
    plt.show()
    
    #Normalizamos matriz tribanda y la imprimimos sin los ejes
    normalizado_tb = ( mat_tb-mat_tb.min() ) / (mat_tb-mat_tb.min()).max()
    plt.figure(2)
    plt.xticks([]), plt.yticks([])
    print(normalizado_tb)
    plt.imshow(normalizado_tb,aspect='equal')

    plt.show()



def pintaMI(vim):
    
    #Redimensionamos las matrices de las imágenes para que tengan los mismos pixeles
    
    #Obtenemos el máximo de filas y de columnas
    width  = max(im.shape[1] for im in vim)
    heigth = max(im.shape[0] for im in vim)
    
    #Redimensionamos las filas y columnas de todas las imagenes
    for i in range(len(vim)):
        vim[i] = cv.resize(vim[i],(heigth,width))
        
    #Convertimos imagenes monocanal a tricanal
    for im_i in range(len(vim)):
        if vim[im_i].ndim != 3:
            vim[im_i] = cv.cvtColor(vim[im_i],cv.COLOR_GRAY2BGR) 
    
    #Concatenamos horizontalmente
    imc = cv.hconcat(vim) #Para concatenar verticalmente usar vconcat
    
    #Mostramos imágenes
        # plt.figure      --> aumentamos el tamaño de la visualizacion
        # plt.axis('off') --> eliminamos los ejes
        # cv.imwrite      --> para guardar la imagen en un archivo
    plt.figure(figsize=(20,20))
    plt.axis('off')
    plt.imshow(imc[:,:,::-1] if imc.ndim == 3 else imc, cmap='gray')
    cv.imwrite('ejercicio2.jpg',imc)
    plt.show()



def cuadrozul(rr):
    
    #Obtenemos las filas y columnas que tiene la foto 
    raw = rr.shape[0]
    col = rr.shape[1]

    #Calculamos el punto medio en entero
    y = int(raw/2)
    x = int(col/2)

    
    lado = 50
    #Modificamos los pixeles de color azul en formato BGR en la matriz 
    #Lo hacemos en el centro de la imagen y en los tres canales 
    rr[y-lado:y+lado,x-lado:x+lado,:] = [255,0,0]
    
    #Mostramos la imagen y la guardamos
    plt.figure(1)
    plt.imshow(rr[:,:,::-1] if rr.ndim == 3 else rr, cmap='gray')
    cv.imwrite('ejercicio4.jpg',rr)
    plt.show()




'''
Función que muestra en una única ventana varias imágenes con sus respectivos títulos
    - vim:  Vector de imágenes
    - vti:  Vector de títulos
    - raws: Número de filas que queremos que tenga la representación
'''
def joinImgTitles(vim,vti,raws):  
    #Calculamos el número de columnas
    col = int(len(vim)/raws)  
    #Para cada imagen hacemos un subplot y mostramos la figura con su título
    for i in range(len(vim)):
        plt.subplot(raws,col,1+i)
        plt.imshow(vim[i][:,:,::-1] if vim[i].ndim == 3 else vim[i], cmap='gray')
        plt.title(vti[i])
        plt.axis('off')
    
    plt.show()




filename = 'orapple.jpg'
flagColor = 1
leer_imagen(filename,flagColor)




#Creamos una matriz 3x3 de numero aleatorios en el rango [0,100] (MONOBANDA y TRIBANDA)
mat_mb = np.random.rand(3,3)*255
mat_tb = np.random.rand(3,3,3)*255
visualizar_matrices(mat_mb,mat_tb)



ej3_im1 = cv.imread('dave.jpg')
ej3_im2 = cv.imread('messi.jpg')
ej3_im3 = cv.imread('orapple.jpg')
ej3_im4 = cv.imread('pepe.jpg')
#ej3_im4 = cv.cvtColor(ej3_im4,cv.COLOR_BGR2GRAY)
#(thresh, ej3_im4) = cv.threshold(ej3_im4, 127, 255, cv.THRESH_BINARY)
vim = [ej3_im1,ej3_im2,ej3_im3,ej3_im4]
pintaMI(vim)



rr = cv.imread('messi.jpg')
cuadrozul(rr)




ej5_im1 = cv.imread('dave.jpg')
ej5_im2 = cv.imread('messi.jpg')
ej5_im3 = cv.imread('orapple.jpg')
ej5_im4 = cv.imread('pepe.jpg')

vim = [ej5_im1,ej5_im2,ej5_im3,ej5_im4]
vti = ['dave','messi','orapple','opencv']
joinImgTitles(vim,vti,2)