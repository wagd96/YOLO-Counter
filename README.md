# CONTEO DE VEHÍCULOS CON YOLO

## Resumen
You Only Look Once (YOLO) es una arquitectura de CNN para realizar la detección de objetos en tiempo real. El algoritmo aplica una sola red neuronal a la imagen completa y luego divide la imagen en regiones y predice cuadros delimitadores y probabilidades para cada región. Para ver información [ver](URL_DEL_PDF).

Este proyecto tiene como objetivo contar cada vehículo (motocicleta, autobús, automóvil, ciclo, camión, tren) detectado en el video de entrada utilizando el algoritmo de detección de objetos YOLOv3.


## Algoritmo en acción 
<p align="center">
  <img src="https://github.com/guptavasu1213/Yolo-Vehicle-Counter/blob/master/example_gif/highwayVideoExample.gif">
</p>
Como se puede apreciar, los vehículos son detectados y clasificados, se encierran en un recuadro y se cuentan al pasar por la región establecida.


## Prerequisitos
* Sistema operativo Linux o MacOS (probado en Ubuntu 18.04).
* Un video de entrada, archivo mp4 de la calle para ejecutar el conteo de vehículos.
* El modelo yolov3 previamente entrenado. Debe descargarse siguiendo estos pasos:

```
cd yolo-coco
wget https://pjreddie.com/media/files/yolov3.weights
``` 

## Dependencias necesarias
* Python3 (Probado en 3.8.2)
```
sudo apt-get upgrade python3
```
* Pip3
```
sudo apt-get install python3-pip
```
* OpenCV 3.4 or above(Tested on OpenCV 3.4.2.17)
```
pip3 install opencv-python==3.4.2.17
```
* Imutils 
```
pip3 install imutils
```
* Scipy
```
pip3 install scipy
```

## Usos
* `--input` o `-i` El argumento requiere la ruta del video de entrada.
* `--output` o `-o` El argumento requiere la ruta al video de salida.
* `--yolo` o `-y` El argumento requiere la ruta a la carpeta donde se almacena el archivo de configuración, los pesos y el archivo coco.names
* `--confidence` o `-c` Es un argumento opcional que requiere un número flotante entre 0 y 1 que denota la confianza mínima de las detecciones. De forma predeterminada, la confianza es 0,5 (50%).
* `--threshold` o `-t` Es un argumento opcional que requiere un número flotante entre 0 y 1 que denota el umbral cuando se aplica una supresión no máxima. De forma predeterminada, el umbral es 0,3 (30%).
```
python3 yolo_video.py --input <PathVideoEntrada> --output <PathVideoSalida> --yolo yolo-coco [--confidence <Número flotante entre 0 y 1>] [--threshold <Número flotante entre 0 y 1>] 
```
Ejemplos: 
* Ejecución valores por defecto
```
python3 yolo_video.py --input input/videoPrueba.mp4 --output output/salidaPrueba.avi --yolo yolo-coco 
```
* Ejecución especificando confianza:
```
python3 yolo_video.py --input input/videoPrueba.mp4  --output output/salidaPrueba.avi --yolo yolo-coco --confidence 0.3
```

## Detalles de implementación
* Las detecciones se realizan en cada frame (fotograma) utilizando el algoritmo de detección de objetos YOLOv3 y se muestran en la pantalla con cuadros delimitadores, clasificando el vehículo.
* Las detecciones se filtran para mantener todos los vehículos como motocicleta, autobús, automóvil, bicicleta, camión, tren. La razón por la que también se cuentan los trenes es porque a veces, los vehículos más largos, como un autobús, se detectan como un tren; por tanto, los trenes también se tienen en cuenta.
* El centro de cada casilla se toma como punto de referencia (indicado por un punto verde cuando se realizan las detecciones) al rastrear los vehículos.   


## Referencias
* [Presentación YOLO curso Procesamiento Digital de Imágenes (PDI)](URL_DEL_PDF)
* [YOLO object detection with OpenCV](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
