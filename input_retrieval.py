import argparse
import os

# PURPOSE: Analizar la entrada de la línea de comando y extraer los valores ingresados por el usuario
# PARAMETERS: N/A
# RETURN:
# - Etiquetas del conjunto de datos COCO
# - Ruta al modelo de datos entrenado con sus pesos
# - Ruta al archivo de configuración
# - Ruta al video de entrada
# - Ruta al video de salida
# - Valor de confianza
# - Valor de umbral
def parseCommandLineArguments():
	# construir el analizador de argumentos y analizar los argumentos
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True,
		help="path to input video")
	ap.add_argument("-o", "--output", required=True,
		help="path to output video")
	ap.add_argument("-y", "--yolo", required=True,
		help="base path to YOLO directory")
	ap.add_argument("-c", "--confidence", type=float, default=0.3,
		help="minimum probability to filter weak detections")
	ap.add_argument("-t", "--threshold", type=float, default=0.3,
		help="threshold when applying non-maxima suppression")

	args = vars(ap.parse_args())

	# cargar las etiquetas de clase COCO en las que se entrenó nuestro modelo YOLO
	labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")
	
	# derivar las rutas a los pesos YOLO y la configuración del modelo
	weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
	configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
	
	inputVideoPath = args["input"]
	outputVideoPath = args["output"]
	confidence = args["confidence"]
	threshold = args["threshold"]

	return LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath, confidence, threshold
