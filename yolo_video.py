# Importación de las librerías necesarias
import numpy as np
import imutils
import time
from scipy import spatial
import cv2
from input_retrieval import *

# Establecer el tamaño para procesar el vídeo
inputWidth, inputHeight = 416, 416

# extraer los valores necesarios de la línea de comandos
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,\
	preDefinedConfidence, preDefinedThreshold = parseCommandLineArguments()

# Inicializar una lista de colores para representar cada etiqueta de clase posible
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# PURPOSE: muestra el contador de vehículos en la esquina superior izquierda del marco.
# PARAMETERS: frame en el que se muestra el contador, el número de recuento de vehículos
# RETURN: ninguno
def displayVehicleCount(frame, vehicle_count):
	cv2.putText(
		frame, 
		'Vehiculos detectados: ' + str(vehicle_count), 
		(20, 100), 
		cv2.FONT_HERSHEY_SIMPLEX, 
		4, 
		(0, 0xFF, 0), 
		3, 
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)

# PURPOSE: Dibuja todos los cuadros de detección con un punto verde en el centro.
# RETURN: N/A
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
	# pregunta que exista al menos una detección
	if len(idxs) > 0:
		# recorrer los índices de los elementos detectados
		for i in idxs.flatten():
			# extraer las coordenadas del bounding box
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# dibuja un rectangulo y una etiqueta en el frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			#Dibuja un punto verde en el medio del rectangulo.
			cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)

# PURPOSE: Inicializando la grabadora de video con el path de salida de video y el mismo número
# de fps, ancho y alto como el video de origen
# PARAMETERS: Ancho del video de origen, Alto del video de origen, stream de video
# RETURN: El escritor de video inicializado
def initializeVideoWriter(video_width, video_height, videoStream):
	# Obteniendo los fps del video fuente
	sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
	# inicializar nuestro escritor de video
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
		(video_width, video_height), True)

# cargue su detección de objetos YOLO entrenada en el conjunto de datos COCO (80 clases)
# y determinar solo los nombres de capa * de salida * que necesitamos de YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# inicializar la secuencia de video, el puntero para generar el archivo de video y
# dimensiones del frame
videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Especificar coordenadas para una línea predeterminada
x1_line = 0
y1_line = video_height//2
x2_line = video_width
y2_line = video_height//2
line_coordinates = x1_line, y1_line, x2_line, y2_line

#Inicialización
vehicle_count = 0
writer = initializeVideoWriter(video_width, video_height, videoStream)
# recorrer los fotogramas de la secuencia del archivo de vídeo
while True:
	# Inicialización para cada iteración
	boxes, confidences, classIDs = [], [], [] 

	# leer el siguiente fotograma del archivo
	(grabbed, frame) = videoStream.read()

	# si no se tomó el fotograma, entonces habremos llegado al final de la secuencia
	if not grabbed:
		break

	# construir un blob a partir del frame de entrada y luego realizar un avance
	# pase del detector de objetos YOLO, dándonos nuestros cuadros delimitadores
	# y probabilidades asociadas
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# bucle sobre cada una de las salidas de capa
	for output in layerOutputs:
		# recorrer cada una de las detecciones
		for i, detection in enumerate(output):
			# extraer la identificación de clase y la confianza (es decir, probabilidad)
			# de la detección de objetos actual
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filtrar predicciones débiles asegurándose de que el detectado
			# probabilidad es mayor que la probabilidad mínima
			if confidence > preDefinedConfidence:
				# escalar las coordenadas del cuadro delimitador en relación con
				# el tamaño de la imagen, teniendo en cuenta que YOLO
				# en realidad devuelve las coordenadas centrales (x, y) de
				# el cuadro delimitador seguido del ancho de los cuadros y
				# altura
				box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
				(centerX, centerY, width, height) = box.astype("int")

				# use las coordenadas centrales (x, y) para derivar la parte superior
				# y esquina izquierda del cuadro delimitador
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# actualizar nuestra lista de coordenadas del cuadro delimitador,
				# confidencias e identificaciones de clase
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	#aplicar supresión no máxima para suprimir la superposición débil
	# cuadros delimitadores
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
		preDefinedThreshold)

	# Dibujar una línea de conteo delimitador
	cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0xFF, 0), 2)

	# Dibujar cuadro de detección 
	drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

########## Se realiza el conteo de vehículos
	# Se verifica que exista al menos una detección
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			# extraer las coordenadas del cuadro delimitador
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# Se calcula el centro del cuadro delimitador	
			centerX = x + (w//2)
			centerY = y + (h//2)

			# Cuando la detección está en la lista de vehículos
			if (LABELS[classIDs[i]] in ["bicycle","car","motorbike","bus","truck","train","person"]):
				print(LABELS[classIDs[i]],":",centerX, centerY)

				# Se verifica si el objeto cruza la línea y se cuenta  
				if((centerX+5 >= x1_line and centerX-5 <= x2_line+5) and (centerY+5 >= y1_line and centerY-5 <= y2_line+5)):
					vehicle_count += 1
					print("VEHICULO[",vehicle_count,"]:",centerX, centerY)
################

	# Mostrar el recuento de vehículos si un vehículo ha pasado la línea 
	displayVehicleCount(frame, vehicle_count)

    # escribe el frame de salida en el disco
	writer.write(frame)

	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break	
	
print("[INFO] cleaning up...")
writer.release()
videoStream.release()