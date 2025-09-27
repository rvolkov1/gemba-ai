import cv2
import time
from ultralytics import YOLO

rtsp_url = "rtsp://gembavision_B057:Jackrossjj%21@10.133.248.245:554/stream1"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
	print("cap err")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("stream opened")

model = YOLO("yoloe-v8s-seg.pt")
print(model.export(format="onxx", imgsz=128))

names = ["person", "glasses", "gloves", "headphones", "earmuffs"]
model.set_classes(names, model.get_text_pe(names))

while True:
	ret, frame = cap.read()
	if not ret:
		break

	#cv2.imshow("win", frame)

	results = model.predict(frame)

	#print(results)
	#print("---------------------------------------------------------")

	# Output the visual detection data
	annotated_frame = results[0].plot(boxes=True, masks=False)

	# Get inference time
	inference_time = results[0].speed['inference']
	fps = 1000 / inference_time  # Convert to milliseconds
	text = f'FPS: {fps:.1f}'

	# Define font and position
	font = cv2.FONT_HERSHEY_SIMPLEX
	text_size = cv2.getTextSize(text, font, 1, 2)[0]
	text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
	text_y = text_size[1] + 10  # 10 pixels from the top

	# Draw the text on the annotated frame
	cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

	# Display the resulting frame
	cv2.imshow("Camera", annotated_frame)

	# Exit the program if q is pressed

	if cv2.waitKey(1) & 0xF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
print("done")
    
