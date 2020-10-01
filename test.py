import cv2
import numpy as np
import jetson.inference
import jetson.utils
import time

# Set screen width and height
width = 1280
height = 720

# Load detection models
people_detection = jetson.inference.detectNet('pednet')

# people_indoor_segmentation = jetson.inference.segNet('fcn-resnet18-sun-512x400')
people_outdoor_segmentation = jetson.inference.segNet('fcn-resnet18-mhp-512x320')

# people_indoor_segmentation.SetOverlayAlpha(120.0)
people_outdoor_segmentation.SetOverlayAlpha(50.0)

# Set up camera and display window
cam = jetson.utils.gstCamera(width, height, '0')
cv2.namedWindow('Display')
cv2.moveWindow('Display',0,0)
cv2.setWindowProperty("Display",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# Set FPS calculation
timeMark = time.time()
fpsFilterd = 0

# Load pictures
main_image = cv2.imread('./images/main_image.jpg')
result_image = cv2.imread('./images/result_image.jpg')
main_image = cv2.resize(main_image,(width,height))
result_image = cv2.resize(result_image,(width,height))

current_time = time.time()
result = main_image
operate = False
cv2.imshow('Display', result)
while True:
    # Capture image frame from the camera
	frame, width, height = cam.CaptureRGBA(zeroCopy=1)
	frame_processed, width, height = cam.CaptureRGBA(zeroCopy=1)
	frame_resize = frame_processed
	# Detect person in the frame
	detection = people_detection.Detect(frame_processed, width, height)
	if(len(detection) != 0 or True):
		operate = True
		current_time = time.time()
	elif (time.time() < current_time + 5):
		operate = True
	else:
		operate = False
		result = main_image
  
	# Segment person and add it to the result image
	if(operate):
		people_outdoor_segmentation.Process(frame_processed, width, height)
		jetson.utils.cudaDeviceSynchronize()
		people_outdoor_segmentation.Mask(frame_processed, width, height)
		jetson.utils.cudaDeviceSynchronize()
		output = jetson.utils.cudaToNumpy(frame,width,height,4)
		output = cv2.cvtColor(output, cv2.COLOR_RGBA2BGR).astype(np.uint8)
		mask = jetson.utils.cudaToNumpy(frame_processed,width,height,4)
		mask = cv2.cvtColor(mask,cv2.COLOR_RGBA2GRAY).astype(np.uint8)
		_,mask = cv2.threshold(mask,10,255,cv2.THRESH_BINARY)
		mask_inv = cv2.bitwise_not(mask)
		output_1 = cv2.bitwise_and(output,output,mask=mask)
		output_2 = cv2.bitwise_and(result_image,result_image,mask = mask_inv)
		result = cv2.add(output_1,output_2)
   
	dt= time.time()-timeMark
	fps = 1/dt
	fpsFilterd= .95 * fpsFilterd + .05 * fps
	fpsFilterd =round(fpsFilterd,1)
	cv2.imshow('Display', result)
	timeMark=time.time()
 
	if cv2.waitKey(30) == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()
