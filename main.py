from tkinter.messagebox import NO
from instance_seg.detector import *
from GazepointAPI import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from nlp.asr import ASR
from nlp.recorder import Recorder
from GazepointAPI import Gaze
from threading import Thread
import cv2
import socket
import numpy as np
import os
import json
import time


# Global variables for getting results from recording thread
transcription = []
word_offsets = {}
audio_recording = False
speech_start_time = 0.0

# Global variables for getting last mouse click coordinates on a cv2 image
mouse_x = 0
mouse_y = 0


def record_and_transcribe(rec, recording_filename="output.wav"):
    # Record until silence is detected for 2 seconds
    global transcription
    global word_offsets
    rec_frames, speech_start_time = rec.record_until_silence()
    rec.stop_recording()
    rec.save_recording(rec_frames, recording_filename)
    # Transcribe the recording
    asr = ASR()
    # Outputs in the form [word for word in recording], {word: {index: {start_time: float, end_time: float}} for word in recording}
    transcription, word_offsets = asr.asr_transcript(recording_filename)
    return transcription, word_offsets, speech_start_time


def match_segments_to_speech(class_names, word_offsets):
    class_name_speech_info = []
    instance_speech_weights = np.zeros(len(class_names))
    for count, name in enumerate(class_names):
        if name in word_offsets:
            class_info = word_offsets[name]
            instance_speech_weights[count] += 1
            class_name_speech_info.append((name, class_info))
    return class_name_speech_info, instance_speech_weights


def get_click_coords(event, x, y, flags, param):
    global mouse_x, mouse_y
    # Record mouse coordinates when clicking on a cv2 window (only works for last pair of images)
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x = x
        mouse_y = y


def in_instance_mask(point_x, point_y, bounding_boxes, masks, eye_gaze=False):
    point = Point(point_x, point_y)
    for i in range(bounding_boxes.shape[0]):
        for segment in masks[i].polygons:
            segment = segment.reshape(-1, 2)
            polygon = Polygon(segment)
            if polygon.contains(point) and eye_gaze:
                global instance_eye_weights, instance_fixation_counts, gaze, output_image
                instance_eye_weights[i] += gaze.FPOGD[-1]
                instance_fixation_counts[i] += 1
                output_image = cv2.circle(output_image.astype(np.uint8), (point_x, point_y), radius=2, color=(0, 0, 255), thickness=-1)
            elif polygon.contains(point):
                return i
    return -1


def process_eye_gaze(frame, image_width, image_height, bounding_boxes, masks):
    global gaze
    cv2.imshow("Image_Input", frame)
    cv2.waitKey(1)
    if gaze.eye_gaze_capture(): # if it's a fixation
        if gaze.FPOGX and gaze.FPOGY: # if list is not empty
            eye_gaze_x = int(gaze.FPOGX[-1] * image_width)
            eye_gaze_y = int(gaze.FPOGY[-1] * image_height)
            # Check if within instances' masks
            in_instance_mask(eye_gaze_x, eye_gaze_y, bounding_boxes, masks, eye_gaze=True)


# Change VideoCaptureindex to 2 for laptop, 0 for desktop
cam = cv2.VideoCapture(2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# move the image to the second screen and full screen it
cv2.namedWindow("Image_Input", cv2.WINDOW_NORMAL)
cv2.moveWindow("Image_Input", 900, 0)
cv2.setWindowProperty("Image_Input", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# Capture the image now (since it's a static scene)
ret, frame = cam.read()
img_name = "captured.png"
cv2.imwrite(img_name, frame)
print("{} written!".format(img_name))
cam.release()

# Detect item in image
detector = Detector(model_type="IS")
output_image, output_boxes, class_names, output_masks = detector.onImage("captured.png")


# Start recording on a thread
rec = Recorder(silence_threshold=50, silence_timeout=2)
rec_thread = Thread(target=record_and_transcribe, args=(rec,), daemon=True)
rec_thread.start()

# Capture the eye gaze
gaze = Gaze()
eye_gaze_start_time = time.time()
while (gaze.time == 1) or (gaze.time == 0):
    gaze.eye_gaze_capture()
init_time = gaze.time

# Initialize the instance weights
instance_eye_weights = np.zeros(output_boxes.shape[0])
instance_fixation_counts = np.zeros(output_boxes.shape[0])
# print(output_boxes.shape)

# Check if eye gaze is in any instance
while (rec.talking == False): # wait until audio command
    process_eye_gaze(frame, detector.image_width, detector.image_height, output_boxes, output_masks)

while (rec.talking == True): # received audio
    if output_masks is None:
        print("No instance detected")
        break
    process_eye_gaze(frame, detector.image_width, detector.image_height, output_boxes, output_masks)

            # Check if within instances' bounding boxes
            # output_boxes = [[x1, y1, x2, y2], ...]
            # for i in range(output_boxes.shape[0]): # check if gaze is in box
            #     # Note top left of screen is (0, 0)
            #     left_x = output_boxes[i][0]
            #     top_y = output_boxes[i][1]
            #     right_x = output_boxes[i][2]
            #     bottom_y = output_boxes[i][3]
            #     if eye_gaze_x > left_x and eye_gaze_x < right_x and eye_gaze_y < bottom_y and eye_gaze_y > top_y:
            #         instance_weights[i] += gaze.FPOGD[-1]
            #         instance_fixation_counts[i] += 1
            #         output_image = cv2.circle(output_image.astype(np.uint8), (eye_gaze_x, eye_gaze_y), radius=2, color=(0, 0, 255), thickness=-1)

# Wait for recording to finish
# Can use rec.finished_rec to determine if the speech stopped before using rec_thread.join()
rec_thread.join()

print("Instances in the image:")
print(class_names)
print("Gaze Point Count:")
print(instance_fixation_counts)
print("Transcribed speech: ")
print(transcription)
class_name_speech_info, instance_speech_weights = match_segments_to_speech(class_names, word_offsets)
print("Class occurrences in speech: ")
for i in range(len(class_name_speech_info)):
    if class_name_speech_info[i][0] and class_name_speech_info[i][1]:
        print(class_name_speech_info[i][0], class_name_speech_info[i][1])

norm_eye_weights = []
norm_speech_weights = []
eye_gain = 1
speech_gain = 2
final_weights = 0
if np.sum(instance_eye_weights) != 0:
    print("Normalized eye weights of instances (gaze count*duration):")
    norm_eye_weights = instance_eye_weights / np.sum(instance_eye_weights)
    print(norm_eye_weights)
    final_weights = eye_gain*norm_eye_weights
if np.sum(instance_speech_weights) != 0:
    print("Normalized speech weights of instance: ")
    norm_speech_weights = instance_speech_weights / np.sum(instance_speech_weights)
    print(norm_speech_weights)
    final_weights += speech_gain*norm_speech_weights

xLoc = (output_boxes[np.argmax(final_weights)][0] + output_boxes[np.argmax(final_weights)][2])/2
yLoc = (output_boxes[np.argmax(final_weights)][1] + output_boxes[np.argmax(final_weights)][3])/2
xItem = [215, 333, 405, 536, 597, 659] # list of x locations
yItem = [376, 455, 351, 421, 322, 462] # list of y locations
pixelRange = 60 # 60 pixel range
msg = "-1" #initialize the message to none
for item in range(len(xItem)):
    if ((xItem[item] - pixelRange) < xLoc < (xItem[item] - pixelRange)) and ((yItem[item] - pixelRange) < yLoc < (yItem[item] - pixelRange)):
        msg = str(item)
        break

print("All instances weights normalized and multiplied with gains:")
print(final_weights)
print("Final predicted intended instance:")
print(class_names[np.argmax(final_weights)])
print("Final predicted location:")
print(msg)
cv2.imshow("Image_Input", output_image)
cv2.waitKey(0)

gaze.close_socket()
cv2.destroyAllWindows()
print("Released Window and Sockets")

in_str = input("Sending data...\n")
while in_str != " ":
    in_str = input("Sending data...\n")

print("Message to send:", msg)
if msg != "-1":
    bytesToSend         = str.encode(msg)
    serverAddressPort   = ("0.0.0.0", 4950)
    bufferSize          = 1024
    # Create a UDP socket at client side
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    # Send to server using created UDP socket
    UDPClientSocket.sendto(bytesToSend, serverAddressPort)
    print("SENT")


collected_data = {}
data_dir = "data"

cv2.namedWindow("Image_Input")
cv2.setMouseCallback("Image_Input", get_click_coords)

for i in range(2):
    # Click once then press a key to exit the image (in the order of desired object and predicted object)
    cv2.imshow("Image_Input", output_image)
    cv2.waitKey(0)
    instance_idx = in_instance_mask(mouse_x, mouse_y, output_boxes, output_masks, eye_gaze=False)
    # Save object class and bounding box
    if i == 0:
        collected_data["actual_object"] = {"instance idx" : instance_idx, "class": class_names[instance_idx], "bounding_box": list(output_boxes[instance_idx])}
    else:
        collected_data["predicted_object"] = {"instance idx" : instance_idx, "class": class_names[instance_idx], "bounding_box": list(output_boxes[instance_idx])}
# Save the timing of the first word and the
collected_data["start_timing"] = {"eye-gaze": eye_gaze_start_time, "word": speech_start_time}
# Save word timings as a list of lists in the format of [word, start time] for each recorded word
collected_data["word_timings"] = [[transcription[i], word_offsets[transcription[i]][i]["start_time"]] for i in range(len(transcription))]
# Save segmentation outputs as a list of lists in the format of [instance_idx, class, bounding box corner 1, bounding box corner 2] for each detected object
collected_data["segmentation_outputs"] = [[i, class_names[i], list(output_boxes[i][:2]), list(output_boxes[i][2:])] for i in range(len(class_names))]
# Save fixation eye gaze data as a list of lists in the format of [gaze x location, gaze y location, gaze time since system initialization] for each recorded gaze
collected_data["eye_gaze_data"] = [[int(gaze.FPOGX[i] * detector.image_width), int(gaze.FPOGY[i] * detector.image_height), gaze.REC_Time[i]] for i in range(len(gaze.FPOGX))]
# Create "data" folder if it does not exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
# Dump collected system data into the data folder as a json file
with open(os.path.join(data_dir, "collected_data.json"), "+w") as data_file:
    json.dump(collected_data, data_file)
# Save original and final images in the data folder
cv2.imwrite(os.path.join(data_dir, "original_camera_image.png"), frame)
cv2.imwrite(os.path.join(data_dir, "final_segemented_gaze_image.png"), output_image)
