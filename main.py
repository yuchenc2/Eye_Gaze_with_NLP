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
import numpy as np


# Global variables for getting results from recording thread
transcription = []
word_offsets = {}
audio_recording = False


def record_and_transcribe(rec, recording_filename="output.wav"):
    # Record until silence is detected for 2 seconds
    global transcription
    global word_offsets
    rec_frames = rec.record_until_silence()
    rec.stop_recording()
    rec.save_recording(rec_frames, recording_filename)
    # Transcribe the recording
    asr = ASR()
    # Outputs in the form [word for word in recording], {word: {index: {start_time: float, end_time: float}} for word in recording}
    transcription, word_offsets = asr.asr_transcript(recording_filename)
    return transcription, word_offsets


def match_segments_to_speech(class_names, word_offsets):
    class_name_speech_info = []
    for name in class_names:
        class_info = word_offsets[name] if name in word_offsets else {}
        class_name_speech_info.append((name, class_info))
    return class_name_speech_info


# set image quality
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# move the image to the second screen and full screen it
cv2.namedWindow("Image_Input", cv2.WINDOW_NORMAL)
cv2.moveWindow("Image_Input", 900, 0)
cv2.setWindowProperty("Image_Input", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# Start recording on a thread
rec = Recorder(silence_threshold=150, silence_timeout=2)
rec_thread = Thread(target=record_and_transcribe, args=(rec,), daemon=True)
rec_thread.start()


# Capture the image now (since it's a static scene)
ret, frame = cam.read()
img_name = "captured.png"
cv2.imwrite(img_name, frame)
print("{} written!".format(img_name))
cam.release()

# Detect item in image
detector = Detector(model_type="IS")
output_image, output_boxes, class_names, output_masks = detector.onImage("captured.png")


# Capture the eye gaze
gaze = Gaze()
while (gaze.time == 1) or (gaze.time == 0):
    gaze.eye_gaze_capture()
init_time = gaze.time

# Initialize the instance weights
instance_weights = np.zeros(output_boxes.shape[0])
instance_fixation_counts = np.zeros(output_boxes.shape[0])
# print(output_boxes.shape)

# Check if eye gaze is in any instance
while (audio_recording == False): # wait until audio command
    cv2.imshow("Image_Input", frame)
    cv2.waitKey(1)
    gaze.eye_gaze_capture()
    if gaze.FPOGX and gaze.FPOGY: # if list is not empty
        eye_gaze_x = int(gaze.FPOGX[-1]*detector.image_width) 
        eye_gaze_y = int(gaze.FPOGY[-1]*detector.image_height)

        # Check if within instances' masks
        point = Point(eye_gaze_x, eye_gaze_y)
        for i in range(output_boxes.shape[0]):
            for segment in output_masks[i].polygons:
                segment = segment.reshape(-1, 2)
                polygon = Polygon(segment)
                if polygon.contains(point):
                    instance_weights[i] += gaze.FPOGD[-1]
                    instance_fixation_counts[i] += 1
                    output_image = cv2.circle(output_image.astype(np.uint8), (eye_gaze_x, eye_gaze_y), radius=2, color=(0, 0, 255), thickness=-1)

while (audio_recording == True): # received audio
    if output_masks is None:
        print("No instance detected")
        break
    cv2.imshow("Image_Input", output_image)
    cv2.waitKey(1)
    if gaze.eye_gaze_capture(): # if it's a fixation
        if gaze.FPOGX and gaze.FPOGY: # if list is not empty
            eye_gaze_x = int(gaze.FPOGX[-1]*detector.image_width) 
            eye_gaze_y = int(gaze.FPOGY[-1]*detector.image_height)

            # Check if within instances' masks
            point = Point(eye_gaze_x, eye_gaze_y)
            for i in range(output_boxes.shape[0]):
                for segment in output_masks[i].polygons:
                    segment = segment.reshape(-1, 2)
                    polygon = Polygon(segment)
                    if polygon.contains(point):
                        instance_weights[i] += gaze.FPOGD[-1]
                        instance_fixation_counts[i] += 1
                        output_image = cv2.circle(output_image.astype(np.uint8), (eye_gaze_x, eye_gaze_y), radius=2, color=(0, 0, 255), thickness=-1)


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
print("Normalized weights of instances (gaze count*duration):")
print(instance_weights / np.sum(instance_weights))
print("Gaze Point Count:")
print(instance_fixation_counts)
print("Final predicted intended instance:")
print(class_names[np.argmax(instance_weights)])
print("Transcribed speech: ")
print(transcription)
class_name_speech_info = match_segments_to_speech(class_names, word_offsets)
print("Class occurrences in speech: ")
for class_info in class_names:
    print(class_info[0], ": ", class_info[1])

gaze.close_socket()
cv2.destroyAllWindows()
print("Released Window and Sockets")
