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
from geometry_msgs.msg import Pose, PoseStamped
import os
import json
import time

import rospy
from papras_stand.command_interface import DualArmCommandInterface

# Global variables for getting results from recording thread
transcription = []
word_offsets = {}
audio_recording = False
speech_start_time = 0.0
start_time = 0.0

# Global variables for getting last mouse click coordinates on a cv2 image
mouse_x = 0
mouse_y = 0

def record_and_transcribe(rec, recording_filename="output.wav"):
    # Record until silence is detected for 2 seconds
    global transcription
    global word_offsets
    global start_time
    global speech_start_time
    rec_frames, speech_start_time = rec.record_until_silence()
    rec.stop_recording()
    rec.save_recording(rec_frames, recording_filename)
    start_time = time.time()
    # Transcribe the recording
    asr = ASR()
    # Outputs in the form [word for word in recording], {word: {index: {start_time: float, end_time: float}} for word in recording}
    transcription, word_offsets = asr.asr_transcript(recording_filename)
    return transcription, word_offsets

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

def in_instance_mask(point_x, point_y, bounding_boxes, masks, instance_eye_weights, instance_fixation_counts, gaze, output_image, eye_gaze=False):
    point = Point(point_x, point_y)
    for i in range(bounding_boxes.shape[0]):
        for segment in masks[i].polygons:
            segment = segment.reshape(-1, 2)
            polygon = Polygon(segment)
            if polygon.contains(point) and eye_gaze:
                print("Fixation within instance!")
                instance_eye_weights[i] += gaze.FPOGD[-1]
                instance_fixation_counts[i] += 1
                output_image = cv2.circle(output_image.astype(np.uint8), (point_x, point_y), radius=2, color=(0, 0, 255), thickness=-1)
            elif polygon.contains(point):
                return i
    return -1

def process_eye_gaze(frame, image_width, image_height, bounding_boxes, masks, instance_eye_weights, instance_fixation_counts, gaze, output_image):
    cv2.imshow("Image_Input", frame)
    cv2.waitKey(1)
    if gaze.eye_gaze_capture(): # if it's a fixation
        # print("Fixation detected!")
        if gaze.FPOGX and gaze.FPOGY: # if list is not empty
            # print("Fixation coordinates: ", gaze.FPOGX[-1], gaze.FPOGY[-1])
            eye_gaze_x = int(gaze.FPOGX[-1] * image_width)
            eye_gaze_y = int(gaze.FPOGY[-1] * image_height)
            # Check if within instances' masks
            in_instance_mask(eye_gaze_x, eye_gaze_y, bounding_boxes, masks, instance_eye_weights, instance_fixation_counts, gaze, output_image, eye_gaze=True)

# def get_user_eyegaze_nlp_command(cnt):
#     # Change VideoCaptureindex to 2 for laptop, 0 for desktop
    
#     return int (msg)

 

# main function
if __name__ == '__main__':

    rospy.init_node("eyegaze_demo")
    rospy.loginfo("eyegaze_demo node started")

    stand = DualArmCommandInterface()

    stand.close_gripper(stand.gripper_group1_2)
    
    for cnt in range(30):
        stand.move_arm_to_named_pose("eye_gaze_init", stand.arm_group1_2)
        

        # move the image to the second screen and full screen it
        cv2.namedWindow("Image_Input", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Image_Input", 900, 0)
        cv2.setWindowProperty("Image_Input", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


        cam = cv2.VideoCapture(2)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # while not cam.isOpened():
        #     print("wait for image")
        # Capture the image now (since it's a static scene)
        ret, frame = cam.read()
        img_name = "captured.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))



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
            process_eye_gaze(frame, detector.image_width, detector.image_height, output_boxes, output_masks, instance_eye_weights, instance_fixation_counts, gaze, output_image)

        while (rec.talking == True): # received audio
            if output_masks is None:
                print("No instance detected")
                break
            process_eye_gaze(frame, detector.image_width, detector.image_height, output_boxes, output_masks, instance_eye_weights, instance_fixation_counts, gaze, output_image)

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

        # print("Instances in the image:")
        # print(class_names)
        # print("Gaze Point Count:")
        # print(instance_fixation_counts)
        # print("Transcribed speech: ")
        # print(transcription)
        class_name_speech_info, instance_speech_weights = match_segments_to_speech(class_names, word_offsets)
        # print("Class occurrences in speech: ")
        # for i in range(len(class_name_speech_info)):
        #     if class_name_speech_info[i][0] and class_name_speech_info[i][1]:
        #         print(class_name_speech_info[i][0], class_name_speech_info[i][1])

        norm_eye_weights = []
        norm_speech_weights = []
        eye_gain = 1
        speech_gain = 1
        final_weights = 0
        if np.sum(instance_eye_weights) != 0:
            # print("Normalized eye weights of instances (gaze count*duration):")
            norm_eye_weights = instance_eye_weights / np.sum(instance_eye_weights)
            # print(norm_eye_weights)
            final_weights = eye_gain*norm_eye_weights
        if np.sum(instance_speech_weights) != 0:
            # print("Normalized speech weights of instance: ")
            norm_speech_weights = instance_speech_weights / np.sum(instance_speech_weights)
            # print(norm_speech_weights)
            final_weights += speech_gain*norm_speech_weights
        
        inference_time = time.time() - start_time
        print("Inference time: ", inference_time)

        xLoc = (output_boxes[np.argmax(final_weights)][0] + output_boxes[np.argmax(final_weights)][2])/2
        yLoc = (output_boxes[np.argmax(final_weights)][1] + output_boxes[np.argmax(final_weights)][3])/2
        xItem = [423, 664, 801, 1066, 1183, 1313] # list of x locations
        yItem = [746, 899, 691, 825, 628, 898] # list of y locations
        pixelRange = 100 # 60 pixel range
        msg = "-1" #initialize the message to none
        for item in range(len(xItem)):
            if ((xItem[item] - pixelRange) < xLoc < (xItem[item] + pixelRange)) and ((yItem[item] - pixelRange) < yLoc < (yItem[item] + pixelRange)):
                msg = str(item)
                break
        
        print("All instances weights normalized and multiplied with gains:")
        print(final_weights)
        print("Final predicted intended instance:")
        print(class_names[np.argmax(final_weights)])
        print("Final predicted location:")
        print(msg)
        # cv2.imshow("Image_Input", output_image)
        # cv2.waitKey(0)

        gaze.close_socket()
        # cv2.destroyAllWindows()
        # print("Released Window and Sockets")

        # in_str = input("Sending data...\n")
        # while in_str != " ":
        #     in_str = input("Sending data...\n")

        # print("Message to send:", msg)
        # if msg != "-1":
        #     bytesToSend         = str.encode(msg)
        #     serverAddressPort   = ("0.0.0.0", 4950)
        #     bufferSize          = 1024
        #     # Create a UDP socket at client side
        #     UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        #     # Send to server using created UDP socket
        #     UDPClientSocket.sendto(bytesToSend, serverAddressPort)
        #     print("SENT")


        collected_data = {}
        data_dir = "data/jooyoung/case2"

        loc_idx = int(msg)
        # loc_idx = 1

        prepoke_joint_values = [[0.07291303631227386, 0.5727112217412902, 0.05970706001502002, -1.5218245377004473, -1.5162562064304979, 2.5258994365026552], # left robot1 #1
                                [0.08566892725123143, 0.7206821449013185, 0.4703561566852317, -1.5080306788997362, -1.5532083902621618, 1.9684195776976718], # left robot1 #2
                                [0.05987169579280138, 1.1799086845388453, -0.22257622664862797, -1.520259614870513, -1.5439103934939604, 2.2022536389580836], # left robot1 #3
                                [-0.08975905983943111, 1.2715264131236221, 0.08445488021920955, 1.4894119098338656, -1.551110055288004, -1.769769183431059], # right robot2 #3
                                [-0.06983775679240178, 1.292009913837342, -0.43982162691453386, 1.5238818524815345, -1.527506827439498, -2.27314040397696], # right robot2 #2
                                [-0.10178388396705973, 0.7780522888303469, 0.5259091251981474, 1.4790568155794954, -1.5435985626435418, -1.8210946668689019], # right robot2 #1
                            ]
        poke_joint_values = [[0.2922703282526138, 0.6272434969362628, -0.03896776729012874, -1.3992546451795107, -1.332639151950402, 2.550092339662555], # left robot1 #1
                            [0.33633080312035357, 0.7490398417712658, 0.39408278952441567, -1.2789883919645977, -1.4480534201297202, 1.99816447135818], # left robot1 #2
                            [0.2545491912937319, 1.2164231511347667, -0.3146777768739346, -1.3666479042540471, -1.42071881050504, 2.242631960682264], # left robot1 #3
                            [-0.30419751913274506, 1.2914498646548145, 0.007959937032882003, 1.2836185877929749, -1.489926083831154, -1.8153829014774212], # right robot2 #3
                            [-0.23591255587598692, 1.3361930262601112, -0.542108056598746, 1.4067208680605878, -1.4094097068576223, -2.3189085465685952], # right robot2 #2  
                            [-0.38185976730054705, 0.8119778110845326, 0.4255304903644497, 1.2148877981245887, -1.4484765001245767, -1.8668621116267872], # right robot2 #1
                        ]

        idx = loc_idx
        if idx == -1:
            continue
        
        if idx < 3:
            stand.move_left_arm_to_joint_values(prepoke_joint_values[idx], 2)
            rospy.sleep(1.0)
            # ret, frame = cam.read()
            # cv2.imshow("Image_Input", frame)
            stand.move_left_arm_to_joint_values(poke_joint_values[idx], 1)
            rospy.sleep(1.0)
            # ret, frame = cam.read()
            # cv2.imshow("Image_Input", frame)
            stand.move_left_arm_to_joint_values(prepoke_joint_values[idx], 2)
        else:
            stand.move_right_arm_to_joint_values(prepoke_joint_values[idx], 2)
            rospy.sleep(1.0)

            # ret, frame = cam.read()
            # cv2.imshow("Image_Input", frame)
            stand.move_right_arm_to_joint_values(poke_joint_values[idx], 1)
            rospy.sleep(1.0)

            # ret, frame = cam.read()
            # cv2.imshow("Image_Input", frame)
            stand.move_right_arm_to_joint_values(prepoke_joint_values[idx], 2)
        
        rospy.sleep(2.0)
        ret, frame = cam.read()
        cv2.imshow("Image_Input", frame)
        stand.move_arm_to_named_pose("eye_gaze_init", stand.arm_group1_2)
        cv2.namedWindow("Image_Input")
        cv2.setMouseCallback("Image_Input", get_click_coords)


        ret, frame = cam.read()
        cv2.imshow("Image_Input", frame)

        cam.release()
        # Click once then press a key to exit the image (desired object)
        cv2.imshow("Image_Input", output_image)
        cv2.waitKey(0)
        instance_idx = in_instance_mask(mouse_x, mouse_y, output_boxes, output_masks,  instance_eye_weights, instance_fixation_counts, gaze, output_image, eye_gaze=False)
        # Save object class and bounding box
        collected_data["actual_object"] = {"instance idx" : instance_idx, "class": class_names[instance_idx], "bounding_box": list(output_boxes[instance_idx])}    
        collected_data["both predicted_object"] = {"instance idx" : np.argmax(final_weights), "class": class_names[np.argmax(final_weights)], "bounding_box": list(output_boxes[np.argmax(final_weights)])}
        # Save the timing of the first word and the
        collected_data["start_timing"] = {"eye-gaze": eye_gaze_start_time, "word": speech_start_time, "total_time": inference_time}
        # Save word timings as a list of lists in the format of [word, start time] for each recorded word
        if transcription is not None and word_offsets is not None:
            collected_data["word_timings"] = [[transcription[i], word_offsets[transcription[i]][i]["start_time"]] for i in range(len(transcription))]
        # Save segmentation outputs as a list of lists in the format of [instance_idx, class, bounding box corner 1, bounding box corner 2] for each detected object
        collected_data["segmentation_outputs"] = [[i, class_names[i], list(output_boxes[i][:2]), list(output_boxes[i][2:])] for i in range(len(class_names))]
        # Save fixation eye gaze data as a list of lists in the format of [gaze x location, gaze y location, gaze time since system initialization] for each recorded gaze
        collected_data["eye_gaze_data"] = [[int(gaze.FPOGX[i] * detector.image_width), int(gaze.FPOGY[i] * detector.image_height), gaze.REC_Time[i]] for i in range(len(gaze.FPOGX))]
        # Prediction using eye-gaze vs speech only
        if np.sum(instance_eye_weights) != 0:
            collected_data["eye-gaze predicted_object"] = {"instance idx" : np.argmax(norm_eye_weights), "class": class_names[np.argmax(norm_eye_weights)], "bounding_box": list(output_boxes[np.argmax(norm_eye_weights)])}
        if np.sum(instance_speech_weights) != 0:
            collected_data["speech predicted_object"] = {"instance idx" : np.argmax(norm_speech_weights), "class": class_names[np.argmax(norm_speech_weights)], "bounding_box": list(output_boxes[np.argmax(norm_speech_weights)])}
        
        # Create "data" folder if it does not exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # Dump collected system data into the data folder as a json file
        with open(os.path.join(data_dir, str(cnt) + "_collected_data.json"), "+w") as data_file:
            json.dump(str(collected_data), data_file)
        # Save original and final images in the data folder
        cv2.imwrite(os.path.join(data_dir, str(cnt) + "_original_camera_image.png"), frame)
        cv2.imwrite(os.path.join(data_dir, str(cnt) + "_final_segemented_gaze_image.png"), output_image)
        cv2.destroyAllWindows()
        rospy.sleep(1)
        keyinput = input("Press Enter to continue experiment...")
        if keyinput == "q":
            exit()

    # stand.move_arm_to_named_pose("eye_gaze_init", stand.arm_group1_2)
    # rostopic echo /joint_states/position[2:8] -n 1
    # rostopic echo /joint_states/position[10:] -n 1
    # spin until shutdown
    
    rospy.spin()
    rospy.signal_shutdown("eyegaze_demo node finished")