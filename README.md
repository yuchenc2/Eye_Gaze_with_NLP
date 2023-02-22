
# Data Format
In each subject data, there are 60 trials in total with four experimental in the order of 
1. Target object has no occlusions + No duplicated target objects
2. Target object has no occlusions + 2 duplicated target objects
3. Target object has occlusions + No duplicated target objects
4. Target object has occlusions + 2 duplicated target objects
Each case contains 5 trials of orange, cup and scissors as the target (15 trials in total).

# Save the human desired instance's index, class, bounding box
collected_data["actual_object"] = {"instance idx" : instance_idx, "class": class_names[instance_idx], "bounding_box": list(output_boxes[instance_idx])}
# Save the eye-gaze and speech predicted instance's index, class, bounding box
collected_data["both predicted_object"] = {"instance idx" : np.argmax(final_weights), "class": class_names[np.argmax(final_weights)], "bounding_box": list(output_boxes[np.argmax(final_weights)])}
# (if eye-gaze is detected and contributed to the weights of the objects) Save the eye-gaze predicted desired instance's index, class, bounding box
collected_data["eye-gaze predicted_object"] = {"instance idx" : np.argmax(norm_eye_weights), "class": class_names[np.argmax(norm_eye_weights)], "bounding_box":
# (if speech is detected and contributed to the weights of the objects) Save the speech predicted desired instance's index, class, bounding box
collected_data["speech predicted_object"] = {"instance idx" : np.argmax(norm_speech_weights), "class": class_names[np.argmax(norm_speech_weights)], "bounding_box": list(output_boxes[np.argmax(norm_speech_weights)])}
# Save the timing of the first eye-gaze, first word in speech, and total time it takes from the last word to the final prediction
collected_data["start_timing"] = {"eye-gaze": eye-gaze_start_time, "word": speech_start_time, "total_time": inference_time}
# Save word timings as a list of lists in the format of [word, start time] for each recorded word
collected_data["word_timings"] = [[transcription[i], word_offsets[transcription[i]][i]["start_time"]] for i in range(len(transcription))]
# Save segmentation outputs as a list of lists in the format of [instance_idx, class, bounding box corner 1, bounding box corner 2] for each detected object
collected_data["segmentation_outputs"] = [[i, class_names[i], list(output_boxes[i][:2]), list(output_boxes[i][2:])] for i in range(len(class_names))]
# Save fixation eye gaze data as a list of lists in the format of [gaze x location, gaze y location, gaze time since system initialization (NOTE: This is the start time for the gazeAPI from the windows computer, so the first instance of the recorded time should be the same as the eye-gaze_start_time recorded above)] for each recorded gaze
collected_data["eye-gaze_data"] = [[int(gaze.FPOGX[i] * detector.image_width), int(gaze.FPOGY[i] * detector.image_height), gaze.REC_Time[i]] for i in range(len(gaze.FPOGX))]


For the units, collected_data["start_timing"], gaze.REC_Time[i], word_offsets[transcription[i]][i]["start_time"]] are all in seconds. For gaze.FPOGX[i] * detector.image_width, the value represents the X pixel value on the screen with the top left of the screen at (0, 0).
 

