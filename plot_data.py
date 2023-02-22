# importing the module
import json
import pdb
from glob import glob
import ast
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# # Check the total number of files
# subjects = ['jooyoung', 'kevin', 'kyungseo', 'omar', 'sankalp']
# data_files = []
# for sub in subjects:
#     sub_files = glob('./'+sub+'/*/*/*.json')
#     print('No. of files for '+sub+': ' +str(len(sub_files)))
#     data_files+=sub_files
# print('Total no. of files: '+str(len(data_files)))

def plot_graph(graph_name):
    
    cases = ['Case 1', 'Case 2', 'Case 3', 'Case 4'] # 'Xocc_Xdup','Xocc_dup','occ_Xdup', 'occ_dup'
    methods = ['speech','eye-gaze', 'speech_eye-gaze'] #{'speech':[],'eye-gaze': [], 'speech_eye-gaze':[]}
    objects = ['cup','orange', 'scissors']
    pred_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) 
    
    subjects = ['jooyoung', 'kevin', 'kyungseo', 'omar', 'sankalp']

    ### Graph1: Prediction accuracy bar graph
    if graph_name == "pred_acc":      
        for subject in subjects:
            for obj in objects:
                for case in cases:                
                    files = glob('./'+subject+'/'+case+'/'+obj+'/'+'*.json')             
                    for file in files:
                        with open(file) as json_file:
                            data_str = json.load(json_file)
                            data = ast.literal_eval(data_str)
                            # data.keys: (['actual_object', 'both predicted_object', 'start_timing', 'word_timings', 'segmentation_outputs', 
                            # 'eye-gaze_data', 'eye-gaze predicted_object', 'speech predicted_object'])

                            # Count only if the 'actual_object' is correct
                            if data['actual_object']['class'] == obj:         
                                if 'speech predicted_object' in data.keys():
                                    speech_result = int(np.logical_and(data['speech predicted_object']['class']==obj, data['speech predicted_object']['instance idx']==data['actual_object']['instance idx']))
                                else: # ! if the speech prediction fails, consider incorrect
                                    speech_result = 0
                                pred_acc[obj][case]['speech'].append(speech_result)  
                                    
                                    
                                if 'eye-gaze predicted_object' in data.keys():
                                    eye_gaze_result = int(np.logical_and(data['eye-gaze predicted_object']['class']==obj, data['eye-gaze predicted_object']['instance idx']==data['actual_object']['instance idx']))
                                else: # ! if the eye-gaze prediction fails, consider incorrect
                                    eye_gaze_result = 0
                                pred_acc[obj][case]['eye-gaze'].append(eye_gaze_result)
                                
                                if 'both predicted_object' in data.keys():# 'speech predicted_object' and  'eye-gaze predicted_object' in data.keys():
                                    speech_eye_gaze_result = int(np.logical_and(data['both predicted_object']['class']==obj, data['both predicted_object']['instance idx']==data['actual_object']['instance idx']))    
                                else: # ! if either the eye_gaze result , consider incorrect
                                    speech_eye_gaze_result = 0                                                                            
                                pred_acc[obj][case]['speech_eye-gaze'].append(speech_eye_gaze_result)


        
        # Build the prediction accuracy plot 
        titles = ['Cup', 'Orange', 'Scissors']       
        for obj, title in zip(objects, titles):
            x_pos = np.arange(len(cases))

            fig, ax = plt.subplots()        
            n=4
            r = np.arange(n)
            width = 0.25
            colors = ['y','g', 'b']
            
            for i, method in enumerate(methods):      
                method_acc = [np.mean(pred_acc[obj][case][method])*100 for case in cases]
                plt.bar(x_pos+width*i, method_acc, color = colors[i],
                        width = width, edgecolor = colors[i],
                        label=method)
            
                # Check how many data are available for evaluation
                print("Number of available samples for ", obj, method," (4 cases):", [len(pred_acc[obj][case][method]) for case in cases])  # 291
                # Number of available samples for  cup speech  (4 cases): [24, 24, 25, 25]
                # Number of available samples for  cup eye-gaze  (4 cases): [24, 24, 25, 25]
                # Number of available samples for  cup speech_eye-gaze  (4 cases): [24, 24, 25, 25]
                # Number of available samples for  orange speech  (4 cases): [24, 25, 24, 25]
                # Number of available samples for  orange eye-gaze  (4 cases): [24, 25, 24, 25]
                # Number of available samples for  orange speech_eye-gaze  (4 cases): [24, 25, 24, 25]
                # Number of available samples for  scissors speech  (4 cases): [25, 25, 25, 20]
                # Number of available samples for  scissors eye-gaze  (4 cases): [25, 25, 25, 20]
                                
            # plt.ylim([60,100])    
            plt.xlabel("Cases")
            plt.ylabel("Prediction Accuracy (%)")
            plt.title(title)
            
            # plt.grid(linestyle='--')
            plt.xticks(r + width/2,['Case 1', 'Case 2', 'Case 3', 'Case 4'])
            plt.legend(loc = "lower left")
            # ax.yaxis.grid(True)

            # Save the figure and show
            plt.tight_layout()
            plt.savefig('prediction_acc_'+obj+'.png')
            plt.show()               

    ### Graph2: target gaze portion vs speech start time plot
    elif graph_name == 'target_gaze_portion':        
        window_size = 0.1
        win_gaze_portion = {'Case 1':[], 'Case 2':[], 'Case 3':[], 'Case 4':[]} 

        for subject in subjects:
            for case in cases:
                for obj in objects:
                    files = glob('./'+subject+'/'+case+'/'+obj+'/'+'*.json')             
                    for file in files:
                        with open(file) as json_file:
                            data_str = json.load(json_file)
                            data = ast.literal_eval(data_str)
                            # data.keys: (['actual_object', 'both predicted_object', 'start_timing', 'word_timings', 'segmentation_outputs', 
                            # 'eye-gaze_data', 'eye-gaze predicted_object', 'speech predicted_object'])

                            # Count only if the 'actual_object' is correct 
                            if data['actual_object']['class'] == obj:    
                                speech_start_time = data['start_timing']['word']
                                # If eye-gaze data exists
                                if data['eye-gaze_data']:
                                    eye_gaze_x, eye_gaze_y, eye_gaze_time = np.array(data['eye-gaze_data']).T
                                    
                                    # Obtain eye-gaze time that matches with speech clock
                                    eye_gaze_start_time = data['start_timing']['eye-gaze'] 
                                    eye_gaze_time = eye_gaze_time-eye_gaze_time[0]
                                    eye_gaze_time += eye_gaze_start_time 
                                    eye_gaze_relTime = eye_gaze_time - speech_start_time
                                    
                                    # Check if the gaze was inside the bounding box of the target object
                                    left_x, top_y, right_x, bottom_y = data['actual_object']['bounding_box']    
                                    correct_pred_flag = np.logical_and(np.logical_and(np.logical_and(eye_gaze_x > left_x, eye_gaze_x < right_x), eye_gaze_y < bottom_y),eye_gaze_y > top_y) 
                                    
                                    # Make 0.1s timewindow eye-gaze data for relative time between -4s~4s (speech start time as 0s).
                                    win_start_t = np.arange(-6, 6+window_size, window_size) 
                                    win_flag = [np.logical_and(eye_gaze_relTime>win_start_t[i], eye_gaze_relTime<= win_start_t[i+1]) for i in range(len(win_start_t)-1)]
                                    win_gaze_portion[case].append([(eye_gaze_relTime[flag], correct_pred_flag[flag].astype(int)) if np.any(flag) else (np.array([np.nan]), np.array([np.nan])) for flag in win_flag])
                    

        # Check how many data are available for evaluation
        print("Number of available samples:", np.sum([len(win_gaze_portion[case]) for case in cases])) # 272/300

        # Build the gaze portion plot                      
        x_pos = np.arange(len(win_start_t))
        colors = ['r','g', 'b', 'y']

        fig, ax = plt.subplots()            
            
        for case, color in zip(cases, colors):
            temp = np.array(win_gaze_portion[case], dtype=object)[:,:,1]
            avg_gaze_portion = [np.nanmean(np.hstack(temp[:,i])) if np.any(~np.isnan(np.hstack(temp[:,i]))) else np.nan for i in range(temp.shape[1])]
            plt.plot(win_start_t[:-1], avg_gaze_portion, color=color, label=case) 

        plt.xlabel("Relative time with respect to the start of the speech")
        plt.ylabel("Target gaze portion")
        plt.title("Relationship between the target gaze portion and the speech starting time")
        plt.legend(loc='lower right')

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('target_gaze_portion.png')
        plt.show()               



        ### Graph3: NASA-TRX

graph_name = "pred_acc" # "target_gaze_portion" # nasa_trx
plot_graph(graph_name)

