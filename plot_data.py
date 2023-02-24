# importing the module
import json
import pdb
from glob import glob
import ast
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import csv
import seaborn as sns
import pandas as pd

#Direct input 
# plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
# #Options
# params = {'text.usetex' : True,
#           'font.size' : 11,
#           'font.family' : 'lmodern',
#           'text.latex.unicode': True,
#           }
# plt.rcParams.update(params) 

import matplotlib.colors as mcolors
import matplotlib
def set_style():
    font = {'family' : "serif",
            # 'weight' : 'bold',
            'size'   : 15}

    matplotlib.rc('font', **font)
    
def get_colors():
            return np.array([
                [0.4, 0.4, 0.4],          # very dark gray
                [0.7, 0.7, 0.7],          # dark gray
                [0.984375, 0.7265625, 0], #'#FFEC8B', #[0.984375, 0.7265625, 0], # yellow
                '#CD8500'#8B6914' #'#8B7500' #'#CDAD00', # darker yellow
                
                # '#607B8B', # dark blue    
                # '#8DB6CD', # light blue                
                # '#CD8162',   # dark orange      
                # '#FFA07A'    # light orange
            ]*12)
            
def darker_colors():
            return np.array([
                '#363636',          # very dark gray
                '#787878',          # dark gray
                '#EE9A00', #'#FFEC8B', #[0.984375, 0.7265625, 0], # yellow
                '#C76114'#8B6914' #'#8B7500' #'#CDAD00', # darker yellow
                
                # '#607B8B', # dark blue    
                # '#8DB6CD', # light blue                
                # '#CD8162',   # dark orange      
                # '#FFA07A'    # light orange
            ]*12)
            
        

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
        
        # # # Plot of each object                  
        # titles = ['Cup', 'Orange', 'Scissors']     
        # # for obj, title in zip(objects, titles):  
        # obj = objects[2]
        # title = titles[2]
        # correct_data = []
        # cases_data = []
        # methods_data = []
        # bar = None
        # for i, method in enumerate(methods):                  
        #     # method_acc = [np.mean(pred_acc[obj][case][method])*100 for case in cases]
        #     dt = [pred_acc[obj][case][method] for case in cases]
        #     correct_data.append(dt)
        #     cases_data.append([[c]*len(dt[i])for i,c in enumerate(cases)])
        #     methods_data.append([method]*len(sum(dt,[])))
        #     # correct_data.append(method_acc)
            
        # #flatten
        # correct_data = sum(sum(correct_data,[]),[])
        # cases_data = sum(sum(cases_data,[]),[])
        # methods_data = sum(methods_data,[])
        # pred_acc_data = np.stack([cases_data, methods_data, np.array(correct_data)*100]).T   
        # # data_cases = cases*
        
        # head = ['cases', 'methods', 'correct']
        # pred_acc_data = pd.DataFrame(pred_acc_data, columns=head)
        # pred_acc_data['correct'] = pred_acc_data.correct.astype(float)  

        # g = sns.barplot(
        #     data=pred_acc_data,#, kind="bar",
        #     x="cases", y="correct", hue="methods"#,
        #     , ci=None #height=8 , ci=None #"sd"  # palette=sns.color_palette(get_colors()) #alpha=.6,
        # )
                        
        # bars = g.patches
                    
        # hatches = np.repeat([None, '..', '////'],4)
        # # colors = np.repeat(get_colors().tolist(),4)
        # colors = get_colors()
        # dark_colors = darker_colors()
        
        # # for i, bar in enumerate(bars):
        # #     bar.set_color(colors[i])
        # i = 0
        # for bar in bars:
        #     print(i)
        #     bar.set_color(colors[i])
        #     bar.set_edgecolor('black') # dark_colors[i]''
        #     bar.set_hatch(hatches[i])
        #     i+=1        
                        
        # g.set(xlabel = "", ylabel = "Accuracy (%)", title=title)
        # g.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
        # plt.savefig('prediction_acc_'+obj+'.png',dpi=300,bbox_inches="tight")  
            
            
        ## Plot average of all objects
        correct_data = []
        cases_data = []
        methods_data = []
        for obj in objects:  
            for i, method in enumerate(methods):                  
                # method_acc = [np.mean(pred_acc[obj][case][method])*100 for case in cases]
                dt = [pred_acc[obj][case][method] for case in cases]
                correct_data.append(dt)
                cases_data.append([[c]*len(dt[i])for i,c in enumerate(cases)])
                methods_data.append([method]*len(sum(dt,[])))
                # correct_data.append(method_acc)
                
        #flatten
        correct_data = sum(sum(correct_data,[]),[])
        cases_data = sum(sum(cases_data,[]),[])
        methods_data = sum(methods_data,[])
        pred_acc_data = np.stack([cases_data, methods_data, np.array(correct_data)*100]).T   
        # data_cases = cases*
        
        head = ['cases', 'methods', 'correct']
        pred_acc_data = pd.DataFrame(pred_acc_data, columns=head)
        pred_acc_data['correct'] = pred_acc_data.correct.astype(float)  

        g = sns.barplot(
            data=pred_acc_data,#, kind="bar",
            x="cases", y="correct", hue="methods"#,
            , ci=None #height=8 , ci=None #"sd"  # palette=sns.color_palette(get_colors()) #alpha=.6,
        )
        
        bars = g.patches
                    
        hatches = np.repeat([None, '..', '////'],4)
        colors = get_colors()
        dark_colors = darker_colors()
        
        for i, bar in enumerate(bars):
            bar.set_color(colors[i])
            bar.set_edgecolor('black') # dark_colors[i]''
            bar.set_hatch(hatches[i])
    
                        
        g.set(xlabel = "", ylabel = "Accuracy (%)", title="All objects")
        g.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    
        plt.savefig('prediction_acc_all.png',dpi=300,bbox_inches="tight")  
        

        
        # # # Build the prediction accuracy plot 
        # titles = ['Cup', 'Orange', 'Scissors']       
        # for obj, title in zip(objects, titles):
        #     x_pos = np.arange(len(cases))

        #     fig, ax = plt.subplots()        
        #     n=4
        #     r = np.arange(n)
        #     width = 0.25
        #     colors = ['y','g', 'b']
            
        #     for i, method in enumerate(methods):      
        #         method_acc = [np.mean(pred_acc[obj][case][method])*100 for case in cases]
        #         plt.bar(x_pos+width*i, method_acc, color = colors[i],
        #                 width = width, edgecolor = colors[i],
        #                 label=method)
            
        #         # Check how many data are available for evaluation
        #         print("Number of available samples for ", obj, method," (4 cases):", [len(pred_acc[obj][case][method]) for case in cases])  # 291
        #         # Number of available samples for  cup speech  (4 cases): [24, 24, 25, 25]
        #         # Number of available samples for  cup eye-gaze  (4 cases): [24, 24, 25, 25]
        #         # Number of available samples for  cup speech_eye-gaze  (4 cases): [24, 24, 25, 25]
        #         # Number of available samples for  orange speech  (4 cases): [24, 25, 24, 25]
        #         # Number of available samples for  orange eye-gaze  (4 cases): [24, 25, 24, 25]
        #         # Number of available samples for  orange speech_eye-gaze  (4 cases): [24, 25, 24, 25]
        #         # Number of available samples for  scissors speech  (4 cases): [25, 25, 25, 20]
        #         # Number of available samples for  scissors eye-gaze  (4 cases): [25, 25, 25, 20]
                                
        #     # plt.ylim([60,100])    
        #     plt.xlabel("Cases")
        #     plt.ylabel("Prediction Accuracy (%)")
        #     plt.title(title)
            
        #     # plt.grid(linestyle='--')
        #     plt.xticks(r + width/2,['Case 1', 'Case 2', 'Case 3', 'Case 4'])
        #     plt.legend(loc = "lower left")
        #     # ax.yaxis.grid(True)

        #     # Save the figure and show
        #     plt.tight_layout()
        #     plt.savefig('prediction_acc_'+obj+'.png')
        #     plt.show()      
            
        # # # Average of all objects
        # x_pos = np.arange(len(cases))

        # fig, ax = plt.subplots()        
        # n=4
        # r = np.arange(n)
        # width = 0.25
        # colors = ['y','g', 'b']
        
        # for i, method in enumerate(methods):      
        #     method_acc = []
        #     for case in cases:
        #         method_acc.append(np.mean(np.hstack([pred_acc[obj][case][method] for obj in objects])))
        #         # pdb.set_trace()
                    
                    
        #     # method_acc = [np.mean(pred_acc[obj][case][method])*100 for case in cases]
        #     plt.bar(x_pos+width*i, method_acc, color = colors[i],
        #             width = width, edgecolor = colors[i],
        #             label=method)
        
        #     # Check how many data are available for evaluation
        #     print("Number of available samples for ", obj, method," (4 cases):", [len(pred_acc[obj][case][method]) for case in cases])  # 291
        #     # Number of available samples for  cup speech  (4 cases): [24, 24, 25, 25]
        #     # Number of available samples for  cup eye-gaze  (4 cases): [24, 24, 25, 25]
        #     # Number of available samples for  cup speech_eye-gaze  (4 cases): [24, 24, 25, 25]
        #     # Number of available samples for  orange speech  (4 cases): [24, 25, 24, 25]
        #     # Number of available samples for  orange eye-gaze  (4 cases): [24, 25, 24, 25]
        #     # Number of available samples for  orange speech_eye-gaze  (4 cases): [24, 25, 24, 25]
        #     # Number of available samples for  scissors speech  (4 cases): [25, 25, 25, 20]
        #     # Number of available samples for  scissors eye-gaze  (4 cases): [25, 25, 25, 20]
                            
        # # plt.ylim([60,100])    
        # plt.xlabel("Cases")
        # plt.ylabel("Prediction Accuracy (%)")
        # plt.title("All objects")
        
        # # plt.grid(linestyle='--')
        # plt.xticks(r + width/2,['Case 1', 'Case 2', 'Case 3', 'Case 4'])
        # plt.legend(loc = "lower left")
        # # ax.yaxis.grid(True)

        # # Save the figure and show
        # plt.tight_layout()
        # plt.savefig('prediction_acc_all.png')
        # # plt.show()                        

    ### Graph2: target gaze portion vs speech start time plot
    elif graph_name == "target_gaze_portion":   
        from palettable.colorbrewer.qualitative import Set2_7
     
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
                                    win_start_t = np.arange(-2, 2+window_size, window_size) 
                                    win_flag = [np.logical_and(eye_gaze_relTime>win_start_t[i], eye_gaze_relTime<= win_start_t[i+1]) for i in range(len(win_start_t)-1)]
                                    win_gaze_portion[case].append([(eye_gaze_relTime[flag], correct_pred_flag[flag].astype(int)) if np.any(flag) else (np.array([np.nan]), np.array([np.nan])) for flag in win_flag])
                    

        # Check how many data are available for evaluation
        print("Number of available samples:", np.sum([len(win_gaze_portion[case]) for case in cases])) # 272/300

        # Build the gaze portion plot                      
        x_pos = np.arange(len(win_start_t))
        colors = get_colors() #mcolors.TABLEAU_COLORS#[:4] #['#54494B',  '#833951', '#B22400', '#006BB2'] #['r','g', 'b', 'y']

        fig, ax = plt.subplots()            
            
        for case, color in zip(cases, colors):
            temp = np.array(win_gaze_portion[case], dtype=object)[:,:,1]
            avg_gaze_portion = [np.nanmean(np.hstack(temp[:,i])) if np.any(~np.isnan(np.hstack(temp[:,i]))) else np.nan for i in range(temp.shape[1])]
            plt.plot(win_start_t[:-1], avg_gaze_portion, color=color, label=case, linewidth=3.5) 

        plt.xlabel("Relative time to initial speech (ms)")
        plt.ylabel("Target gaze portion")
        # plt.title("Relationship between the target gaze portion and the speech starting time")
        plt.legend(loc='lower right')

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('target_gaze_portion.png')
        # plt.show()               

    ### Graph3: NASA-TLX
    elif graph_name =="nasa_tlx":            
        with open('NASA TLX and Interview Questions.csv', mode='r') as csvfile:
            reader = csv.reader(csvfile)
            tlx_data = []
            for row in reader:
                # print(row[2:-3])
                tlx_data.append(row[2:-3])        
        
        ## To use Seaborn    
        tlx_data = np.array(tlx_data[1:],dtype=int).flatten()    
        evals =  ['Mental \n Demand', 'Physical \n Demand', 'Temporal \n Demand', 'Performance', 'Effort', 'Frustration']
        metrics = sum(([[met]*4 for met in evals]), []) 
        cases = ['Case 1', 'Case 2', 'Case 3', 'Case 4'] * 6      
        scores = tlx_data.T.flatten().tolist() 
        sns_data = np.stack([metrics*5, cases*5, scores]).T       
        head = ['metrics', 'cases', 'scores']        
        tlx_data = pd.DataFrame(sns_data, columns=head)
        tlx_data['scores'] = tlx_data.scores.astype(float)   
        
        
        g = sns.catplot(
            data=tlx_data, kind="bar",
            x="metrics", y="scores", hue="cases",
            palette=sns.color_palette(get_colors()), height=8, ci=None #"sd"  # palette=sns.color_palette(get_colors()) #alpha=.6,
        )
        
        # pattern=[None, '////']

        # hatches=np.repeat(pattern,12)
        # # colors = np.repeat([[0.4, 0.4, 0.4],  [0.984375, 0.7265625, 0]],12)#get_colors()
        # kinds = np.repeat(["bar", "box"], 12)
        
        # for bar, h in zip(g.axes.patches,hatches):
        #     bar.set_hatch(h)
        
        # for ax in g.axes.flat:
        #     patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
        #     for pat,c, kind, bar in zip(hatches,colors, kinds, patches):
        #         bar.set_hatch(pat)
        #         # bar.set_edgecolor(c)                
        #         # bar.set_facecolor('none')
        #         # bar.set_kind(kind)
            
        # g.despine(left=True, right)
        g.set_axis_labels("", "Scale (1-7)")
        g._legend.set_title("")
        
        
    
        g.set(title="NASA-TLX Scores")
        fig = g.fig
        fig.savefig('nasa_tlx_sns.png',dpi=300,bbox_inches="tight")  
        
        
        ## Not using Seaborn
        
        # tlx_data = np.array(tlx_data[1:],dtype=int) # (5, 24)
        # tlx_mean = np.mean(tlx_data, axis=0).reshape((6,4))       
        
        # x_pos = np.arange(tlx_mean.shape[0])
        # fig, ax = plt.subplots()        
        # n=6
        # r = np.arange(n)
        # width = 0.2
        # colors = ['y','g', 'b', 'r']
        
        # for i in range(tlx_mean.shape[1]):      
        #     tlx_scale = [tlx_mean[c, i] for c in range(tlx_mean.shape[0])]

        #     plt.bar(x_pos+width*i, tlx_scale, color = colors[i],
        #             width = width, edgecolor = colors[i],
        #             label=cases[i])                            

        # plt.ylabel("Scale (1-7)")
        # plt.title("NASA-TLX Scores")
        
        # plt.xticks(r + width/2,['Mental \n Demand', 'Physical \n Demand', 'Temporal \n Demand', 'Performance', 'Effort', 'Frustration'])
        # plt.legend(loc = "upper right")

        # # Save the figure and show
        # plt.tight_layout()
        # plt.savefig('nasa_tlx.png')
        # plt.show()               
                                
set_style()
graph_name = "pred_acc" #"pred_acc" # "target_gaze_portion" # "nasa_tlx"
plot_graph(graph_name)

