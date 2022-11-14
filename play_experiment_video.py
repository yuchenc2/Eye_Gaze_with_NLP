import cv2
import os
import time

item_amounts = [3, 6]
item_configs = ["Duplicated Object", "Duplicated Shape"]
item_classes = ["Apples", "Bowls", "Cups", "Books", "Bowls", "Scissors"]
robot_reaction_times = [1, 3, 5]

def choose_scene(config_args):
    for i in range(2):
        for j in range(2):
            for k in range(3):
                if len(config_args) < 12:
                    config_args.append([i * 6 + j * 3 + k + 1, item_amounts[i], item_classes[j * 3 + k], item_configs[j]])
                print("Type {} for scene with {} items and {} in a {} configuration".format(*config_args[i * 6 + j * 3 + k]))
    return input(), config_args

def play_video(file_path):
    video_capture = cv2.VideoCapture(file_path)
    print("\n", file_path, "\n")
    if not video_capture.isOpened():
        print("Error opening video file")
        return -1
    while(video_capture.isOpened()):
        valid, frame = video_capture.read()
        if valid:
            cv2.imshow("Frame", frame)
            cv2.waitKey(25)
        else:
            break
    video_capture.release()
    cv2.destroyAllWindows()
    return 1

if __name__ == "__main__":
    scene_choice = -1
    config_args = []
    while scene_choice < 0 or scene_choice > 11:
        scene_input, config_args = choose_scene(config_args)
        try:
            scene_choice = int(scene_input) - 1
        except:
            scene_choice = -1
            print("Invalid choice. Choose again. ")
    print("\n", config_args[scene_choice], "\n")
    item_config_name = "ds"
    if (scene_choice // 3) % 2 == 0:
        item_config_name = "do"
    folder_name = "{}_{}_{}".format(config_args[scene_choice][1], item_config_name, config_args[scene_choice][2].lower())
    # Make the robot reaction time index random?
    robot_reaction_index = 0
    trial = -1
    while trial < 1 or trial > 3:
        try:
            trial = int(input("Choose a trial (1-3): "))
        except:
            trial = -1
            print("Invalid choice. Choose again. ")
    trial_folder = "trial_{}".format(trial)
    initial_scene = cv2.imread(os.path.join(folder_name, trial_folder, folder_name + "_" + trial_folder + "_scene.jpg"))
    cv2.imshow("Frame", initial_scene)
    pressed_key = cv2.waitKey(0)
    time.sleep(robot_reaction_times[robot_reaction_index])
    if pressed_key == ord("z"):
        play_video(os.path.join(folder_name, trial_folder, folder_name + "_" + trial_folder + "_left.mp4"))
    elif pressed_key == ord("x"):
        play_video(os.path.join(folder_name, trial_folder, folder_name + "_" + trial_folder + "_right.mp4"))

