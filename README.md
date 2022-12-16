# Eye_Gaze_with_NLP

## Description
In this work, we propose and evaluate our intention prediction algorithm for tele-manipulation through two in-depth human experiments. Our main contributions are highlighted as follows: 
* A novel algorithm that infers human’s intended grasping object in tele-operation using speech recognition and eye-gaze
* Two extensive human experiments to understand human psychophysics patterns and evaluate the performance of our approach

Please contanct Johnny Chang (yuchenc2@illinois.edu) and Nitish Gandi (gandi2@illinois.edu) for more details about this project.

## Getting Started

### Dependencies

* Linux 20.04
* Detectron2
* pytorch and torchvision(cpu-only) 
* OpenCV
* librosa, transformers, and pyaudio

### Installing

* Clone this project to your repository
  ```
  git clone https://github.com/yuchenc2/Eye_Gaze_with_NLP.git
  ```

* Install Detectron2 by following this tutorial (Make sure to install the cpu version of pytorch): https://detectron2.readthedocs.io/en/latest/tutorials/install.html

* Install the dependencies for the wav2vec2 NLP model: librosa, transformers, and pyaudio

* Add this to your ~/.bashrc file, replace {} with your username: 
  ```
  export PYTHONPATH=$PYTHONPATH:/home/{your_user_name}/Eye_Gaze_with_NLP/instance_seg/
  ```

* Run the download.py file to cache wav2vec2's model locally
  ```
  python3 download.py
  ```


### Executing program
* Run the Gazepoint program to capture the eye-gaze data on a windows machine. Make sure that the data is sent to the linux machine by setting the linux's IP address in the program's setting. Change the IP address in GazepointAPI.py to your windows machine's IP address.
* Run the main.py program 
  ```
  python3 main.py
  ```


## Acknowledgments
* [Deterctron2](https://github.com/facebookresearch/detectron2)
