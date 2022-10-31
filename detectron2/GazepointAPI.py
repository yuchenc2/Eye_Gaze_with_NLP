import socket
import re

class Gaze:
    def __init__(self):
        # Host machine IP
        self.HOST = '192.168.0.22'
        # Gazepoint Port
        self.PORT = 4242
        self.ADDRESS = (self.HOST, self.PORT)

        self.curr_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.curr_socket.connect(self.ADDRESS)

        # s.send(str.encode('<SET ID="ENABLE_SEND_CURSOR" STATE="1" />\r\n'))
        self.curr_socket.send(str.encode('<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n'))
        self.curr_socket.send(str.encode('<SET ID="ENABLE_SEND_TIME" STATE="1" />\r\n'))
        self.curr_socket.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
        
        self.REC_Time = []
        self.FPOGX = []
        self.FPOGY = []
        self.FPOGS = []
        self.FPOGD = []
        self.FPOGV = []
        self.fixation_counts = 0

        self.time = 0

    def eye_gaze_capture(self): # returns 1 if fixation detected
        # extract the values from the gaze output string
        # <REC TIME="39.56323" FPOGX="0.00000" FPOGY="0.00000" FPOGS="38.72517" FPOGD="0.08258" FPOGID="30" FPOGV="0" />
        rxdat = self.curr_socket.recv(1024)   
        eye_raw = bytes.decode(rxdat)
        extracted_floats = [float(rxdat) for rxdat in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", eye_raw)]  
        self.time = extracted_floats[0] 
        
        if len(extracted_floats) == 7 and int(extracted_floats[6]) == 1: #If eye gaze is a fixation (filter out none fixation points)
            self.REC_Time.append(extracted_floats[0]) #Time since system initialization
            self.FPOGX.append(extracted_floats[1]) #X coordinate from 0 to 1 as a fraction of the screen size
            self.FPOGY.append(extracted_floats[2]) #Y coordinate from 0 to 1 as a fraction of the screen size
            self.FPOGS.append(extracted_floats[3]) #Starting time of the fixation since system initialization
            self.FPOGD.append(extracted_floats[4]) #Duration of the fixation
            self.fixation_counts += 1
            return 1 
        else:
            return 0
            

            
    def close_socket(self):
        self.curr_socket.close()