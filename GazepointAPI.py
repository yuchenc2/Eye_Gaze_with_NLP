######################################################################################
# GazepointAPI.py - Example Client
# Written in 2013 by Gazepoint www.gazept.com
#
# To the extent possible under law, the author(s) have dedicated all copyright 
# and related and neighboring rights to this software to the public domain worldwide. 
# This software is distributed without any warranty.
#
# You should have received a copy of the CC0 Public Domain Dedication along with this 
# software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
######################################################################################

import socket
import re

# Host machine IP
HOST = '192.168.0.22'
# Gazepoint Port
PORT = 4242
ADDRESS = (HOST, PORT)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(ADDRESS)

# s.send(str.encode('<SET ID="ENABLE_SEND_CURSOR" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_TIME" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))

REC_Time = []
FPOGX = []
FPOGY = []
FPOGS = []
FPOGD = []
FPOGV = []

time = 0

while time < 30:
    rxdat = s.recv(1024)    
    eye_raw = bytes.decode(rxdat)
    extracted_floats = [float(s) for s in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", eye_raw)]
    # <REC TIME="39.56323" FPOGX="0.00000" FPOGY="0.00000" FPOGS="38.72517" FPOGD="0.08258" FPOGID="30" FPOGV="0" />
    if len(extracted_floats) == 7 and int(extracted_floats[6]) == 1: #If eye gaze is a fixation (filter out none fixation points)
        REC_Time.append(extracted_floats[0])
        FPOGX.append(extracted_floats[1])
        FPOGY.append(extracted_floats[2])
        FPOGS.append(extracted_floats[3])
        FPOGD.append(extracted_floats[4])
        time = extracted_floats[0]

    

s.close()