
'''
    Brandon Wood
    FRC 334
    2/15/2018
    R18 network tables code. 

    Communicates with robot and execurse vision code. 
'''
import numpy as np
import cv2
from networktables import NetworkTables

from pc import PC
from tune import Tuner

from subprocess import call

class Networker:
     def __init__(self, server, table_name):
        # Initialize server
        NetworkTables.initialize(server_name)  # Should be a string
        
        # Connect to table and store
        self.TABLE = NetworkTables.getTable(table_name)        

        # Configure camera
        call(["~/cv/configure.sh"], shell=True)

    # Runtimes
    def pc_and_tape(self):
        # Execute vision code
        cap = cv.VideoCapture(0)
        
        # Initialize detectors
        pc = PC()
        
        while True:
            ret, frame = cap.read()
            
            last = 0

            c_names, c_values = pc.find_cubes_with_contours(frame)
            t_names, t_values = pc.find_cubes_with_contours(frame)
            self.send(t_names, t_values)

            # Keep old coordinates when cube is lost
            if c_values[0] == 0:
                self.send(c_names, [last])
            else:
                last = c_values[0]
                self.send(c_names, c_values)

            if cv2.WaitKey(1) ==  27:
                break
        cap.release()
        cv2.destroyAllWindows()

    # Utility functions
    def send(self, names, values):
        for name, value in zip(names, values):
            self.sd.putNumber(name, value)
