""" 
    AI thermometer: Code for automatically measuring the temperature of people using a thermal camera.
   
    IIT : Istituto italiano di tecnologia
    Pattern Analysis and Computer Vision (PAVIS) research line

    Description: The software first detect people with an off-the-shelf body pose detector and then extract location of the face where the
    temperature is measured. The software requires a known reference temperature and the value and position are provided by the user
    (this information is shown as a single small green circle on the image).
    It is possible to have the absolute temperature but you need an image from a thermal camera

    Usage Example:  
                    Without radiometric image, you need to specify reference temperature point value and position
                    python3 ai_thermometer.py --config_file ./config.ini

    that this as example code for that specific input, more work is necessary to extend it for other thermal cameras

    Disclaimer:
    The information and content provided by this application is for information purposes only. 
    You hereby agree that you shall not make any health or medical related decision based in whole 
    or in part on anything contained within the application without consulting your personal doctor.

    The software is provided "as is", without warranty of any kind, express or implied, 
    including but not limited to the warranties of merchantability, 
    fitness for a particular purpose and noninfringement. In no event shall the authors, 
    PAVIS or IIT be liable for any claim, damages or other liability, whether in an action of contract, 
    tort or otherwise, arising from, out of or in connection with the software 
    or the use or other dealings in the software.

    LICENSE:
    This project is licensed under the terms of the MIT license.
    This project incorporates material from the projects listed below (collectively, "Third Party Code").  
    This Third Party Code is licensed to you under their original license terms.  
    We reserves all other rights not expressly granted, whether by implication, estoppel or otherwise.

    The software can be freely used for any non-commercial applications and it is useful
    for the automatic early-screening of fever symptoms. The code is open and can be 
    improved with your support, please contact us at aithermometer@iit.it if you want to help us.
"""

import os
import PySpin
import sys
import time
import cv2
from sys import platform
import numpy as np
import socket
import threading
import queue
import signal
import datetime
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
import copy 
import argparse
import configparser
import json
import threading

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON`' \
            'in CMake and have this Python script in the right folder?')
        sys.exit(-1)
except Exception as e:
    print(e)
    sys.exit(-1)

# Release version
RELEASE = "0.5"

'''
    AiThermometer class, connect to remote thermal camera, get images, use openpose to find faces, send images to video servers
'''

class AiThermometer:
    '''
        Initialize parameters
    '''
    def __init__(self, args):
        # Create configurator
        config = configparser.ConfigParser()
        
        # Read configuration
        try:
            config.read(args[0].config_file)
        except:
            print("unable to find configuration file", flush=True)
            sys.exit(-1)

        # Print configuration
        print("Configuration:", flush=True)

        # Print Configuration
        for key in config:
            print ("[{0}]".format(key), flush=True)
            for argument in config[key]:  
                print("{0} = {1}".format(argument, config[key][argument]), flush=True)

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()

        # Openpose params

        # Model path
        params["model_folder"] = config['openpose']['models']

        # Face disabled 
        params["face"] = False

        # Hand disabled
        params["hand"] = False

        # Net Resolution
        params["net_resolution"] = config['openpose']['network']

        # Gpu number
        params["num_gpu"] = 1 # Set GPU number

        # Gpu Id
        params["num_gpu_start"] = 0 # Set GPU start id (not considering previous)

        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
        
        # Process Image
        self.datum = op.Datum()

        # Continue recordings until this param is True
        self.continue_recording = True
        
        # Listen thermal port
        self.thermal_port = int(config['network']['thermal_port'])

        # Listen openpose port
        self.openpose_port = int(config['network']['openpose_port'])

        # Listen openpose port
        self.js_port = int(config['network']['js_port'])

        # Listen openpose port
        self.image_port = int(config['network']['image_port'])

        # Create a queue list to store client queues (thermal images)
        self.thermal_list = []

        # Create a queue list to store client queues (openpose images)
        self.openpose_list = []

        # Create a queue list to store client queues (json frames)
        self.js_list = []

        # Minimal temperature (image reconstuction)
        self.min_temperature = float(config['thermal']['min_temperature'])

        # Maaximum temperature (image reconstruction)
        self.max_temperature = float(config['thermal']['max_temperature'])

        # Set camera id (multiple camera available)
        self.id = int(config['thermal']['id'])

        # Set face min size x
        self.min_sizex = int(config['face']['min_sizex'])

        # Set face min size y
        self.min_sizey = int(config['face']['min_sizey'])

        # Set font size
        self.font_scale = float(config['face']['font_scale'])

        # Set alarm temperature
        self.alarm_temperature = float(config['face']['alarm_temperature'])

        # Camera resolution x
        self.resolution_x = int(config['thermal']['resolution_x']) 

        # Camera resolution y
        self.resolution_y = int(config['thermal']['resolution_y'])

        # Reflected temperature
        self.reflected_temperature = float(config['thermal']['reflected_temperature'])

        # Atmosferic temperature
        self.atmospheric_temperature = float(config['thermal']['atmospheric_temperature'])

        # Object distance
        self.object_distance = float(config['thermal']['object_distance'])

        # Object emissivity
        self.object_emissivity = float(config['thermal']['object_emissivity'])

        # Relative humidity
        self.relative_humidity = float(config['thermal']['relative_humidity'])

        # ext_optics_temperature
        self.extoptics_temperature = float(config['thermal']['extoptics_temperature']) 
        
        # ext_optics_transmission
        self.extoptics_transmission = float(config['thermal']['extoptics_transmission']) 

        # ext_optics_transmission
        self.estimated_transmission = float(config['thermal']['estimated_transmission']) 

        # Lines to be removed to correct a camera error on retrived image
        self.unused_lines = int(config['thermal']['unused_lines'])

        # Set compression
        self.compression = int(config['mjpeg']['compression'])

        # Show video
        self.show_video = True if int(config['debug']['show_video'])==1 else False 

        # Min detected temperature
        self.min_detection_temperature = int(config['face']['min_detection_temperature']) 

        # Max detected temperature
        self.max_detection_temperature = int(config['face']['max_detection_temperature']) 

        # Min detected temperature
        self.delta_temperature = float(config['face']['delta_temperature']) 

        # Record secquence
        self.record_image = True if int(config['debug']['record_image'])==1 else False 

        # Record dir
        self.record_dir = config['debug']['record_dir']

        # Record csv
        self.record_csv = True if int(config['debug']['record_csv'])==1 else False

        # Record csv filename
        self.filename_csv = config['debug']['filename_csv']

        # Record csv
        self.debug = True if int(config['debug']['debug'])==1 else False

        # Record raw
        self.recorder_raw = True if int(config['debug']['recorder_raw'])==1 else False

        # Player raw
        self.player_raw = True if int(config['debug']['player_raw'])==1 else False

        # Record raw filename
        self.filename_raw = config['debug']['filename_raw']

        # DEBUG
        try:
            # Open recorder file
            if self.recorder_raw:
                self.raw = open(config['debug']['filename_raw'],"wb")

            # Open player file
            if self.player_raw:
                self.raw = open(config['debug']['filename_raw'],"rb")
        except:
            print("Unable to open {0} local file!".format(config['debug']['filename_raw']), flush=True)
            os._exit(-1)

        try:
            # Create target Directory
            os.mkdir(self.record_dir)
            print("Directory " , self.record_dir ,  " Created ", flush=True) 
        except FileExistsError:
            print("Directory " , self.record_dir ,  " already exists", flush=True)
        
        # Initialize thermal server
        self.thermal_server = StreamServer(self.thermal_port, self.thermal_list, "image/jpeg")

        # Initialize openpose server
        self.openpose_server = StreamServer(self.openpose_port, self.openpose_list, "image/jpeg")
        
        # Initialize json server
        self.js_server = ResponseServer(self.js_port, "application/json")

        # Initialize image server
        self.image_server = ResponseServer(self.image_port, "image/jpeg")
        
        # Initialize temperature FIFO length and array
        self.max_t_fifo = []
        self.fifo_size = 15

        self.max_t_face_fifo = []
        self.fifo_face_size = 15

        # Initializing the mask
        self.mask = cv2.imread(config['thermal']['mask_filename'], 1)
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        self.mask = self.mask > 100

        # Create jpeg object
        self.jpeg = TurboJPEG()

    '''
        Connect to thermal camera gigE
    '''
    def connect(self):
        # DEBUG: select file if debug and player are selected
        if self.player_raw:
            print("Read images from file, skip camera connect")
            return True

        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()

        # Get current library version
        version = self.system.GetLibraryVersion()
        print('Spinnaker Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))
        
        # Retrieve list of cameras from the system
        cam_list = self.system.GetCameras()

        # Get camera number
        num_cameras = cam_list.GetSize()

        # Print detected camera number
        print('Number of cameras detected: %d' % num_cameras)

        # Finish if there are no cameras
        if num_cameras == 0:

            # Clear camera list before releasing system
            cam_list.Clear()

            # Release system instance
            self.system.ReleaseInstance()

            print('Not enough cameras!')

            return False

        # Use first camera (we use one camere)
        self.camera = cam_list[self.id]

        # Clear camera list before releasing system
        cam_list.Clear()

        return True

    '''
        Acquire data from remote thermal camera
    '''
    def acquire_process(self):
        if self.player_raw:
            return self.player()

        return self.run_camera()

    '''
        Disconnect from camera and close all
    '''
    def disconnect(self):
        try:
            # Stop data recording
            self.continue_recording = False

            time.sleep(1)

            # Thermal server
            self.thermal_server.disconnect()

            # Openpose server
            self.openpose_server.disconnect()

            # js server
            self.js_server.disconnect()

            # image server
            self.image_server.disconnect()

            # DEBUG: reading raw
            if self.player_raw:
                return
            
            # Stopping acquisition
            self.camera.EndAcquisition()

            # Wait some time to stop recording
            time.sleep(5)

            # Deinitialize camera
            self.camera.DeInit()

            # Wait some time
            time.sleep(1)
            
            # Delete camera
            del self.camera
            
            # Release system instance
            self.system.ReleaseInstance()
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)

    '''
        Get images from camera, analyze it with openpose and evaluate temperature into a box over face 
    '''
    def acquire_images(self, cam, nodemap, nodemap_tldevice):

        sNodemap = cam.GetTLStreamNodeMap()

        # Change bufferhandling mode to NewestOnly
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve integer value from entry node
        node_newestonly_mode = node_newestonly.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

        print('*** IMAGE ACQUISITION ***\n')
        
        try:
            # Get acquisition mode
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Acquisition mode set to continuous...')

            #  Begin acquiring images
            cam.BeginAcquisition()

            print('Acquiring images...')

            #  Retrieve device serial number for filename
            device_serial_number = ''
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            
            if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()
                print('Device serial number retrieved as %s...' % device_serial_number)

            # Retrieve and display images
            while(self.continue_recording):
                #  Retrieve next received image
                image_result = cam.GetNextImage()

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                    image_result.Release()
                    continue

                # DEBUG: record image on raw sequence
                if self.recorder_raw:
                    self.rec_image(image_result)

                # Analyze image
                self.analyze_image(image_result)

                #  Release image
                image_result.Release()

            #  End acquisition
            cam.EndAcquisition()
            
            # DEBUG: Close csv file
            if self.record_csv:
                self.csv.flush()
                self.csv.close()

            # DEBUG: Close raw file
            if self.recorder_raw:
                self.raw.flush()
                self.raw.close()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex, flush=True)
            return False

        return True
    
    '''
        Configure selected camera
    '''
    def run_camera(self):
        try:
            nodemap_tldevice = self.camera.GetTLDeviceNodeMap()

            # Initialize camera
            self.camera.Init()

            # Retrieve GenICam nodemap
            nodemap = self.camera.GetNodeMap()

            # Retrive IRFormat node
            node_irformat_mode = PySpin.CEnumerationPtr(nodemap.GetNode("IRFormat"))
            
            # Check if param is available and writable
            if PySpin.IsAvailable(node_irformat_mode) and PySpin.IsWritable(node_irformat_mode):
                # Turn to IRRadiation Temperature linear, 0.01K resolution
                node_irformat_mode.SetIntValue(2)

                # Read value from IRFormat node
                print ("IRFormat:{0}".format(node_irformat_mode.GetIntValue()))
                
            time.sleep(0.1)
            # Retrive Width
            node_width = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
            # Check if param is available and writable
            if PySpin.IsAvailable(node_width) and PySpin.IsWritable(node_width):
                # Width
                node_width.SetValue(self.resolution_x)

                # Read value from IRFormat node
                print ("Image width:{0}".format(node_width.GetValue()))

            time.sleep(0.1)
            # Retrive Height node
            node_height = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
            # Check if param is available and writable
            if PySpin.IsAvailable(node_height) and PySpin.IsWritable(node_height):
                # Set Height
                node_height.SetValue(self.resolution_y) # 246 not good temperature because 6 black lines, but 240 generate incomplete images

                # Read value from IRFormat node
                print ("Image height:{0}".format(node_height.GetValue()))

            time.sleep(0.1)
            # Retrive PixelFormat node
            node_pixelformat = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
            # Check if param is available and writable
            if PySpin.IsAvailable(node_pixelformat) and PySpin.IsWritable(node_pixelformat):
                # Set Mono16
                node_pixelformat.SetIntValue(node_pixelformat.GetEntryByName("Mono16").GetValue())

                # Print pixel format
                print ("PixelFormat:{0}".format(node_pixelformat.GetIntValue()), flush=True)

            time.sleep(0.1)
            # Retrive Reflected Temperature node
            node_reflected_temperature = PySpin.CFloatPtr(nodemap.GetNode("ReflectedTemperature"))
             # Check if param is available and writable
            if PySpin.IsAvailable(node_reflected_temperature) and PySpin.IsWritable(node_reflected_temperature):
                # Set Value
                node_reflected_temperature.SetValue(self.reflected_temperature)

                # Print Reflected Temperature
                print ("ReflectedTemperature:{0}".format(node_reflected_temperature.GetValue()), flush=True)

            # Retrive Atmospheric Temperature node
            node_atmospheric_temperature = PySpin.CFloatPtr(nodemap.GetNode("AtmosphericTemperature"))
            # Check if param is available and writable
            if PySpin.IsAvailable(node_atmospheric_temperature) and PySpin.IsWritable(node_atmospheric_temperature):
                # Set Value
                node_atmospheric_temperature.SetValue(self.atmospheric_temperature)

                # Print Atmospheric Temperature
                print ("AtmosphericTemperature:{0}".format(node_atmospheric_temperature.GetValue()), flush=True)

            time.sleep(0.1)
            # Retrive Object Emissivity node
            node_object_emissivity = PySpin.CFloatPtr(nodemap.GetNode("ObjectEmissivity"))
            # Check if param is available and writable
            if PySpin.IsAvailable(node_object_emissivity) and PySpin.IsWritable(node_object_emissivity):
                # Set Value
                node_object_emissivity.SetValue(self.object_emissivity)

                # Print Object Emissivity
                print ("ObjectEmissivity:{0}".format(node_object_emissivity.GetValue()), flush=True)

            time.sleep(0.1)
            # Retrive and Change Object Emissivity node
            node_relative_humidity = PySpin.CFloatPtr(nodemap.GetNode("RelativeHumidity"))
            # Check if param is available and writable
            if PySpin.IsAvailable(node_relative_humidity) and PySpin.IsWritable(node_relative_humidity):
                # Set Value
                node_relative_humidity.SetValue(self.relative_humidity)

                # Print Object Emissivity
                print ("Changed RelativeHumidity To {0}".format(node_relative_humidity.GetValue()), flush=True)

            time.sleep(0.1)
            # Retrive and Change ExtOpticsTemperature
            node_extoptics_temperature = PySpin.CFloatPtr(nodemap.GetNode("ExtOpticsTemperature"))
            
            # Check if param is available and writable
            if PySpin.IsAvailable(node_extoptics_temperature) and PySpin.IsWritable(node_extoptics_temperature):
                # Set Value
                node_extoptics_temperature.SetValue(self.extoptics_temperature)

                # Print Object Emissivity
                print ("Changed ExtOpticsTemperature To {0}".format(node_extoptics_temperature.GetValue()), flush=True)

            time.sleep(0.1)
            # Retrive and Change ExtOpticsTransmission
            node_extoptics_transmission = PySpin.CFloatPtr(nodemap.GetNode("ExtOpticsTransmission"))

            # Check if param is available and writable
            if PySpin.IsAvailable(node_extoptics_transmission) and PySpin.IsWritable(node_extoptics_transmission):
           
                # Set Value
                node_extoptics_transmission.SetValue(self.extoptics_transmission)

                # Print Object Emissivity
                print ("ExtOpticsTransmission:{0}".format(node_extoptics_transmission.GetValue()), flush=True)

            time.sleep(0.1)
            # Retrive and change Estimated Transmission 
            node_estimated_transmission = PySpin.CFloatPtr(nodemap.GetNode("EstimatedTransmission"))
            if PySpin.IsAvailable(node_estimated_transmission) and PySpin.IsWritable(node_estimated_transmission):
           
                # Set Value
                node_estimated_transmission.SetValue(self.estimated_transmission)

                # Print Object Emissivity
                print ("EstimatedTransmission:{0}".format(node_estimated_transmission.GetValue()), flush=True)

            time.sleep(1)

            # Start video servers
            self.thermal_server.activate()

            self.openpose_server.activate()

            # Start json server
            self.js_server.activate()

            # Start image server
            self.image_server.activate()

            # Acquire images
            self.acquire_images(self.camera, nodemap, nodemap_tldevice)

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)

    '''
        Helper function for recovering temperature from raw pixel value
        Get temperature https://graftek.biz/system/files/15690/original/FLIR_Genicam.pdf?1571772310
    '''
    def get_temperature(self, pixel_value):
        temperature = pixel_value * 0.01 - 273.15
        return temperature

    '''
        Helper function for updating a FIFO
    '''
    def update_fifo(self, fifo, fifo_size, new_value):
        if (len(fifo) > fifo_size):
            fifo.pop(0)

        fifo.append(new_value)
        return fifo

    '''
        Helper function for collecting average FIFO value
    '''
    def get_fifo_avg(self, fifo, min_len_to_compute=0):
        if (len(fifo) > min_len_to_compute):
            return np.average(fifo)
        else:
            return 0

    '''
        Analyze images (and send over network queues)
    '''
    def analyze_image(self, image_result):

        # Get image dimensions
        width = image_result.GetWidth()
        height = image_result.GetHeight()

        # Getting the image data as a numpy array
        image_data = image_result.GetNDArray()

        ## MBMB ->
        # Updating the FIFO
        min_image_temperature = self.get_temperature(np.amin(image_data)) + self.delta_temperature

        max_image_temperature = self.get_temperature(np.amax(image_data * self.mask)) + self.delta_temperature

        if max_image_temperature > 25:
            self.max_t_fifo = self.update_fifo(self.max_t_fifo, self.fifo_size, max_image_temperature)
        else:
            self.max_t_fifo = []

        temp_smooth = self.get_fifo_avg(self.max_t_fifo, self.fifo_size)
        ## <- MBMB

        '''
            Calculate image to send via mjpeg to remote client and openpose compliant image
        '''
        # Convert image to BGR, using threshold temperatures (manual parameters)
        in_img = image_result.GetData().reshape( (height, width) )
        
        temp_max_thr= self.max_temperature # Max temperature
        temp_min_thr= self.min_temperature # Min temperature

        # Calculate thresholds
        pixel_max_thr = int((temp_max_thr + 273.15)/0.01)
        pixel_min_thr = int((temp_min_thr + 273.15)/0.01)

        # Threshold image
        in_img_rw = copy.deepcopy(in_img)
        in_img_rw[in_img_rw>pixel_max_thr]=pixel_max_thr
        in_img_rw[in_img_rw<pixel_min_thr]=pixel_min_thr
        in_img_rw[0,0] = pixel_max_thr
        in_img_rw[0,1] = pixel_min_thr

        # Normalize image
        raw_frame = cv2.normalize(in_img_rw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Get correct image
        raw_frame = raw_frame[0:self.resolution_x][0:self.resolution_y-self.unused_lines]

        # Invert levels
        gray_inverted = cv2.bitwise_not(raw_frame)

        # Convert inverted grayscale to Color RGB format (openpose input)
        image_openpose = cv2.cvtColor(gray_inverted, cv2.COLOR_GRAY2BGR)

        # Colorize Image (Use it to write geometry and send to streaming)
        to_send_image = cv2.applyColorMap(raw_frame , cv2.COLORMAP_JET)

        ## MBMB ->
        # DEBUG: Plotting the location of the max temperature
        if self.debug:
            # Trace circles on minimal and maximum temperature
            coords_max = np.unravel_index(np.argmax(image_data*self.mask, axis=None), image_data.shape)
            cv2.circle(to_send_image, (coords_max[1], coords_max[0]), radius=10, color=(255, 255, 255), thickness=2)

            coords_min = np.unravel_index(np.argmin(image_data, axis=None), image_data.shape)
            cv2.circle(to_send_image, (coords_min[1], coords_min[0]), radius=10, color=(0, 0, 0), thickness=2)

            # Print Temperatures
            if temp_smooth > 0:
                text_str = 'Max T: {:.2f}C - Min T: {:.2f}C - Smooth: {:.2f}C'.format(
                    self.get_temperature(np.amax(image_data * self.mask)),
                    self.get_temperature(np.amin(image_data)), temp_smooth)
            else:
                text_str = 'Max T: {:.2f}C - Min T: {:.2f}C'.format(
                    self.get_temperature(np.amax(image_data * self.mask)),
                    self.get_temperature(np.amin(image_data)))

            font_temperature = cv2.FONT_HERSHEY_DUPLEX
            font_scale = self.font_scale
            font_thickness = 1
            color = (255,255,255)

            text_w, text_h = cv2.getTextSize(text_str, font_temperature, font_scale, font_thickness)[0]

            px = int(5)
            py = int(5)

            # Draw text rectangle
            cv2.rectangle(to_send_image, (px-5, py-5), (px + text_w + 5, py + text_h + 5), color, -1)

            # Draw Text
            cv2.putText(to_send_image, text_str, (px, py + text_h), font_temperature,
                        font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        ## <- MBMB

        # Set image to openpose video
        self.datum.cvInputData = image_openpose

        self.opWrapper.emplaceAndPop([self.datum])

        # Get openpose output (convert to int all value)
        bodys = np.array(self.datum.poseKeypoints).astype(int).tolist()

        # record only one thermal shot
        one_snapshot = True

        # Open csv
        if self.record_csv:
            self.csv = open(self.filename_csv, "a")

        #face geometry and temperature container
        js_packet = {}

        # Json dataset
        body_packet = []

        # If a body is recognized
        if type(bodys) is list:
            # Remove probability from joints
            temporary_bodys = []
            for body in bodys:
                temporary_bodys.append([reduced[0:2] for reduced in body])
            
            bodys = temporary_bodys

            for body in bodys:
                # Face points (0, 15, 16) refered to body_25 openpose format
                face = [[int(body[0][0]),int(body[0][1])], # Nose
                    [int(body[15][0]),int(body[15][1])], # Right eye
                    [int(body[16][0]),int(body[16][1])], # Left eye
                    [int(body[17][0]),int(body[17][1])], # Right ear
                    [int(body[18][0]),int(body[18][1])]] # Left ear

                # Select the best face size 
                if 0 not in face[0] and 0 not in face[1] and 0 not in face[2]:
                    # Get line values from eyes line to neck
                    if (0 not in face[4]) and (0 not in face[3]): # Both ears visibles
                        size_x = int(abs(face[3][0]-face[4][0])/2)
                        size_y = int(abs(face[3][0]-face[4][0])/2)
                    elif (0 in face[4]) and (0 not in face[3]): # Right Ear, no Left Ear 
                        size_x = int(abs(face[3][0]-face[2][0])/2)
                        size_y = int(abs(face[3][0]-face[2][0])/2)
                    elif (0 not in face[4]) and (0 in face[3]): # Left and Right ears ok
                        size_x = int(abs(face[1][0]-face[4][0])/2)
                        size_y = int(abs(face[1][0]-face[4][0])/2)
                    else: # Left and Right ears are not available
                        size_x = int(abs(face[1][0]-face[2][0])/2)
                        size_y = int(abs(face[1][0]-face[2][0])/2)

                    # Set min face size x
                    min_sx = self.min_sizex

                    # Set min face size y
                    min_sy = self.min_sizey

                    # If face is to smal select minimal size
                    size_x = size_x if size_x > min_sx else min_sx
                    size_y = size_y if size_y > min_sy else min_sy

                    # Set face center
                    reference_x = face[0][0]
                    reference_y = face[0][1]

                    offset_x = 0
                    offset_y = 0

                    # Calculate average values in face rect 
                    counter = 0
                    average = 0
                    max_temperature = 0.0
                    for y in range(reference_x-size_x+offset_x, reference_x+size_x+offset_x):
                        for x in range(reference_y-size_y+offset_y, reference_y+size_y+offset_y):
                            # Get temperature https://graftek.biz/system/files/15690/original/FLIR_Genicam.pdf?1571772310
                            if x<self.resolution_y and y<self.resolution_x:
                                temperature = self.get_temperature(image_data[x][y])

                                # Find max temperature
                                if temperature > max_temperature:
                                    max_temperature = temperature

                                average += temperature
                                counter += 1
                    
                    # Calculate average
                    if counter!=0:
                        temperature= average / counter

                    # Compensate uncalibrated temperature 
                    temperature += self.delta_temperature
                    max_temperature += self.delta_temperature

                    # Filter too low temperature face detection error
                    if temperature < self.min_detection_temperature:
                        continue

                    # Filter too hig temperature face detection error
                    if temperature > self.max_detection_temperature:
                        continue
                        
                    # json alarm flag
                    alarm = 0

                    # Alarm temperature show red color rectangle   
                    if max_temperature>self.alarm_temperature:
                        color = (0,0,255)
                        alarm = 1
                    else:
                        color = (255,0,0)
                        alarm = 0

                    # Draw face Rectangle
                    cv2.rectangle(to_send_image, 
                        (reference_x-size_x+offset_x, reference_y-size_y+offset_y), 
                        (reference_x+size_x+offset_x, reference_y+size_y+offset_y), 
                        color, 5)

                    # Write temperature
                    ## MBMB

                    text_str = '{0:.2f}C'.format(max_temperature)
                    font_temperature = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = self.font_scale
                    font_thickness = 1
                    
                    text_w, text_h = cv2.getTextSize(text_str, font_temperature, font_scale, font_thickness)[0]

                    px = int(reference_x)
                    py = int(reference_y + size_y/2)
                    
                    # Draw text rectangle
                    cv2.rectangle(to_send_image, (px, py), (px + text_w, py - text_h), color, -1)

                    # Draw Text
                    cv2.putText(to_send_image, text_str, (px, py), font_temperature,
                        font_scale, (255,255,255), font_thickness, cv2.LINE_AA)

                    # Get right eye temperature form thermal image
                    righteye_temperature = image_data[face[1][1]][face[1][0]] * 0.01 - 273.15 + self.delta_temperature

                    cv2.circle(to_send_image, (face[1][0],face[1][1]), 2, color, 2)

                    cv2.putText(to_send_image, "{0:.2f}".format(righteye_temperature), (face[1][0], face[1][1]), font_temperature,
                        font_scale/2, (255,255,255), font_thickness, cv2.LINE_AA)

                    # Get left eye temperature form thermal image
                    lefteye_temperature = image_data[face[2][1]][face[2][0]] * 0.01 - 273.15 + self.delta_temperature

                    cv2.circle(to_send_image, (face[2][0], face[2][1]), 2, color, 2)

                    cv2.putText(to_send_image, "{0:.2f}".format(lefteye_temperature), (face[2][0], face[2][1]), font_temperature,
                        font_scale/2, (255,255,255), font_thickness, cv2.LINE_AA)

                    # Print data 
                    ts = int(round(time.time() * 1000))

                    dt_string = "{0:.2f},{1:.2f},{2:.2f},{3:.2f},{4:.2f},{5:.2f},{6:.2f},{7}\n".format(
                        temperature, 
                        max_temperature,
                        min_image_temperature,
                        max_image_temperature,
                        lefteye_temperature,
                        righteye_temperature,
                        temp_smooth,
                        ts)
                    
                    print(dt_string, end="", flush=True)

                    if self.record_image and one_snapshot:
                        f = open(self.record_dir+"/"+str(ts)+".raw", "wb")
                        f.write(image_result.GetData())
                        f.close()
                        one_snapshot = False

                    if self.record_csv:
                        self.csv.write(dt_string)
                        self.csv.flush()

                    body_packet.append([body, reference_x, reference_y,
                        size_x, size_y, "{0:.2f}".format(max_temperature), alarm])

        # Store face geometry
        js_packet["geometries"] = body_packet

        if self.show_video:
            # Show openpose
            cv2.imshow("Openpose output", self.datum.cvOutputData)

            # Show mjpeg output
            cv2.imshow("Mjpeg colorized", to_send_image)

            # Handle signals and wait some time
            cv2.waitKey(1)

        # Get timestamp
        ts = int(round(time.time() * 1000))

        # Store timestamp
        js_packet["ts"] = ts

        # Put thermal image into queue for each server thread
        self.send_image(self.thermal_list, to_send_image, ts)

        # Put openpose image into queue for each server thread
        self.send_image(self.openpose_list, self.datum.cvOutputData, ts)

        # Put json into instant locked memory
        self.js_server.put(bytes(json.dumps(js_packet), "UTF-8"))

        # Put image into instant locked memory
        self.image_server.put(self.jpeg.encode(to_send_image, quality=self.compression))

    '''
        Send image over queue list and then over http mjpeg stream
    '''
    def send_image(self, queue_list, image, ts):
        
        encoded_image = self.jpeg.encode(image, quality=self.compression)
        # Put thermal image into queue for each server thread
        for q in queue_list:
            try:
                block = (ts, encoded_image)
                q.put(block, True, 0.02)
            except queue.Full:
                pass
    '''
        Send block over queue list and then over http mjpeg stream
    '''
    def send_jsdata(self, queue_list, js_data, ts):
        for q in queue_list:
            try:
                block = (ts, js_data)
                q.put(block, True, 0.02)
            except queue.Full:
                pass

    '''
        DEBUG: Record images on raw stream
    '''
    def rec_image(self, image_result):
        self.raw.write(image_result)

    '''
        DEBUG: Play images directly from raw file
    '''
    def player(self):
        # Start video servers
        self.thermal_server.activate()

        self.openpose_server.activate()

        # Start json server
        self.js_server.activate()

        # Start image server
        self.image_server.activate()

        time.sleep(1)

        while self.continue_recording:
            # Read image from file raw
            image_result = self.raw.read(self.resolution_x*self.resolution_y*2)
            
            # If file is eof, rewind
            if len(image_result) != self.resolution_x*self.resolution_y*2:
                print("Eof, Rewind!", flush=True)
                self.raw.seek(0)
                continue
            
            # Create Image
            image = PySpin.Image.Create(self.resolution_x, self.resolution_y, 0, 0,
                PySpin.PixelFormat_Mono16, np.array(image_result))

            # Analyze image
            self.analyze_image(image)

            # Simulate real acquisition
            time.sleep(0.02)

        self.raw.close()

'''
    VideoServer class, send images to remote clients
'''
class StreamServer:
    '''
        Initialize Video server with port and queue_list
    '''
    def __init__(self, port, queue_list, mt):
        self.port = port
        self.queue_list = queue_list
        self.run = True
        self.mt = mt

    '''
        Activate litening
    '''
    def activate(self):
        self.run = True
         # Start listen thread
        threading.Thread(target=self.listen).start()

    '''
        Stop listening
    '''
    def disconnect(self):
        self.s.shutdown(socket.SHUT_RDWR)
        self.s.close()
        self.run = False

    '''
        Listen thermal from network
    '''
    def listen(self):
        # Create server socket
        port = self.port
          
        # Configure server and reuse address
        self.s = socket.socket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind(('0.0.0.0', port))
        self.s.listen()

        # Wait for new connections
        while self.run:
            try:
                # Wait new connection
                c, addr = self.s.accept()
                
                # Print connection
                print("Connection from {0} to local port {1}".format(addr, self.port), flush=True)

                # Create new sending client queue
                q = queue.Queue(128)

                # Add queue to queue clients list
                self.queue_list.append(q)

                # Crete new server thread
                threading.Thread(target=self.client_handler, args=(c,q,)).start()
            except socket.error as e:
                print ("Error while listening :{0}".format(e), flush=True)

        print("Server on {0} listen stop".format(self.port), flush=True)

    '''
        Send images over network
    '''
    def client_handler(self, c, q):
    
        # Read request from remote web client
        data = c.recv(1024)

        # Decode data to eventually use it
        data = data.decode("UTF-8")

        # Print received data
        #print(data)

        # Create a fake header to send to remote client
        response = "HTTP/1.0 200 OK\r\n" \
                    "Cache-Control: no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0\r\n" \
                    "Pragma: no-cache\r\n" \
                    "Expires: Thu, 01 Dec 1994 16:00:00 GMT\r\n" \
                    "Connection: close\r\n" \
                    "Content-Type: multipart/x-mixed-replace; boundary=myboundary\r\n\r\n"

        # Print sending response
        #print(response)

        # Send header
        c.send(bytes(response, "UTF-8"))

        # While module run, send images to remote client
        while self.run:
            # Read image from queue
            try:
                block = q.get(True, 0.5)
            except queue.Empty:
                continue

            # Create image header to client response
            response = "--myboundary\r\n" \
                        "X-TimeStamp: " + str(block[0]) + "\r\n" \
                        "Content-Type: " + self.mt + "\r\n" \
                        "Content-Length: " + str(len(block[1])) + "\r\n\r\n"
                       

            # Print multipart response
            #print (response)
            
            # Try to send data until socket is valid
            try:
                c.send(bytes(response, "UTF-8"))
            except socket.error as e:
                print(e, flush=True)
                break

            # Try to send data until socket is valid
            try:
                c.send(block[1])
            except socket.error as e:
                print(e, flush=True)
                break
    
        #Remove Id from queue
        self.queue_list.remove(q)

        # Close connection
        c.close()

        print('Client handler closed')

'''
    VideoServer class, send images to remote clients
'''
class ResponseServer:
    '''
        Initialize Video server with port and queue_list
    '''
    def __init__(self, port, mt):
        self.port = port
        self.run = True
        self.mt = mt
        self.key_lock = threading.Lock()

    '''
        Activate litening
    '''
    def activate(self):
        self.run = True
         # Start listen thread
        threading.Thread(target=self.listen).start()

    '''
        Stop listening
    '''
    def disconnect(self):
        self.s.shutdown(socket.SHUT_RDWR)
        self.s.close()
        self.run = False
        self.block = None

    '''
        Listen thermal from network
    '''
    def listen(self):
        # Create server socket
        port = self.port
          
        # Configure server and reuse address
        self.s = socket.socket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind(('0.0.0.0', port))
        self.s.listen()
        self.block = None

        # Wait for new connections
        while self.run:
            try:
                # Wait new connection
                c, addr = self.s.accept()
                
                # Print connection
                #print("Connection from {0} to local port {1}".format(addr, self.port), flush=True)

                # Crete new server thread
                threading.Thread(target=self.client_handler, args=(c,)).start()
            except socket.error as e:
                print ("Error while listening :{0}".format(e), flush=True)

        print("thermal listen stop", flush=True)

    '''
        Send images over network
    '''
    def client_handler(self, c):
        if self.block is None:
            c.close()
            return

        # Read request from remote web client
        data = c.recv(1024)

        # Decode data to eventually use it
        data = data.decode("UTF-8")

        # Print received data
        #print(data)
        
        self.key_lock.acquire() 

        # Create a fake header to send to remote client
        response = "HTTP/1.0 200 OK\r\n" \
                    "Cache-Control: no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0\r\n" \
                    "Pragma: no-cache\r\n" \
                    "Expires: Thu, 01 Dec 1994 16:00:00 GMT\r\n" \
                    "Connection: close\r\n" \
                    "Content-Type: " + self.mt + "\r\n" \
                    "Content-Length: " + str(len(self.block)) + "\r\n\r\n"

        # Print sending response
        #print(response)

        # Try to send data until socket is valid
        try:
            c.send(bytes(response, "UTF-8"))
        except socket.error as e:
            print(e, flush=True)

        # Try to send data until socket is valid
        try:
            c.send(self.block)
        except socket.error as e:
            print(e, flush=True)

        self.key_lock.release() 

        c.close()

    def put(self, dt):
        self.key_lock.acquire() 
        self.block = dt
        self.key_lock.release() 
'''
    Runtime start here
'''
# Entry point
print("AI-Thermometer Release {0}: {1}".format(RELEASE,datetime.datetime.now()))

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default="./config.ini",
                        help="Selected path to the config")
                        
args = parser.parse_known_args()

# Create object 
ai_thermometer = AiThermometer(args)

# Handle close
def signal_handler(sig, frame):
    global ai_thermometer
    # Disconenct from Thermocamera 
    ai_thermometer.disconnect()
    time.sleep(1)
    os._exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def main():
    global ai_thermometer

    # Connect to Thermocamera
    ret_value = ai_thermometer.connect()
    
    if ret_value:
        # Acquire, recognize and send process
        ai_thermometer.acquire_process()

        # Return from thread
        return True
    else:
        return False

if __name__ == '__main__':
    main()

