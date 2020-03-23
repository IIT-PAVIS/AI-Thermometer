#!/usr/bin/python3
# -*- coding: utf-8 -*-
 
""" 
    AI thermometer: Code for automatically measuring the temperature of people using a thermal camera.
   
    IIT : Istituto italiano di tecnologia
    Pattern Analysis and Computer Vision (PAVIS) research line

    Description: The software first detect people with an off-the-shelf body pose detector and then extract location of the face where the
    temperature is measured. The software requires a known reference temperature and the value and position are provided by the user
    (this information is shown as a single small green circle on the image).
    It is possible to have the absolute temperature but you need an image from a thermal camera with correct radiometric calibration
    and radiometric exif data loaded into image.

    Usage Example:  
                    Without radiometric image, you need to specify reference temperature point value and position
                    ./ai_thermometer.py --image_in=./image1.jpeg --image_out=./image1_out.jpg --reference_px=200 --reference_py=400 --reference_temperature=31

                    You need a radiometric jpeg to use this software in this configuration
                    ./ai_thermometer.py --image_in ../IR_2412.jpg --image_out ../IR_2412_out.jpg --radiometric True

    Tested on https://www.flir.com/it/oem/adas/adas-dataset-form/

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

import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import flirimageextractor

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

    # Flags
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_in", default="./input_image.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--image_out", default="./output_image.jpg", help="Image output")
    parser.add_argument("--reference_px", default="200", help="X reference T position")
    parser.add_argument("--reference_py", default="400", help="Y reference T position")
    parser.add_argument("--head_point", default="2", help="Openpose head point (around it will be created a roi)")
    parser.add_argument("--roi_sizex", default="8", help="Roi size on X")
    parser.add_argument("--roi_sizey", default="8", help="Roi size on Y")
    parser.add_argument("--reference_temperature", default="25.5", help="Reference temperature")
    parser.add_argument("--limit_temperature", default="37.5", help="Limit temperature")
    parser.add_argument("--radiometric", default="False", help="User radiometric temperature, else reference temperature is used")

    parser.add_argument("--openpose_folder", default="/ai_thermometer/openpose/models/",
                        help="Path to the local OpenPose installation directory")

    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()

    params["model_folder"] = args[0].openpose_folder
    params["face"] = False
    params["hand"] = False
    params["net_resolution"] = "512x384"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    radiometric = True if args[0].radiometric == "True" else False

    # Get radiometric matrix
    if radiometric:
        try:
            flir = flirimageextractor.FlirImageExtractor()
            flir.process_image(args[0].image_in) 
            thermal = flir.get_thermal_np()
        except:
            print("Input image is not radiometric!")
            os._exit(-1)

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()

    # Read image
    imageToProcess = cv2.imread(args[0].image_in)

    # If Radiometric is True
    if radiometric:
        # Convert to gray levels
        gray = cv2.cvtColor(imageToProcess, cv2.COLOR_BGR2GRAY)
        
        # Invert levels
        gray_inverted = cv2.bitwise_not(gray)

        # Convert inverted grayscale to Color RGB format
        imageToProcess = cv2.cvtColor(gray_inverted, cv2.COLOR_GRAY2BGR)

    # Set image to openpose video
    datum.cvInputData = imageToProcess 

    opWrapper.emplaceAndPop([datum])

    # Get openpose output
    bodys = np.array(datum.poseKeypoints).tolist()

    imageToShow = datum.cvOutputData

    # If a body is recognized
    if type(bodys) is list:
        for body in bodys:
            # Face points (0, 15, 16) refered to body_25 openpose format
            face = [[int(body[0][0]),int(body[0][1])],
                [int(body[15][0]),int(body[15][1])],
                [int(body[16][0]),int(body[16][1])]]

            if 0 not in face[0] and 0 not in face[1] and 0 not in face[2]:
                # Get line values from eyes line to neck
                
                # Set parameters
                size_x = int(args[0].roi_sizex)
                size_y = int(args[0].roi_sizey)

                reference_x = face[int(args[0].head_point)][0]
                reference_y = face[int(args[0].head_point)][1]

                reference_px = int(args[0].reference_px)
                reference_py = int(args[0].reference_py)

                offset_x = 0
                offset_y = 0

                counter = 0
                average = 0

                # Calculate average values in face rect 
                for y in range(reference_x-size_x+offset_x, reference_x+size_x+offset_x):
                    for x in range(reference_y-size_y+offset_y, reference_y+size_y+offset_y):
                        if radiometric:
                            average += thermal[x, y]
                        else:    
                            average += imageToProcess[x, y][0]

                        counter += 1
                
                #Calculate average
                if counter!=0:
                    average = average / counter
                
                #Print data
                if counter!=0:
                    if radiometric:
                        # Assign temperature
                        temperature = average

                        # Print some data about temperature
                        print("Face rect temperature: T:{0:.2f}C".format(temperature))
                    else:
                        # Get pixel value of reference point  
                        reference_temperature = imageToProcess[reference_px, reference_py][0];

                        # Temperature calculation with reference point temperature
                        temperature = (average * float(args[0].reference_temperature))/reference_temperature

                        # Print some data about temperature
                        print("Face rect temperature: T:{0:.2f}C, {1:.2f}".format(temperature, average))
                        print("Reference floor pixel: {0}".format(str(reference_temperature)))

                        cv2.circle(imageToShow, (reference_px, reference_py), 5, (0, 250, 0), -1)
   
                    # Print temperature on image
                    text_str = 'T:{0:.2f}C'.format(temperature)

                    x1 = reference_x
                    y1 = reference_y

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.9
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    if temperature > float(args[0].limit_temperature):
                        color = (0, 0, 255)
                    else:
                        color = (255, 0, 0)

                    # Draw rectangle on face
                    cv2.rectangle(imageToShow, (reference_x-size_x+offset_x, reference_y-size_y+offset_y), 
                        (reference_x+size_x+offset_x, reference_y+size_y+offset_y), (0, 250, 0), 2)

                    # Draw Text rectangle
                    cv2.rectangle(imageToShow, (x1-16, y1-20), ((x1-16) + text_w, (y1-20) - text_h - 4), color, -1)

                    # Draw Text
                    cv2.putText(imageToShow, text_str, (x1-16, y1-20), font_face,
                        font_scale, (255,255,255), font_thickness, cv2.LINE_AA)

    cv2.imwrite(args[0].image_out, imageToShow)

    #cv2.imshow("AiTemperature", imageToShow)
    #cv2.waitKey(5000)

except Exception as e:
    print(e)
    sys.exit(-1)
