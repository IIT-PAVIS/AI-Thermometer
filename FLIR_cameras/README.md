# :thermometer: AiThermometer Flir :thermometer:
<!--
Code for automatically measuring the temperature of people using a thermal camera.
The software can be freely used for any non-commercial applications and it is useful
for the automatic early-screening of fever symptoms. The code is open and can be 
improved with your support, please contact us at aithermometer@iit.it if you want to help us.
-->


## Description

The software first detect people with an off-the-shelf body pose detector and then extract location of the face where the temperature is measured. In this version flir spinnaker driver camera is used. It can connect to remote gigE camera. It acquire 16bit images with temperature level and convert it to real temperatures measures matrix. Maximum face temperature and eyes temperature are extracted. Extracted video is converted to RGB24 false color images (COLORMAP_JET) and can be accessed from http mjpeg multipart streaming embedded server, data can be accessed in json format with rest/json calls on configurable ports.

config.ini contains all configurable parameters and any parameters is documented such as each python code line  

 
## Installation steps
Code is developed in Python3 and tested on Ubuntu 18.04 with NVidia driver, Cuda 10.0 and Cudnn 7.6.5, Flir A600. 

AiThermometer can be run in a Docker container, as described in [this section](#markdown-dockerfile).

* [x] **Install the requirements**  
To run this code you need to install:

    * **OpenPose**:    
    Please follow the instruction in the repository [gitHub](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and install OpenPose in `ai_thermometer/openpose/` folder.   
    In case you prefer to use a different OpenPose installation folder, you can pass it using the `--openpose_folder` argument. 

    * **TurboJpeg**:  
        `sudo apt-get install libturbojpeg`
        `sudo pip3 install PyTurboJPEG`
     
    * **OpenCV**:    
        `apt-get install python3-opencv`  
        `pip3 install opencv-python`

    * **Ubuntu 18.04 x64 Spinnaker Driver and Python Library**:
        follow download/install istruction from https://www.flir.com/products/spinnaker-sdk/

## Usage
### Live capture from Flir camera 
Tested with Flir A600, it acts like web server listening on some video/data ports cofigured in config.ini file

Usage Example:
```
python3 ai_thermometer.py --config_file ./config.ini
```

## Disclaimer
The information and content provided by this application is for information purposes only. 
You hereby agree that you shall not make any health or medical related decision based in whole or in part on anything contained within the application without consulting your personal doctor.

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors, PAVIS or IIT be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

## LICENSE
This project is licensed under the terms of the MIT license.

This project incorporates material from the projects listed below (collectively, "Third Party Code").  This Third Party Code is licensed to you under their original license terms.  We reserves all other rights not expressly granted, whether by implication, estoppel or otherwise.

1. [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 
2. [OpenCV](https://opencv.org)
3. [flirimageextractor](https://pypi.org/project/flirimageextractor/)
4. [flir spinnaker](https://www.flir.com/products/spinnaker-sdk/)
5. [turbo jpeg](https://libjpeg-turbo.org/)
6. [PyTurboJPEG](https://pypi.org/project/PyTurboJPEG/)

<img src="../iit-pavis.png" alt="iit-pavis-logo" width="200"/>
