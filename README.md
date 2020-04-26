# OpenCV based human detection and social distance measurement.

# Steps to reproduce the code in Windows 10:

* Step 1: Install Anaconda as Administrator for Windows (3.7 Version) https://www.anaconda.com/distribution/#windows
* Step 1A: Read the instructions to install. https://docs.anaconda.com/anaconda/install/windows/
Install as instructed (all the default options). 
* Step 2: After successful installation, open Anaconda Prompt as Administrator from Start button on taskbar (type in Anaconda Prompt). A black screen would appear.
* Step 3: The first line would be (Replace YourName with your username)
`(base) C:\Users\YourName>`
* Step 4: Extract the contents of the zip folder supplied in the Desktop.
* Step 5: `cd C:\Users\YourName\Desktop\Social_distancing_project`
* Step 6: You have now entered the Project directory.
* Step 7: `conda env create  -f environment.yml`
This will take some time to install. This is our environment, without which our project won’t work.
* Step 8: If no errors (no red texts), all set to go.
* Step 9: `conda activate  social_dist_copy`
* Step 10: `python social_distancing.py --input videos/TownCentreXVID.avi > logs/{Time}.txt`
Provide a unique name (such as Time) to the log file such that it does not mix with other subsequent runs.
This command will create log file (in txt format) in the logs folder in the project directory.
* Step 11: Ctrl + C (While on Anaconda Prompt window) to stop the application.
* Step 12: If running multiple instances of the application, Open the Anaconda Prompt, Go to the Project Directory and follow step 9 and 10.

# Output

![](Assets/output.gif)

![](Assets/output_1.gif)


# Techniques Used

* OpenCV based Object Detection SSD MobileNet Model for identifying people.
* dlib based object tracker to track people and provide IDs.
* OpenCV based Perspective correction (Homography) to get bird view of the street.
* Triangle similarity based distance estimation between group of people.

# Limitations (Short Term)

* Currently, A weak human detector (MobileNet SSD) is employed for test purposes, other detectors (YOLO or Faster RCNN) can be easily tested which may provide better accuracy. The current detector is used so as to run with fewer system resources.
* Algorithm not currently tested in the work (indoor) environment, only in webcam videos of pedestrians (outdoor). 
* Algorithm can make use of multiple cameras focusing the view, if available to arrive at better localization and estimation of distances.

# Limitations (Long Term)

* Occlusion: The application will fail to detect persons, if they are hidden behind another person or machines/furnitures as seen from the camera perspective. The Indoor environment may provide more challenges as part of human body would be occluded with furnitures, machines, etc.
* The Calibration step (while using Triangle Similarity Technique) is important to arrive at maximum accuracy in depth estimation. Currently, a hardcoded value is used in the algorithm, which might solve the said purpose but with less accuracy.


# References
A big shoutout to the people involved in the Similar projects. This work is nothing but a compilation of codes from these repositories.
You can visit individual link for in depth discussion on the methods.

* Triangle similarity technique (https://github.com/Subikshaa/Social-Distance-Detection-using-OpenCV) by Subikshaa (https://github.com/Subikshaa)
* People Counter (https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/) by Adrian Rosebrock
* Object Detection with Deep Learning and OpenCV (https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/) by Adrian Rosebrock

Also

* Landing AI (https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/)


 
 
