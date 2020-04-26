# OpenCV based human detection and social distance measurement.

# Requirements: 
Use Anaconda for Python 3.7 to create environment from `environment.yml`

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
 
