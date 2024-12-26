# This is a U-net 34 model made to Segment the Nucleus of mostly HeLa cells.

## Quick Startup

### Package Management

In the package_management folder in this repository I have created a .yml file which can be run in anaconda to install the packages essential for this code. 
Anaconda can be installed here: https://www.anaconda.com/download/success
Here is an example usage of this file: 

conda env create --file /path/to/NucSeg_environment.yml

Make sure your python IDE (ex: Pycharm/Spyder/VScode) points to the conda environment.
Here is the download for pycharm (make sure to scroll down for the community edition as the first shown one is a paid for version with a free trial): https://www.jetbrains.com/pycharm/download/?section=windows#section=windows

### Models

Add the files in Nucleus_Segmenter_App called CreateMask.py and createmask_main.py to your python IDE. 
Also download the model file itself. Since github won't allow such large files I have uploaded to my google drive here:

https://drive.google.com/drive/folders/18JyO2-udz5NiRr88vHjSQWx0nH6MF3ag?usp=sharing

### How to use

In createmask_main.py near the bottom of the script on lines 103-105 update the paths to the location of the requested folder. 
Line 103 expects the absolute file path to the model. Line 104 expects the file path of where you would like the output binary masks to be saved.
Line 105 expects the brightfield images that should be made predictions on. 

If you use spyder_CreateMask.py you will need to change line 1 in createmask_main.py to 'from spyder_CreateMask import CreateMask' instead of 'from CreateMask import CreateMask'.

With this you can run the script from createmask_main.py.
The script can handle having multiple image types and sizes in the given brightfield image folder, it will skip any that don't match the correct size and file type. 
When an appropriate image is reached, it will display the unchanged image, the image with the subtracted background, and the image with just the edge of the ROI shown on the original image. 
You can just to interactively edit the ROI edge, save the image, increase the size of the ROI a small amount, reduce the size of the ROI a small amount, or skip the image altogether
(any skipped images will be logged in a text file, saved in an automatically created folder called log_folder).

## ImageJ Macro

Once you have created the binary masks for each cell, I have created an imagej macro to take binary masks and complete the preprocessing steps for Single Nucleosome Movie .nd2 files.
The macro will create and save an enhanced contrast and background subtracted version of the movie. Next, the macro will use the binary mask to create a ROI and save the individual frames of each movie (with each movie in a
separate folder) into the specified output folder. 

There are some parameters to change at the bottom of the macro. Starting on line 82 there is 4 folder_path variables. These should point to location of a folder with nd2 movies inside. There does not need to be 4 file paths here,
I simply have been working with batches of 4 different movie folder files for 4 different treatment groups. This could be anywhere between 1 to infinity amount of movie folder files depending on how many movie folder files you want 
to be processed one after another. These are then independently called in lines 95-98 which can be seen process_nd2_files(folder_path1...), process_nd2_files(folder_path2...), and so on so forth (the given folder_path # number is
different). 
On line 87 the mask_folder variable should be given the path to where your currently existing masks for the respective movies are. 
On line 88 the utrack_folder should be given the file path to where you would like the 
movie frames with background subtracted to be saved (each cell will have an independent folder that is made in the script inside this folder that you give). For example, I call this something like 
r"C:\path\to\frame\folder\for\experiment\2024-12-23_Jason" for my experiment. Which is outputted in a way that can be directly passed to the image folder for our u-track software. 
On line 89 this is essentially how many frames you want to skip, by inputting what frame the duplicated movie should start on. I typically skip 50, so the starting frame is 51.
On line 90 this is normally the last frame in your movie, so if your movie is 300 frames this should be 300.

On line 92 is a parameter for how to calculate which index number on the movie is associated with which mask. For example the default is x * 2. Which assumes the movie file looks like this 'name_001.nd2' and the mask looks like this
'name_002'. The second example  on line 93, which is commented out, assumes you have to subtract by 80 before multiplying by 2.

This macro is at imagej_macros/make_image_seq_from_roi.py in this repository. You can run the macro like this in imagej: plugins > macros > run... and then by clicking on the file in your files. 
