# U-net 34 Model Training

## Quick Start-up

### Notes and parameters to change

This is a script made to train on 16-bit brightfield images of cells. It expects cell images and a binary mask that the model will take to be the correct segmentation that should be made 
on a given image. For required packages please read the 'Package Management' section in the readme file in the home bf_nucleus_segmentation folder. 

On line 12 of NSM_main.py, brightfield_folders is initialized and expects to be directed to folders containing brightfield images. This variable can handle being passed multiple folders at 
a time as long as the subsequent mask_folders also has the same number of folders. They should also follow in the same order. For example, the first folder passed to brightfield_folders 
assumes that its respective mask folder is the first one passed to mask_folders. 
Inside the brightfield_folders, you can have files with images of any size and any other file type and the model will still only take those who are meant to be used for training. This 
works because the code assumes that the brightfield image and its corresponding mask have the exact SAME name, character for character including file extension. 
On line 16, mask_folders is initialized and expected to be directed to folders containing the respective binary masks for the brightfield images given in brightfield_folders. 

As this model trains, it can give output predictions on given training samples after each epoch as the model trains on the sample data. These are specified on lines 25-28. Any number of
samples can be given. It can even be removed. If you don't want this functionality please remove lines 25-33 and lines 40-43. 
On line 33, the directory should be given to where you would want the given sample predictions to be saved to. 
At the end of training this code outputs a prediction on every image in the training and validation set to verify to allow the user to manually. This directory is automatically made in 
the same directory that the model is saved to. So if the model is saved to /path/to/model/model.keras then the final predictions are saved to 
/path/to/model/final_predictions_oblong_fitting/ automatically. If you don't want this functionality, feel free to comment out lines 204-232.

Initialization of history on line 36 of NSM_main.py:
  -on line 39, epochs is initialized. This parameter is relatively important. If you are doing intial function testing just to make sure the code runs and that is it, you can set this to
1. When attempting to generate a full fledged model, something between 50-100 epochs is recommended. For now, I suggest to use 50. However, in some instances it may make sense to
use even 300. If after training, and you notice in your epochs that at some point the predictions become all black, then it is likely that your model has become over-fitted and that you
are using to many epochs.
  -on line 39, batch_size is initialized. This parameter is also relatively important. For now, I simply recommend leaving it at 4. However, if your sample size is relatively small (less
than 100) then I may try 16.
  -on line 44, iterations is initialized. I also recommend leaving this parameter as 1, but if your model is having a bit of trouble, your sample size is relatively small, and you can't
quite increase the epochs as the model is starting to over-fit then try increasing this to 2. If that does not solve the issue and the model simply starts over-fitting in the 2nd iteration
then likely you will need to adjust some parameters that are not given in the call. 

Given that changing the above parameters is not helping I would suggest changing some of the values in tversky_loss(), composite_loss(), or ElasticDeformation() in
NucleusSegmentationModel.py. These can have very drastic changes to the behavior of the model. Which can greatly increase the efficiency, but does take some knowledge. My suggestion is
to individually google the parameters and how changing them affects model behavior so that you can make an educated guess on what to change and by how much. Note that a lot of machine
learning optimization is literally just trying all the variations of the changes in these numbers and manually checking which produced the best result as there is little way to know
beforehand what may lead the model to the best predictions.

On line 47, the location of where the model should be saved to and its name should be given. 

On lines 54-57, there is some additional code that is commented out, but I recommend using if you want to use it. Its purpose is simply creating a log file of what your current 
NucleusSegmentationModel.py and NSM_main.py look like.

### How to run

The main interface of this code is NSM_main.py. When you have set up all of your variables as you've desired. You should run NSM_main.py and that will work with 
NucleusSegmentationModel.py and begin training the model. 