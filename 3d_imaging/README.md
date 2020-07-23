# Udacity 3D Imaging Project

## Dataset

This data was taken from the [Medical Decathalon](http://medicaldecathlon.com/) competition, specifically the hippocampus images.

## Problem/Task

Given a set of 3D images, which have been cropped to include the hippocampus region, perform a segmentation that allows measuring the volume of the hippocampus to check for Alzheimer's disease progression. The suggested architecture for this is the U-Net.


## Libraries and Tech Stack

- `pandas` and `pyplot` were used for the exploratory data analysis. `nibabel` is used to load NIFTI files and `3D Slicer` (desktop app) was used to view them.
- `pytorch` was used to process the data into tensors, and to define and train the network.
- `orthanc` and `dcmtk` were used to replicate a clinical network.


## Important Files and Concepts

- `Final Project EDA.ipynb` contains the exploratory data analysis. It was less important for this task than in typical scenarios as the 3D volumes were pre-cropped and properly labeled.
- `src/build_train_model` contains the Python files defining the pytorch UNet and all the associated code for converting the data into tensors and training the model. It is a mixture of my own code as well as pre-written functions.
- `src/generate_reports` contains code to take a finished model, run it on a new example, and create a PNG report that is sent to an OHIF web-viewer for medical images. `inference_dcm.py` is largely (~ >80%) my code, the code in the `utils/` folder is mine as well. The shell scripts and UNet definition were provided.
- `out/` contains screenshots of the final outputted reports for test files. It also contains the tensorboard output for the training progress and the validation results.
- `validation.pdf` is a document for a clinical partner discussing the necessary dataset to be acquired for FDA validation.

## Performance, Personal Commentary, and Potential Followup

This algorithm performed exceptionally well with a mean Dice coefficient of .903, mean Jaccard index of .825, mean sensitivity of .902 and mean specificity of .998. Furthermore, since the goal of the algorithm is to track changes in the hippocampal volume these small errors won't affect our performance so long as the algorithm consistently over/under estimates on the same patient. In other words if the segmented volume measurement says the patient's hippocampus has declined in size we need to ensure that this is not just a fluke of the algorithm.

Moving forward I would want to better understand any edge cases in the training data. Are there certain features of an image that cause problems for our approach? Can those features be ameliorated with a different imaging protocol or machine settings? Can we identify patients whose images are likely to have poorer accuracy automatically. Answers to these questions are likely to be more important to the future of the algorithm than any attempts to push the already excellent performance numbers up even higher.
