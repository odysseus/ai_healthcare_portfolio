# Udacity 2D Imaging Project

## Dataset

The [NIH Chest X-Ray dataset](https://www.kaggle.com/nih-chest-xrays/data) was used for this project.


## Problem/Task

The prompt for this project was to differentiate between patients with pneumonia and those without using the chest x-ray data and a convolutional neural net (CNN) to perform the classification. Otherwise the approach to the problem was open-ended.


## Libraries/Tech Stack

- **Pandas** and **pyplot** were used for the exploratory data analysis
- **Keras/Tensorflow** was used to split data, augment it, and train the model.
- The choice of CNN was a pre-trained **InceptionV3** model whose weights were initially locked to train the newly-added dense layers, and later opened up to training the remaining convolutional layers as medical imaging bears less resemblance to ImageNet classes than in many other transfer-learning tasks.

## Important Files and Concepts

- `EDA.ipynb` contains the exploratory data analysis.
  - Of special note is the definition of the positive/negative classes, while the dataset *does* contain a label for `pneumonia`, classifying on this alone proved ineffective and required a deeper look at how other conditions were affecting the model.
- `"Build and Train Model.ipynb"` contains all the data splitting, training set augmentation, and multi-stage model training code.
  - The work was done in a remote environment that had a habit of disconnecting during training so it ultimately required short training stints with aggressive backup of any non-reproducible data.
  - The final model didn't have a remarkable ability to identify positive cases so the chosen threshold for classifying as positive was chosen in such a way that it excelled at eliminating negative cases. This is covered in more detail at the bottom of the notebook.
- `FDA_Submission.pdf` is an example FDA approval submission listing the intended/indicated uses, it also includes performance information about the algorithm
- `*.png` files are the performance graphs (ROC, precision-recall, training history) that are also included in the FDA submission pdf.


## Reflections/Commentary

The size of this data set gave a lot of reasons for hope but this size was as much a challenge as an asset as so many of the conditions contained within could be mistaken for one another. Ultimately I had to approach the problem as one of identifying everything that looked like pneumonia or was comorbid with it to the extent that it could be treated as an indicator of the desired positive class.

This change was a substantial improvement over the first iteration which was incapable of evolving past a no-skill level of classification, but the end results were not remarkable. Ultimately if I approach this problem again I will probably look at a pipeline of CNNs to first try and separate the x-rays labeled `No Finding` from the ones with any other finding label. From there it would hopefully be easier to identify specific structures belonging to any of our other labels like `Consolidation` or `Mass`. Essentially the model needs the ability to make more specific judgments. For the same reason an ensemble approach could work well here.
