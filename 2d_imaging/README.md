# Udacity 2D Imaging Project

## Dataset

The [NIH Chest X-Ray dataset](https://www.kaggle.com/nih-chest-xrays/data) was used for this project.


## Problem/Task

The prompt for this project was to differentiate between patients with pneumonia and those without using the chest x-ray data and a convolutional neural net (CNN) to perform the classification. Otherwise the approach to the problem was open-ended.


## Libraries and Tech Stack

- `pandas` and `pyplot` were used for the exploratory data analysis
- `keras` and `tensorflow` were used to split data, augment it, and train the model.
- The choice of CNN was a pre-trained `inceptionV3` model whose weights were initially locked to train the newly-added dense layers, and later opened up to training the remaining convolutional layers as medical imaging bears less resemblance to ImageNet classes than in many other transfer-learning tasks.

## Important Files and Concepts

- `EDA.ipynb` contains the exploratory data analysis.
  - Of special note is the definition of the positive/negative classes, while the dataset *does* contain a label for `pneumonia`, classifying on this alone proved ineffective and required a deeper look at how other conditions were affecting the model.
- `"Build and Train Model.ipynb"` contains all the data splitting, training set augmentation, and multi-stage model training code.
  - The work was done in a remote environment that had a habit of disconnecting during training so it ultimately required short training stints with aggressive backup of any non-reproducible data.
  - The final model didn't have a remarkable ability to identify positive cases so the chosen threshold for classifying as positive was chosen in such a way that it excelled at eliminating negative cases. This is covered in more detail at the bottom of the notebook.
- `FDA_Submission.pdf` is an example FDA approval submission listing the intended/indicated uses, it also includes performance information about the algorithm
- `out/` contains several output files regarding performance: ROC, precision-recall, and training history. These are also included in the FDA submission pdf.


## Performance, Personal Commentary, and Potential Followup

The final performance was influenced by the potential clinical context of the algorithm. Rather than simply shooting for the highest F1 score I instead chose a threshold that had what I considered to be clinical utility. The algorithm achieves 95% recall on the positive class at the cost of only 24% precision. The negative class has 21% recall with 94% precision. In other words it is unlikely to generate false negatives and still catches roughly 20% of the true negatives. As a screening or prioritization tool this has virtue.

The size of this dataset gave a lot of reasons for hope but this size was as much a challenge as an asset as so many of the conditions contained within could be mistaken for one another. Ultimately I had to approach the problem as one of identifying everything that looked like pneumonia or was comorbid with it to the extent that it could be treated as an indicator of the desired positive class.

This is one approach, and it resulted in a substantial improvement over a more simplistic model, but there are a number of different ways to approach this data. I strongly suspect that the construction of a good training set is the dominant feature in the final quality of the output.

Improvements on this approach boil down to rethinking the way the problem is addressed. A pipeline approach that separates the `No Finding` x-rays from the remainder using one CNN, and then tries to assess the remainder with one or more CNNs might perform better. An ensemble approach could help for much the same reasons: There needs to be more fine-grained distinction between similar-looking classes.

This would be an interesting case-study to take a lottery-ticket approach to x-rays: If we find a CNN that works well on one class of chest x-rays will it generalize to others after substantial pruning.

Lastly, this seems like a case where the CNN output should be combined with other patient information data to inform classification and to minimize the cost of misclassification. If we are using this as a way to prioritize x-rays for a radiologist we can do the most good by erring on the side of caution for very high-risk patients. 
