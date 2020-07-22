# Udacity Electronic Health Records Project

## Dataset

The [dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) is from UC Irvine covering diabetes patients. As this dataset is significantly smaller than the ones used for training CNNs it is included in this folder.

## Problem/Task

The task is to perform a regression based on the EHR data that predicts how long someone would stay in the hospital. The goal for this is to identify patients that will already require a longer hospitalization so they can be recruited as candidates for a new drug study. If we chose patients at random some would inevitably require a shorter stay and we would need to spend the extra money to keep them hospitalized over the multi-day treatment course.

## Libraries and Tech Stack

- The standard `pandas`, `numpy`, and `scikit-learn` are included for storing and splitting data.
- Exploratory data analysis is done with `pyplot` and `seaborn`.
- The model is trained using `tensorflow` and `tensorflow-probability` to create a Bayesian neural net that models the probability distributions of our outputs as well.
- `aequitas` is used to assess algorithmic bias across demographic groups.

## Important Files and Concepts

- `Project_EDA.ipynb` contains all the exploratory data analysis.
- `Project_Data_ETL.ipynb` loads the data and performs preprocessing tasks.
- `Project_NN.ipynb` performs the training, calculates performance metrics, and ends with demographic analysis.
- `student_utils.py` is my code and performs much of the preprocessing.
- `project_tests.py` performs tests on `student_utils.py` to ensure that the preprocessing is being dong correctly. The files exist in the folder to run it successfully, the first test can take a while.
- `utils.py` was provided and performs some dataframe manipulation and builds vocab files for various categorical features.
- Everything remaining is in place so the project can be run, if desired. This is possible here because the dataset is small and the training time is short, and I wanted to retain that where possible.

## Reflections and Commentary

The algorithm performed fairly well (~75% precision), and would certainly improve outcomes over randomly choosing patients. The Bayesian layer to the network grants additional information in the form of the predicted mean, which was used, as well as the estimated standard deviation for each observation, which was not used. It would be straightforward to extend this algorithm to use a more complex decision function that incorporates a confidence interval rather than a simple threshold on the predicted mean. 
