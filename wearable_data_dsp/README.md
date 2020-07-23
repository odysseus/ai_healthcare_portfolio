# Udacity Wearable Health Device / Digital Signal Processing Project

## Dataset

This dataset was originally from the [Cardiac Arhythmia Suppression Trial (CAST)](https://physionet.org/content/crisdb/1.0.0/). The exact dataset used in this project can also be found in the original project repo [here](https://github.com/udacity/nd320-c4-wearable-data-project-starter). The algorithm this is based on is loosely adapted from the following [paper](https://arxiv.org/pdf/1409.5181.pdf).

## Problem/Task

The task was to build a motion-compensated heart rate detection algorithm that filters out signal noise originating from the motion of the arm. The heart rate estimation is done with a signal processing approach on the photoplethysmogram signal. To compensate for motion we have access to a separate accelerometer signal from the same device that can help compensate for noise in the PPG signal.

## Libraries and Tech Stack

- `numpy` and `scipy` were used for the majority of this with `scipy-signal` being particularly important in computing spectrograms of the signals.
- `pyplot` was used to visualize the signals and the output of the algorithm.

## Important Files and Personal Contribution

- Everything is contained in `pulse_rate_starter.ipynb`. Essentially all of the code outside of `LoadTroikaDataset` and `LoadTroikaDataFile` is my own.

## Performance, Personal Commentary, and Potential Followup

On the noisy signals used in training this achieved roughly 11bpm error on measuring heart rate. Given more time to implement them I think this error could be lowered by 2-4bpm by having the algorithm exclude extreme values and extreme value changes, I discuss that in the final paragraph. On an unseen test set of less noisy data the error was 3.7bpm.

This was a fascinating DSP project as it involved combining two signals taken from entirely different sensors but affected by the same physical processes. The suggested approach (and the one I implemented) was simply to rule out a heart rate signal if it matched the accelerometer signal, or one of its harmonics.

While the performance was, overall, pretty good there were definite issues with the way it functioned that could be improved upon. It has a tendency to take extreme value swings which we know to be illogical, heart rate doesn't increase 40bpm in two seconds and then drop back down. It similarly has no problems predicting fairly high (>180) or low (<70) heart rates because nothing otherwise tells the algorithm that these are uncommon. There are also cases where the heart rate signal is completely lost in the PPG signal and there's no real way to justify a prediction that matches the true heart rate. Examples like these can't be fixed but they could be identified and ignored, passing forward a previous more confident value in their place, for example.

My hunch for a new approach would be to try and incorporate all of this into a Bayesian model that gives a confidence to each frequency in the spectrogram slice, as well as an overall confidence in the prediction. Given that most of the previously discussed problems/improvements take the form of prior knowledge a Bayesian approach seems natural. This is something I hope to implement in the future.
