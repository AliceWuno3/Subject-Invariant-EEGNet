# Subject-Invariant-EEGNet
A EEG based Compact Convolution Neural Network
## Evironment Requirement
* Python 3
* Pytorch 0.2+
## Dataset Description
Data collection and reduction for the Gamification EEG study under ARL’s Training Effectiveness research.

Each subject has one corresponding EEG signal bdf file; each file can be segmented into 600 trails.

The model was trained based on 99*600 trails of EEG signals.
## Model Description
EEGNet + Subject Invariant layer that can reversely reduce subject bias
## Method
subject independent 10-folder cross validation
## Results
Baseline: Dropout Rate 0.25, F1 score: 0.75

SI-EEGNET: Dropout Rate 0.15 based on 0.25 dpr of baseline, F1 score: 0.77
## Citation
EEGNet - https://arxiv.org/abs/1611.08024

SIDANN - https://doi.org/10.1145/3382507.3418813
## Others
Many Thanks to Yufeng Yin, Soheil Rayatdoost, and Professor Mohammad Soleymani
