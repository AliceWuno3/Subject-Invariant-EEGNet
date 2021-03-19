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
## Results
Baseline: Dropout Rate 0.25, F1 score: 0.75
SI-EEGNET: Dropout Rate 0.15 based on 0.25 dpr of baseline, F1 score: 0.77
## Citation
Many Thanks to Yufeng Yin, Soheil Rayatdoost, and Professor Mohammad Soleymani
