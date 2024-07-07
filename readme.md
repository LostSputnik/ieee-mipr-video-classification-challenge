This repository contains codes and relevant materials for team ServerDown's submission to 2nd AVA Challenge@IEEE MIPR 2024.

Experiment Results as Presented in Technical Report w/ Codes:

| Experiment                                                                                                                           | Public ROC | Private ROC |
| ------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ----------- |
| Per-Frame Considering Last Three Frames [code (training + inference)](per-frame-cnn.ipynb)                                           | 0.6873     | 0.7428      |
| VidSwin [training code](vidswin-training.ipynb), [inference code](vidswin-inference.ipynb)                                           | 0.6619     | 0.7118      |
| Ensemble (Average) (CNN-Transformer and Per Frame)                                                                                   | 0.7459     | 0.7005      |
| End-to-End CNN-Transformer [code (training + inference)](end-to-end-cnn-transformer.ipynb)                                           | 0.6905     | 0.6820      |
| Ensemble (Weighted) (CNN-Transformer and Per Frame) (Selected Submission on Kaggle) [ensemble code](weighted_ensemble_submission.py) | 0.7560     | 0.6576      |
| Pretrained-CNN + RNN [code (training + inference)](pretrained-cnn-rnn.ipynb)                                                         | 0.6280     | 0.6571      |
