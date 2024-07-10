This repository contains codes and relevant materials for team ServerDown's submission to 2nd AVA Challenge@IEEE MIPR 2024.

Experiment Results as Presented in Technical Report w/ Codes and Submission Files:

| Experiment                                                                                                                           | Public ROC | Private ROC |
| ------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ----------- |
| Per-Frame Considering Last Three Frames [code](./per-frame-cnn/)                                           | 0.6873     | 0.7428      |
| VidSwin [code](./VidSwin/)                                         | 0.6619     | 0.7118      |
| Ensemble (Average) (CNN-Transformer and Per Frame) [code](./ensemble-script/)                                                                                  | 0.7459     | 0.7005      |
| End-to-End CNN-Transformer [code](./e2e-cnn-transformer/)                                           | 0.6905     | 0.6820      |
| Ensemble (Weighted) (CNN-Transformer and Per Frame) (Selected Submission on Kaggle) [code](./ensemble-script/) | 0.7560     | 0.6576      |
| Pretrained-CNN + RNN [code](./pretrained-cnn-rnn/)                                                         | 0.6280     | 0.6571      |

Given the large size of the model weights, we did not upload them to GitHub. You can download the trained weights from [Google Drive](https://drive.google.com/drive/folders/1Daqk3htA5xQcGdkq-Me314fJHKSVetez?usp=sharing) and update the paths in the inference codes to reproduce the results. 