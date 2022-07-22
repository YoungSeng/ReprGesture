# The ReprGesture entry to the GENEA Challenge 2022

Submitted to GENEA Challenge & Workshop of ACM ICMI 2022

Anonymous Authors

## 1. Abstract

This paper describes the ReprGesture entry to the Generation and Evaluation of Non-verbal Behaviour for Embodied Agents (GENEA) challenge 2022. The GENEA challenge provides the processed datasets and performs crowdsourced evaluations to compare the performance of different gesture generation systems. In this paper, we explore an automatic gesture generation system based on multimodal representation learning. We use WavLM features for audio, FastText features for text and position and rotation matrix features for gesture. Each modality is projected to two distinct subspaces: modality-invariant and modality-specific. To learn inter modality-invariant commonalities and capture the characters of modality-specific representations, gradient reversal layer based adversarial classifier and modality reconstruction decoders are used during training. The gesture decoder generates proper gestures using all representations and features related to the rhythm in the audio.

## 2. Videos

### 2.1 Ablation study

(GT / ReprGestrue / without Wavlm / without GAN loss / without domain loss / without Repr)

Please Click & Download：Ablation_study~3.mp4

### 2.2 Additional experiments

(GT / ReprGestrue / with diff loss / with text emotion / with diff loss and text emotion)

Please Click & Download：Additional_experiments~5.mp4

However, the results of these experiments did not turn out very well, so they were not mentioned in the final submission of the system or the paper.

## 3. Code and Pre-trained Model

### 3.1 Data Processing

The distribution of speaker IDs:

<div align=center>
<img src="https://user-images.githubusercontent.com/37477030/180232909-cc325614-95dc-41f3-82cc-bc0cf1de20dc.png" width="500px">
</div>

<!---
![image](https://user-images.githubusercontent.com/37477030/180232909-cc325614-95dc-41f3-82cc-bc0cf1de20dc.png)
--->

We noted that the data in the training, validation and test sets were extremely unbalanced, so we only used the data from the speaker with identity "1" for training.

<div align=center>
<img src="https://user-images.githubusercontent.com/37477030/180236015-11316fe1-025c-4fca-8b6a-a0dbab7e5d51.png" width="300px">
</div>

<!---
![screenshot_on_val_2022_v1_000](https://user-images.githubusercontent.com/37477030/180236015-11316fe1-025c-4fca-8b6a-a0dbab7e5d51.png)
--->

Due to the poor quality of hand motion-capture, we only used 18 joints corresponding to the upper body without hands or fingers.

### 3.2 Code

The code will be uploaded later.

### 3.2 Pre-trained Model

The pre-trained model will be uploaded later.

(_Double-anonymous Page For ICMI 2022 GENEA challenge_)
