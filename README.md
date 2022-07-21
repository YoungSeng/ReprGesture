# The ReprGesture entry to the GENEA Challenge 2022

Submitted to GENEA Challenge & Workshop of ACM ICMI 2022

Anonymous Authors

## 1. Abstract

This paper describes the ReprGesture entry to the Generation and Evaluation of Non-verbal Behaviour for Embodied Agents (GENEA) challenge 2022. The GENEA challenge provides the processed datasets and performs crowdsourced evaluation to compare the performance of different gesture generation systems. In this paper, we explore an automatic gesture generation system based on multimodal representation learning. We use Wavlm features for audio, Fasttext features for text and position and rotation matrix for gestrue. Then each modality are projected to two distinct subspaces: modality-invariant and modality-specific. In order to learn inter-modality-invariant commonalities and capture the characteristics of the modality-specific, gradient reversal layer based adversarial classifier and modality reconstruction decoders are used during training. The gesture decoder generates proper gestures using all representations and features related to the rhythm in the audio. 

## 2. Videos

### 2.1 Ablation study

(GT / ReprGestrue / without Wavlm / without GAN loss / without domain loss / without Repr)

Please Click & Download：Ablation_study~3.mp4

### 2.2 Additional experiments

(GT / ReprGestrue / with diff loss / with text emotion / with diff loss and text emotion)

Please Click & Download：Additional_experiments~5.mp4

However, the results of these experiments did not turn out very well, so they were not mentioned in the final submission of the system or the paper.

## 3. Code and Pre-trained Model

### 3.1 Code

The code will be uploaded later.

### 3.2 Pre-trained Model

The pre-trained model will be uploaded later.

(_Double-anonymous Page For ICMI 2022 GENEA challenge_)
