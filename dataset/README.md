# Dataset and data processing

This page contains the main information you need to know about the GENEA Challenge 2022 dataset and data processing.

## Release schedule

The dataset available at the start of the challenge is a beta version. It is missing a subset of the transcriptions (TSV files) and does not contain a validation set.

The full version of the data, including all transcriptions and a validation set, will be made available two weeks after the start of the challenge. That version will also include a document representing the manual used by the transcribers. (A Dockerised version of the official visualisation script will also be released at the same time.)

The inputs (audio and transcripts) for the challenge evaluation will be made available in late June. They will follow the same format as the validation set, and have been selected using the same criteria, except that motion data (BVH) is not included.

## Data

The data for the GENEA Challenge 2022 has been adapted from [Talking With Hands 16.2M](https://github.com/facebookresearch/TalkingWithHands32M). It comes from recordings of dyadic interactions between different speakers. For the 2022 challenge, each dyad has been separated into two independent sides with one speaker each, and generating interlocutor-aware behaviour is not part of the challenge.

### Folder structure and information about individual file formats

The challenge training data, which you were given access to in the folder `trn/`, contains three subfolders, each distributed as a separate ZIP file:

`wav/` (audio): Recorded audio in WAV format from a speaking and gesticulating actor, recorded with a close-talking microphone. Each file is mono and contains the audio from a single person, but different files can be different actors. Parts of the audio recordings have been muted to omit personally identifiable information. You can identify these portions by looking for places where the audio is completely silent (completely zeroed out).

`tsv/` (text): Word-level time-aligned text transcriptions of the above audio recordings in TSV format (tab-separated values). For privacy reasons, the transcriptions do not include references to personally identifiable information, similar to the audio files. Each line in a TSV file contains three fields, separated by tabs: the start time of a time interval (in seconds), the end time of that interval (in seconds), and a text transcription of the speaker's speech in the time interval. The transcriptions have been manually edited to improve their suitability for automatic text processing, e.g., by writing out ambiguous contractions, correcting grammar mistakes, and adding punctuation; see the annotation manual. As a result of this processing, some annotations contain more than one word.

`bvh/` (motion): Time-aligned 3D full-body motion-capture data in BVH format from a speaking and gesticulating actor. Each file is a single person, but different files can be different actors. Some files include finger motion capture, whilst others do not (see the section on metadata below). Files without finger-motion data will generally have fixed rotations for the finger joints, corresponding to a T-pose. Be aware that, since the data comes from dyadic interactions, there are two positions where the actors can stand, such that they are facing each other during the conversation. These positions have different global positions and orientations in the data.

### Filenames and metadata

The files in these three directories follow the same naming convention, `trn_2022_v0_NNN.*`, where `NNN` at the end of the prefix is a three-digit recording number with leading zeroes (`000` and up). Files with the same prefix correspond to the same person and recording. Each recording is approximately one minute or longer.

There is also a file `trn_2022_v0_metainformation.csv` with metadata about all the recording in the training set, one line per recording. Each line contains three fields, separated by commas: the filename prefix of the specific recording, an indicator of whether or not that file contains motion capture data for the fingers, and a numeric speaker ID. Every actor in the data has a unique two-digit numeric ID, and every recording tagged with that speaker ID corresponds to data from the same actor.

### Validation and test data

The validation data uses a similar organisation and folder structure, but with all strings `trn` replaced by `val`. Validation and test data will only contain speakers that also are part of the training data.

## Data processing scripts

We provide a number of optional scripts for encoding and processing the challenge data:

Audio: Scripts for extracting basic audio features, such as spectrograms, prosodic features, and mel-frequency cepstral coefficients (MFCCs) can be found [at this link](https://github.com/genea-workshop/Speech_driven_gesture_generation_with_autoencoder/blob/GENEA_2022/data_processing/tools.py#L105).

Text: A script to encode text transcriptions to word vectors using BERT or FastText will be provided shortly.

Motion: If you wish to encode the joint angles from the BVH files to and from an exponential map representation, you can use scripts by [Simon Alexanderson](https://github.com/simonalexanderson) based on the [PyMo library](https://omid.al/projects/pymo/), which are available here:
* [`bvh2features.py`](https://github.com/genea-workshop/Speech_driven_gesture_generation_with_autoencoder/blob/GENEA_2022/data_processing/bvh2features.py)
* [`features2bvh.py`](https://github.com/genea-workshop/Speech_driven_gesture_generation_with_autoencoder/blob/GENEA_2022/data_processing/features2bvh.py)

We provide code for several baseline systems via the challenge [`baselines` folder](https://github.com/genea-workshop/genea_challenge_2022/tree/main/baselines). These baselines also contain some data processing scripts.

If anything is unclear, please let us know!
