# Simple skeleton visualization

This is a simple visualization tool that draws stick figures after converting BVH motion to 3D joint positions. We hope it will help you in the development process.

## Step-by-step instruction

* Run `$ python bvh_to_position_data.py`
  * Read BVH files, convert to 3D Cartesian positions, and save to NPY files.
* Run `$ python visualize_skeleton.py`
  * Read NPY files, draw stick figures (using Matplotlib), and save to MP4 videos.
  * Note that it takes several minutes to render a video.
* Run `$ python merge_video_audio.py`
  * Merge MP4 video and WAV audio files.
  * FFMPEG is needed. Please install FFMPEG tool `$ sudo apt install ffmpeg`

## Sample output
https://user-images.githubusercontent.com/9062897/168600590-bb4baa49-5318-4463-97a9-e4cbed37a899.mp4

## Notes

* This code is tested on Ubuntu 18.04, Python 3.6 and 3.8
* [PyMo](https://omid.al/projects/pymo/) is used to parse BVH and convert it to joint positions. Based on [Simon Alexanderson's forked version](https://github.com/simonalexanderson/PyMO), I’ve added RootNormalizer class. It calculates the mean root translation of each recording to translate the character to be mostly around the origin point. And it makes the character face the same direction by rotating 90 or -90 degrees.
