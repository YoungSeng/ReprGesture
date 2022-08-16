import glob
import subprocess


# def merge(mp4_path, wav_path):
#     out_path = mp4_path.replace('.mp4', '_with_audio.mp4')
#     cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', mp4_path, '-i', wav_path, '-strict', '-2', out_path, '-shortest']
#     subprocess.call(cmd)
#
#
# if __name__ == '__main__':
#     files = sorted([f for f in glob.iglob('data_path/*.mp4')])
#     for i, mp4_path in enumerate(files):
#         print(mp4_path)
#         wav_path = mp4_path.replace('.mp4', '.wav')
#         merge(mp4_path, wav_path)

def merge(mp4_path, wav_path):
    out_path = mp4_path.replace('.mp4', '_with_audio.mp4')
    cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', mp4_path, '-i', wav_path, '-strict', '-2', out_path, '-shortest']
    subprocess.call(cmd)


if __name__ == '__main__':
    mp4_path = "<..your path/GENEA/genea_challenge_2022/baselines/Tri/output/infer_sample/Sheet1_generated_0000-0220.mp4>"
    wav_path = "<..your path/e2e/output2/infer_sample/google_TTS.wav>"
    merge(mp4_path, wav_path)
