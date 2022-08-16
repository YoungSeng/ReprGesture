# Copyright 2020 by Patrik Jonell.
# All rights reserved.
# This file is part of the GENEA visualizer,
# and is released under the GPLv3 License. Please see the LICENSE
# file that should have been included as part of this package.


import requests
from pathlib import Path
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('bvh_file', type=Path)
parser.add_argument('-m', "--visualization_mode", help='The visualization mode to use for rendering.',type=str, choices=['full_body', 'upper_body'], default='full_body', required=True)
parser.add_argument('-s', '--server_url', default="http://localhost:5001")
parser.add_argument('-a', '--audio_file', help="The filepath to a chosen .wav audio file.", type=Path)
parser.add_argument('-r', '--rotate', help='Set to "cw" to rotate avatar 90 degrees clockwise, "ccw" for 90 degrees counter-clockwise, "flip" for 180-degree rotation, and leave at "default" for no rotation (or ignore the flag).', type=str, choices=['default', 'cw', 'ccw', 'flip'], default='default')
parser.add_argument('-o', '--output', help='The file path for the rendered .MP4 file from the server. If not specified, will use the directory of the supplied BVH file.', type=Path)

args = parser.parse_args()

server_url = args.server_url
bvh_file = args.bvh_file
audio_file = args.audio_file
output = args.output if args.output else bvh_file.with_suffix(".mp4")

headers = {"Authorization": "Bearer j7HgTkwt24yKWfHPpFG3eoydJK6syAsz"}

files = {"bvh_file": (bvh_file.name, bvh_file.open())}
if audio_file:
	files['audio_file'] = (audio_file.name, audio_file.open('rb'))

req_data = {'p_rotate': args.rotate, 'visualization_mode': args.visualization_mode}

try:
	print("Connecting to server...")
	render_request = requests.post(
		f"{server_url}/render",
		params=req_data,
		files=files,
		headers=headers,
		timeout=10
	)
except requests.Timeout:
	print("The request to the server timed out. Either the supplied server URL ({}) is incorrect, your firewall/router settings are blocking traffic between your system and the server, or the server is down.".format(args.server_url))
	exit()

print("Got response from server.")
job_uri = render_request.text

done = False
while not done:
	resp = requests.get(server_url + job_uri, headers=headers)
	resp.raise_for_status()

	response = resp.json()
	
	if response["state"] == "PENDING":
		jobs_in_queue = response["result"]["jobs_in_queue"]
		print(f"pending.. {jobs_in_queue} jobs currently in queue")
	
	elif response["state"] == "PROCESSING":
		print("Processing the file (this can take a while depending on file size)")
	
	elif response["state"] == "RENDERING":
		current = response["result"]["current"]
		total = response["result"]["total"]
		print(f"Rendering BVH: {int(current/total*100)}% done ({current}/{total} frames)")

	elif response["state"] == "COMBINING A/V":
		print(f"Combining audio with video. Your video will be ready soon!")

	elif response["state"] == "SUCCESS":
		file_url = response["result"]
		done = True
		print("Done!")
		break

	elif response["state"] == "FAILURE":
		raise Exception(response["result"])
	else:
		print(response)
		raise Exception("should not happen..")
	time.sleep(5)


video = requests.get(server_url + file_url, headers=headers).content
output.write_bytes(video)
