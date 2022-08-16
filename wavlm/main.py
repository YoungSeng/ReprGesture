import torch
from WavLM import WavLM, WavLMConfig

# load the pre-trained checkpoints
checkpoint = torch.load('/path/to/wavlm.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

# extract the representation of last layer
wav_input_16khz = torch.randn(1,10000)
rep = model.extract_features(wav_input_16khz)[0]

# extract the representation of each layer
wav_input_16khz = torch.randn(1,10000)
rep, layer_results = model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
layer_reps = [x.transpose(0, 1) for x, _ in layer_results]

print(len(layer_reps))
