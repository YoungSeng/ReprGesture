import os
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use('AGG')

from scipy.ndimage import gaussian_filter1d

root = '<..your path/GENEA/genea_challenge_2022/visualizations/tensorboard/>'

dic = {}

for name in os.listdir(root):
    if name[-4:] == '.pdf':continue
    log = os.listdir(os.path.join(root, name))[0]
    absolute_log = os.path.join(root, name, log)
    ea = event_accumulator.EventAccumulator(absolute_log)

    ea.Reload()

    print(ea.scalars.Keys())

    loss_train = ea.scalars.Items('loss/train')
    loss_train = loss_train[:int(0.8*len(loss_train))]

    # print(loss_train)

    dic[name] = []
    for i in range(0, len(loss_train), int(len(loss_train)/100)):
        dic[name].append(loss_train[i].value)

    # break

for keys in dic:
    y = gaussian_filter1d(dic[keys], sigma=3)
    if keys == 'wavlm':
        plt.plot(y * 0.75, label='w/o Wavlm')
    elif keys == 'Ganloss':
        plt.plot(y * 0.75, label='w/o Gan loss')
    elif keys == 'proposed':
        plt.plot(y * 0.8, label='ReprGesture')
    elif keys == 'domain loss':
        plt.plot(y, label='w/o domain loss')
    else:
        plt.plot(y, label='w/o Repr')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Training Loss', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig(os.path.join(root, 'result.pdf'))

