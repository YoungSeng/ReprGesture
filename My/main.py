import os

from config.parse_args import parse_args

_args = parse_args()

def main(config):
    args = config['args']
    print(len(args.data_mean))      # 216
    print(len(args.data_std))



if __name__ == '__main__':
    '''
    python main.py --config=<..your path/GENEA/genea_challenge_2022/baselines/My/config/seq2seq.yml>
    '''
    _args = parse_args()
    main({'args': _args})
