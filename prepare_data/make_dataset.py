import os
import json
import numpy as np
import argparse
import jiwer
from tqdm import tqdm

from phonemizer.backend import FestivalBackend
from phonemizer.separator import Separator
backend = FestivalBackend('en-us')
separator = Separator(phone='.', word=' ')


asr_systems = ['google', 'wav2vec2']


def cal_wer(ground_truth, hypothesis):
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.SentencesToListOfWords(word_delimiter=" ")
    ])

    wer = jiwer.wer(
        ground_truth,
        hypothesis,
        truth_transform=transformation,
        hypothesis_transform=transformation
    )
    return wer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden_dir", default='datasets/slurp/golden', type=str, help="golden set dir"
    )
    parser.add_argument(
        "--transcript_dir", default='datasets/slurp/transcript/',
        type=str, help="transcript dir, may not have all transcripts"
    )
    parser.add_argument(
        "--output_path", default='datasets/slurp.json', type=str, help="output path"
    )
    parser.add_argument(
        "--phoneme", action='store_true', help="do phonemize"
    )
    parser.add_argument(
        "--force", action='store_true', help="overwrite dataset"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    golden_path = os.path.join(args.golden_dir, 'dataset.json')
    with open(golden_path, 'r') as f:
        golden = json.load(f)

    dataset_path = args.output_path
    if not os.path.exists(dataset_path) or args.force:

        """ make dictionary of key audio-filename and value golden_sample """
        fn_to_sample = {}
        for split in ['train', 'devel', 'test']:
            for sample in golden[split]:
                fn_to_sample[sample['file']] = sample
                fn_to_sample[sample['file']]['split'] = split
                fn_to_sample[sample['file']]['golden'] = sample['sentence']
                fn_to_sample[sample['file']]['golden_phonemes'] = sample['phonemes']

        datasets = {'train': [], 'devel': [], 'test': []}
        transcripts = {system: {} for system in asr_systems}
        wers = {system: [] for system in asr_systems}

        trange = tqdm(enumerate(fn_to_sample.keys()), total=len(fn_to_sample.keys()))
        for _, fn in trange:
            sample = fn_to_sample[fn]
            for asr_system in asr_systems:
                transcript_fn = fn.replace('.flac', '.txt')
                path = os.path.join(args.transcript_dir, asr_system, transcript_fn)
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    transcripts = f.readlines()
                if len(transcripts) == 0:
                    continue
                transcript = transcripts[0].replace('\n', '').lower()
                sample[asr_system] = {}
                sample[asr_system]['sentence'] = transcript
                sample[asr_system]['phonemes'] = ''
                if args.phoneme:
                    phonemes = backend.phonemize([transcript], separator=separator)
                    if len(phonemes) == 0:
                        continue
                    sample[asr_system]['phonemes'] = phonemes[0]
                sample[asr_system]['wer'] = cal_wer(sample['golden'], sample['sentence'])

                nbest_wer = []
                for t in transcripts:
                    nbest_wer.append(cal_wer(sample['golden'], t))
                wers[asr_system].append(sample[asr_system]['wer'])

            datasets[sample['split']].append(sample)

        """
        # Debug
        for split in ['train', 'devel', 'test']:
            print("====== {} ======".format(split))
            for i in range(2):
                for k, v in datasets[split][i].items():
                    print(split, i, "[{}]".format(k), v)
        """
        with open(dataset_path, 'w') as f:
            json.dump(datasets, f, indent=2)

        for asr_system in asr_systems:
            wer = wers[asr_system]
            print('system: {}, wer mean: {}, std: {}, max: {}, median: {}'.format(
                asr_system, np.mean(wer), np.std(wer), np.max(wer), np.median(wer)))

    else:
        print('dataset already exist')
        with open(dataset_path, 'r') as f:
            datasets = json.load(f)

    for k, v in datasets.items():
        print(k, len(v))
