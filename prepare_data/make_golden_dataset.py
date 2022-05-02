import os
import json
import jsonlines
import argparse
from tqdm import tqdm
from pprint import pprint
from phonemizer.backend import FestivalBackend
from phonemizer.separator import Separator


backend = FestivalBackend('en-us')
separator = Separator(phone='.', word=' ')


def get_label_set(data_list):
    scenario = {}
    action = {}
    # we did not experimeny on entity extraction in this work
    # entities = {}
    for i, data in enumerate(data_list):
        scenario[data['scenario']] = scenario.get(data['scenario'], 0) + 1
        action[data['action']] = action.get(data['action'], 0) + 1
    return scenario, action


def gen_meta(args):
    scenario = {}
    action = {}

    for split in ['train', 'devel', 'test']:
        data_file = os.path.join(args.slurp_dir, '{}.jsonl'.format(split))
        with jsonlines.open(data_file) as f:
            data_list = [line for line in f.iter()]
        scenario[split], action[split] = get_label_set(data_list)

    # unseen scenario: devel {likeness}, test {locations}
    # unseen action: None
    scenario = list(set(scenario['test']) | set(scenario['train']) | set(scenario['devel']))
    action = list(set(action['test']) | set(action['train']) | set(action['devel']))
    meta = {'scenario': scenario, 'action': action}
    meta['s2id'] = {k: i for i, k in enumerate(scenario)}
    meta['a2id'] = {k: i for i, k in enumerate(action)}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    return meta


def gen_data(data_json, meta, idx, dedup=False):
    # info_keys = ['slurp_id', 'sentence', 'action', 'scenario']
    data = []
    phonemes = backend.phonemize([data_json['sentence']], separator=separator)[0]

    for r in data_json['recordings']:
        sample = {
            'id': idx,
            'file': r['file'],
            'slurp_id': data_json['slurp_id'],
            'sentence': data_json['sentence'],
            'phonemes': phonemes,
            'scenario': meta['s2id'][data_json['scenario']],
            'action': meta['a2id'][data_json['action']],
        }
        idx += 1
        data.append(sample)
        if dedup:
            break
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slurp_dir", default='slurp/dataset/slurp', type=str, help="slurp annotation dir"
    )
    parser.add_argument(
        "--output_dir", default='datasets/slurp/golden', type=str, help="output dir"
    )
    parser.add_argument(
        "--force", action='store_true', help="overwrite meta & dataset"
    )
    parser.add_argument(
        "--dedup", action='store_true', help="single training sample for a sentence (multiple recordings)"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, 'meta.json')
    if not os.path.exists(meta_path) or args.force:
        meta = gen_meta(args)
    else:
        with open(meta_path, 'r') as f:
            meta = json.load(f)

    print("Number of labels:")
    print("scenario: {}".format(len(meta['scenario'])))
    print("action: {}".format(len(meta['action'])))

    dataset_path = os.path.join(args.output_dir, 'dataset.json')
    if not os.path.exists(dataset_path) or args.force:
        datasets = {}
        for split in ['train', 'devel', 'test']:
            data_file = os.path.join(args.slurp_dir, '{}.jsonl'.format(split))
            dataset = []
            with jsonlines.open(data_file) as f:
                data_list = [line for line in f.iter()]
            trange = tqdm(enumerate(data_list), total=len(data_list), desc=split)
            for _, data_json in trange:
                dataset.extend(gen_data(data_json, meta, len(dataset), args.dedup))
            datasets[split] = dataset

            # Debug
            print("====== {} ======".format(split))
            pprint(dataset[-3:])

        with open(dataset_path, 'w') as f:
            json.dump(datasets, f)
    else:
        print('dataset already exist')
        with open(dataset_path, 'r') as f:
            datasets = json.load(f)
    print("Dataset statistics:")
    for k, v in datasets.items():
        print("{}: {}".format(k, len(v)))
