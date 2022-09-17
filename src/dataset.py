import json
import os
import re
import torch.utils.data
import torchaudio


def normalize(x: torch.Tensor): # normalize to [-1, 1]
    mi, ma = x.max(), x.min()
    return (x-mi)/(ma-mi)*2-1


instrument_dict = {None: 0, "string": 1, "cello": 1, "violin": 1, "viola": 1,
                   "piano": 2, "grand": 2, "clarinet": 3, "flute": 3, "wind": 3, "horn": 3}
ensemble_dict = {None: 0, "sonata": 1, "duetto": 0, "trio": 2, "quartet": 3, "quintet": 4,
                 "sextet": 4, "septet": 4, "octet": 4, "nonet": 4, "decet": 4}
tempo_dict = {None: 0, "grave": 1, "largo": 1, "lento": 1, "adagio": 1, "lent": 1, "kräftig": 1, "langsam": 1,
              "larghtto": 2, "andante": 2, "andantino": 2, "modera": 2, "allegretto": 2, "modéré": 2, "lebhaft": 2, "mäßig": 2,
              "allegro": 3, "vif": 3, "vite": 3, "rasch": 3, "schnell": 3, "vivace": 4, "presto": 4, "prestissimo": 4, "bewegt": 4}

mode_labels = ["None", "major", "minor"]
instrument_labels = ["None", "string", "piano", "wind"]
ensemble_labels = ["None", "sonata", "trio", "quartet", "above"]
tempo_labels = ["None", "slow", "medium", "fast", "faster"]

labels = {"mode": mode_labels, "instrument": instrument_labels, "ensemble": ensemble_labels, "tempo": tempo_labels}


def get_tags(curl_id, move_name):
    curl_id, move_name = curl_id.lower(), move_name.lower()
    mode = re.search("[a-g](-flat|-sharp)?-(major|minor)", curl_id)
    instrument = re.search("(string|piano|cello|violin|clarinet|flute|wind|grand|horn|viola)", curl_id)
    ensemble = re.search("(sonata|solo|duetto|trio|quartet|quintet|sextet|septet|octet|nonet|decet)", curl_id)
    tempo = re.search("(grave|largo|lento|adagio|larghtto|andante|andantino|modera|allegretto|allegro|vivace|presto|prestissimo|"
                      "lent|modéré|vif|vite|kräftig|langsam|lebhaft|mäßig|rasch|schnell|bewegt)", move_name)
    mode = 0 if mode is None else (1 if "major" in mode.group(0) else 2)
    instrument = instrument.group(0) if instrument else None
    ensemble = ensemble.group(0) if ensemble else None
    tempo = tempo.group(0) if tempo else None
    tags = {"mode": mode, "instrument": instrument_dict[instrument], "ensemble": ensemble_dict[ensemble], "tempo": tempo_dict[tempo]}
    return tags


class ClassicDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split, args):
        super(ClassicDataset, self).__init__()
        self.args = args
        self.path = args.dataset_dir
        self.Tokenizer = tokenizer
        self.sent_movement_pairs = []
        with open(f"{self.path}/music_text_pairs_sent_len_split.jsonl") as f:
            self.sent_movement_pairs = [json.loads(pair_json) for pair_json in f.readlines()]
            self.sent_movement_pairs = [pair for pair in self.sent_movement_pairs if pair["music_info"] != "N/A"]
        for pair in self.sent_movement_pairs:
            sent = re.sub("^(\\d)\\. ", "", pair["sent_text"])
            sent = re.sub(r"([Tt]he ?[\S]* )(opening|first|second|third|fourth|final) (movement)", r"\1\3", sent)
            sent = re.sub(r"([Tt]he ?[\S]* )(finale)", r"\1movement", sent)
            sent = re.sub(r" which follows", "", sent)
            pair["sent_text"] = sent
            pair["tags"] = get_tags(pair["music_id"], pair["music_info"]["name"])
        self.sent_movement_pairs = sum([self.sent_movement_pairs[s] for s in split], [])
        self.MuLawDecoder = torchaudio.transforms.MuLawDecoding(65536).cuda()

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            new_dataset = ClassicDataset(self.Tokenizer, [slice(None)], self.args)
            new_dataset.sent_movement_pairs = self.sent_movement_pairs[indices]
            return new_dataset
        sent_tokens = self.Tokenizer(self.sent_movement_pairs[indices]["sent_text"], return_tensors='pt')
        video_id = self.sent_movement_pairs[indices]["music_info"]["video_id"]
        waveform = torch.load(f'{self.path}/wave16000mulaw65536-32768/{video_id}.pkl').cuda()
        waveform = self.MuLawDecoder(waveform.long() + 32768)
        return sent_tokens, waveform, self.sent_movement_pairs[indices]

    def __len__(self):
        return len(self.sent_movement_pairs)
