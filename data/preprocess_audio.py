import tqdm
import json
import multiprocessing
import torch
import torchaudio


def load_music(vid):
    return torchaudio.load(f'split/{vid}.ogg')


if __name__ == "__main__":
    with open("music_text_pairs_sent_len_split.jsonl") as f:
        sent_movement_pairs = [json.loads(pair_json) for pair_json in f.readlines()]
        sent_movement_pairs = [pair for pair in sent_movement_pairs if pair["music_info"] != "N/A"]
        all_vid_list = [pair["music_info"]["video_id"] for pair in sent_movement_pairs]
    print(len(all_vid_list))
    with multiprocessing.Pool(8) as pool:
        for i in tqdm.trange(0, len(all_vid_list), 32):
            vid_list = all_vid_list[i:i+32]
            src = pool.map(load_music, vid_list)
            files, wavelist = [], []
            for vid, (waveform, sample_rate) in zip(vid_list, src):
                waveform = waveform.mean(dim=0, keepdim=True).cuda()
                waveform = torchaudio.transforms.Resample(sample_rate, 16000).cuda()(waveform)
                files.append(vid)
                wavelist.append(waveform.cpu())
            for fn, w in list(zip(files, wavelist)):
                torch.save((torchaudio.functional.mu_law_encoding(w, 65536)-32768).short(), f"wave16000mulaw65536-32768/{fn}.pkl")
