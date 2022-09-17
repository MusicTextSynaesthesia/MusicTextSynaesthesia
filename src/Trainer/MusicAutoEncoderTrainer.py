import os
import shutil
import torch
import torchaudio
import tqdm
from PIL import Image
from .BaseTrainer import Trainer


def norm(x):
    return (x - x.min()) / (x.max() - x.min())


def wave_sample(waveform, length):
    start = torch.randint(0, waveform.shape[1]-length, (1,)).item()
    return waveform[:, start: start+length]


class MusicAutoEncoderTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MusicAutoEncoderTrainer, self).__init__(*args, **kwargs)
        self.best_loss = float("inf")
        self.SpecTrans = torchaudio.transforms.Spectrogram(n_fft=510, normalized=True).cuda()
        self.InvSpec = torchaudio.transforms.GriffinLim(n_fft=510).cuda()
        self.criterion = torch.nn.MSELoss().cuda()

    def compute_loss(self, recon, refer):
        loss = self.criterion(recon, refer)
        loss += self.criterion(recon[:, :, 1:, :] - recon[:, :, :-1, :], refer[:, :, 1:, :] - refer[:, :, :-1, :])
        loss += self.criterion(recon[:, :, :, 1:] - recon[:, :, :, :-1], refer[:, :, :, 1:] - refer[:, :, :, :-1])
        return loss

    def train_step(self, sent_tokens, waveform, tags=None):
        try:
            length = torch.randint(20, min(waveform.shape[1]//64//255, 512), (1,)).item()*64
            waveform = wave_sample(waveform, 255*length-1)
            spectrogram = self.SpecTrans(waveform).unsqueeze(1)
            decoded = self.model(waveform)
            loss = self.compute_loss(decoded, spectrogram)
            loss.backward()
        except RuntimeError as e:
            print(e)
            return torch.tensor(0)
        return loss

    def valid(self, dataset, desc="Valid"):
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            p_bar = tqdm.tqdm(self.valid_dataset, desc=f"validation")
            os.makedirs(self.args.output_dir, exist_ok=True)
            for sent_tokens, waveform, pair in p_bar:
                decoded, loss = self.valid_step(sent_tokens, waveform.cuda(), pair)
                total_loss += loss
                p_bar.set_postfix(loss=loss)
            self.model.train()
        return {"epoch": self.epoch, "Loss": total_loss}

    def valid_step(self, sent_tokens, waveform, pair):
        length = min(waveform.shape[1]//64//255, 256)*64
        waveform = wave_sample(waveform, 255*length-1)
        spectrogram = self.SpecTrans(waveform).unsqueeze(1)
        decoded = self.model(waveform)
        loss = self.compute_loss(decoded, spectrogram)
        vid = pair["music_info"]['video_id']
        if vid[0] == "A":
            recon, refer = norm(decoded).squeeze(), norm(spectrogram).squeeze()
            Image.fromarray(refer.mul(255).detach().cpu().numpy()).convert("L").save(f"{self.args.output_dir}/refer.jpg")
            Image.fromarray(recon.mul(255).detach().cpu().numpy()).convert("L").save(f"{self.args.output_dir}/recon.jpg")
            recon, refer = self.InvSpec(recon.unsqueeze(0)), self.InvSpec(refer.unsqueeze(0))
            torchaudio.save(f"{self.args.output_dir}/{vid}_refer.ogg", refer.cpu(), self.args.sample_rate)
            torchaudio.save(f"{self.args.output_dir}/{vid}.ogg", recon.cpu(), self.args.sample_rate)
        return decoded, loss.item()

    def save_model(self, epoch_metrics):
        if self.args.save_last_checkpoint:
            torch.save(self.model.state_dict(), f"{self.args.model_save_dir}/LastModel.pkl")
        if epoch_metrics["Loss"] < self.best_loss:
            self.best_loss = epoch_metrics["Loss"]
            torch.save(self.model.state_dict(), f"{self.args.model_save_dir}/BestModel.pkl")
            shutil.rmtree(f"{self.args.output_dir}_best", ignore_errors=True)
            shutil.copytree(self.args.output_dir, f"{self.args.output_dir}_best")
        print(f"Epoch{self.epoch} loss: {epoch_metrics['Loss']} (best: {self.best_loss})")
        if epoch_metrics['Loss'] < 1 and self.model.add_layer_num():
            self.best_loss = float("inf")
