import os
import json
import time
import math
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import librosa
except Exception:
    librosa = None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =============================
# Stable interfaces
# =============================

@dataclass
class TriggerEvent:
    source_file: str
    trigger_time_s: float
    peak_rms: float
    peak_flux: float
    zcr: float
    bandwidth_hz: float
    snr_db: float


@dataclass
class ConfirmationResult:
    source_file: str
    trigger_time_s: float
    is_gunshot: bool
    confidence: float
    inference_ms: float
    label: str = "gunshot"


@dataclass
class TimelineEvent:
    event_type: str
    source_file: str
    trigger_time_s: float
    clip_start_s: float
    clip_end_s: float
    shaman_i: Dict
    shaman_ii: Dict


class ShamanIPrefilter(Protocol):
    def detect(self, audio: np.ndarray, sr: int, source_file: str) -> List[TriggerEvent]:
        ...


class ShamanIIConfirm(Protocol):
    def fit(self, positive_clips: List[np.ndarray], sr: int) -> None:
        ...

    def confirm(self, clip: np.ndarray, sr: int, source_file: str, trigger_time_s: float) -> ConfirmationResult:
        ...


# =============================
# Audio helpers
# =============================

def require_audio_stack() -> None:
    if sf is None or librosa is None:
        raise RuntimeError(
            "This MVP needs soundfile and librosa installed. Install with: \n"
            "pip install soundfile librosa"
        )


def load_audio(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    require_audio_stack()
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    if np.max(np.abs(audio)) > 0:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio.astype(np.float32), sr


def extract_centered_clip(audio: np.ndarray, sr: int, center_s: float, clip_s: float = 3.0) -> Tuple[np.ndarray, float, float]:
    half = int((clip_s * sr) / 2)
    center = int(center_s * sr)
    start = max(0, center - half)
    end = min(len(audio), center + half)
    clip = audio[start:end]
    target_len = int(clip_s * sr)
    if len(clip) < target_len:
        pad_left = 0
        pad_right = target_len - len(clip)
        clip = np.pad(clip, (pad_left, pad_right))
    return clip.astype(np.float32), start / sr, min(len(audio), end) / sr


def sliding_windows(audio: np.ndarray, sr: int, win_s: float, hop_s: float) -> List[Tuple[int, int, float]]:
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    out = []
    for start in range(0, max(1, len(audio) - win + 1), hop):
        end = start + win
        if end > len(audio):
            break
        out.append((start, end, start / sr))
    return out


# =============================
# Shaman I: lightweight impulsive prefilter
# =============================

class BasicGunshotPrefilter:
    """
    Fast heuristic detector for impulsive events.
    Looks for sharp broadband transients using RMS, spectral flux,
    zero crossing rate, bandwidth, and local SNR.
    """

    def __init__(
        self,
        win_s: float = 0.064,
        hop_s: float = 0.016,
        rms_z_thresh: float = 2.5,
        flux_z_thresh: float = 2.0,
        snr_db_thresh: float = 8.0,
        min_gap_s: float = 0.40,
    ):
        self.win_s = win_s
        self.hop_s = hop_s
        self.rms_z_thresh = rms_z_thresh
        self.flux_z_thresh = flux_z_thresh
        self.snr_db_thresh = snr_db_thresh
        self.min_gap_s = min_gap_s

    def detect(self, audio: np.ndarray, sr: int, source_file: str) -> List[TriggerEvent]:
        require_audio_stack()
        n_fft = max(256, int(2 ** math.ceil(math.log2(self.win_s * sr))))
        hop_length = max(1, int(self.hop_s * sr))
        frame_length = max(1, int(self.win_s * sr))

        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length, center=True)[0]
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length, center=True)[0]
        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=frame_length))
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]

        flux = np.sqrt(np.sum(np.diff(S, axis=1, prepend=S[:, :1]) ** 2, axis=0))
        flux = flux[: len(rms)]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        rms_mu, rms_sd = float(np.mean(rms)), float(np.std(rms) + 1e-8)
        flux_mu, flux_sd = float(np.mean(flux)), float(np.std(flux) + 1e-8)

        # local noise floor from a rolling median
        local_floor = np.array([
            np.median(rms[max(0, i - 25): min(len(rms), i + 25)])
            for i in range(len(rms))
        ])
        snr_db = 20.0 * np.log10((rms + 1e-8) / (local_floor + 1e-8))

        triggers: List[TriggerEvent] = []
        last_t = -1e9
        for i, t in enumerate(times):
            rms_z = (rms[i] - rms_mu) / rms_sd
            flux_z = (flux[i] - flux_mu) / flux_sd
            impulsive = (
                rms_z >= self.rms_z_thresh
                and flux_z >= self.flux_z_thresh
                and snr_db[i] >= self.snr_db_thresh
                and bandwidth[i] >= 1200.0
            )
            if not impulsive:
                continue
            if t - last_t < self.min_gap_s:
                continue
            last_t = float(t)
            triggers.append(
                TriggerEvent(
                    source_file=source_file,
                    trigger_time_s=float(t),
                    peak_rms=float(rms[i]),
                    peak_flux=float(flux[i]),
                    zcr=float(zcr[i]),
                    bandwidth_hz=float(bandwidth[i]),
                    snr_db=float(snr_db[i]),
                )
            )
        return triggers


# =============================
# Shaman II: PyTorch confirmation model
# =============================

class TinyGunshotNet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchGunshotConfirm:
    """
    Demo-friendly PyTorch confirmation step.
    Trains a tiny MLP on log-mel summary features.

    Positives: 3-second centered clips from the provided gunshot files.
    Negatives: offset windows from the same audio away from the central event.
    This is a simple stand-in, intentionally swappable later.
    """

    def __init__(
        self,
        n_mels: int = 64,
        epochs: int = 25,
        batch_size: int = 16,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.n_mels = n_mels
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TinyGunshotNet] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

    def _clip_to_features(self, clip: np.ndarray, sr: int) -> np.ndarray:
        require_audio_stack()
        mel = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=self.n_mels, fmax=sr // 2)
        logmel = librosa.power_to_db(mel + 1e-8)
        delta = librosa.feature.delta(logmel)
        delta2 = librosa.feature.delta(logmel, order=2)
        feat = np.concatenate(
            [
                np.mean(logmel, axis=1),
                np.std(logmel, axis=1),
                np.mean(delta, axis=1),
                np.std(delta, axis=1),
                np.mean(delta2, axis=1),
                np.std(delta2, axis=1),
            ],
            axis=0,
        )
        return feat.astype(np.float32)

    def _make_negative_clips(self, positive_clips: List[np.ndarray], sr: int) -> List[np.ndarray]:
        negatives = []
        clip_len = len(positive_clips[0]) if positive_clips else int(3.0 * sr)
        for clip in positive_clips:
            # carve out likely background-only chunks from start / end of positive clip
            head = clip[:clip_len // 2]
            tail = clip[clip_len // 2:]
            if len(head) < clip_len:
                head = np.pad(head, (0, clip_len - len(head)))
            if len(tail) < clip_len:
                tail = np.pad(tail, (0, clip_len - len(tail)))
            negatives.append(head.astype(np.float32))
            negatives.append(tail.astype(np.float32))
        return negatives

    def fit(self, positive_clips: List[np.ndarray], sr: int) -> None:
        if len(positive_clips) < 4:
            raise ValueError("Need at least a few positive clips to fit Shaman II.")

        negative_clips = self._make_negative_clips(positive_clips, sr)

        X_pos = np.stack([self._clip_to_features(c, sr) for c in positive_clips], axis=0)
        X_neg = np.stack([self._clip_to_features(c, sr) for c in negative_clips], axis=0)
        y_pos = np.ones(len(X_pos), dtype=np.float32)
        y_neg = np.zeros(len(X_neg), dtype=np.float32)

        X = np.concatenate([X_pos, X_neg], axis=0)
        y = np.concatenate([y_pos, y_neg], axis=0)

        self.feature_mean = X.mean(axis=0, keepdims=True)
        self.feature_std = X.std(axis=0, keepdims=True) + 1e-6
        Xn = (X - self.feature_mean) / self.feature_std

        ds = TensorDataset(torch.tensor(Xn, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True)

        self.model = TinyGunshotNet(in_dim=Xn.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

    def confirm(self, clip: np.ndarray, sr: int, source_file: str, trigger_time_s: float) -> ConfirmationResult:
        if self.model is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Shaman II model is not fit yet.")
        x = self._clip_to_features(clip, sr)[None, :]
        x = (x - self.feature_mean) / self.feature_std
        xt = torch.tensor(x, dtype=torch.float32, device=self.device)
        self.model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            logit = self.model(xt)
            prob = torch.sigmoid(logit).item()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return ConfirmationResult(
            source_file=source_file,
            trigger_time_s=float(trigger_time_s),
            is_gunshot=bool(prob >= 0.5),
            confidence=float(prob),
            inference_ms=float(dt_ms),
        )



# Timeline assembly


def build_timeline(
    files: List[str],
    sr: int,
    shaman_i: ShamanIPrefilter,
    shaman_ii: ShamanIIConfirm,
) -> List[TimelineEvent]:
    events: List[TimelineEvent] = []
    for path in files:
        audio, sr_ = load_audio(path, sr)
        triggers = shaman_i.detect(audio, sr_, source_file=os.path.basename(path))
        for trig in triggers:
            clip, start_s, end_s = extract_centered_clip(audio, sr_, trig.trigger_time_s, clip_s=3.0)
            conf = shaman_ii.confirm(clip, sr_, source_file=os.path.basename(path), trigger_time_s=trig.trigger_time_s)
            events.append(
                TimelineEvent(
                    event_type="gunshot_candidate" if not conf.is_gunshot else "gunshot_confirmed",
                    source_file=os.path.basename(path),
                    trigger_time_s=trig.trigger_time_s,
                    clip_start_s=start_s,
                    clip_end_s=end_s,
                    shaman_i=asdict(trig),
                    shaman_ii=asdict(conf),
                )
            )
    events.sort(key=lambda x: (x.source_file, x.trigger_time_s))
    return events



# Training data bootstrap


def bootstrap_positive_clips(files: List[str], sr: int) -> List[np.ndarray]:
    """
    For a concise MVP, assume each gunshot file contains a salient gunshot event.
    Use the strongest impulsive trigger if found; otherwise fall back to center clip.
    """
    pre = BasicGunshotPrefilter()
    positive_clips: List[np.ndarray] = []
    for path in files:
        audio, sr_ = load_audio(path, sr)
        triggers = pre.detect(audio, sr_, source_file=os.path.basename(path))
        if triggers:
            trig = max(triggers, key=lambda t: t.peak_flux + t.snr_db)
            clip, _, _ = extract_centered_clip(audio, sr_, trig.trigger_time_s, clip_s=3.0)
        else:
            center_s = len(audio) / (2.0 * sr_)
            clip, _, _ = extract_centered_clip(audio, sr_, center_s, clip_s=3.0)
        positive_clips.append(clip)
    return positive_clips



# Main app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", type=str, required=True, help="Folder containing gunshot mp3 files.")
    ap.add_argument("--out_dir", type=str, default="gunshot_mvp_outputs")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--save_timeline_only", action="store_true")
    args = ap.parse_args()

    safe_dir = Path(args.out_dir)
    safe_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [str(p) for p in Path(args.audio_dir).glob("*.mp3")]
        + [str(p) for p in Path(args.audio_dir).glob("*.wav")]
    )
    if not files:
        raise FileNotFoundError(f"No mp3/wav files found in {args.audio_dir}")

    # Train Shaman II from the provided demo set.
    positive_clips = bootstrap_positive_clips(files, args.sr)
    shaman_ii = TorchGunshotConfirm()
    shaman_ii.fit(positive_clips, args.sr)

    # Run full two-stage pipeline.
    shaman_i = BasicGunshotPrefilter()
    timeline = build_timeline(files, args.sr, shaman_i, shaman_ii)

    out_timeline = [asdict(e) for e in timeline]
    with open(safe_dir / "ai_event_timeline.json", "w") as f:
        json.dump(out_timeline, f, indent=2)

    if not args.save_timeline_only:
        summary = {
            "n_files": len(files),
            "n_bootstrap_positive_clips": len(positive_clips),
            "n_timeline_events": len(timeline),
            "n_confirmed_gunshots": int(sum(e["shaman_ii"]["is_gunshot"] for e in out_timeline)),
            "files": [os.path.basename(x) for x in files],
        }
        with open(safe_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Done. Wrote timeline to {safe_dir / 'ai_event_timeline.json'}")


if __name__ == "__main__":
    main()
