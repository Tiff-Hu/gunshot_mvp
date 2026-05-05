from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from audio_event_common import (
    BasicGunshotPrefilter,
    TimelineEvent,
    TorchGunshotConfirm,
    bootstrap_negative_clips,
    bootstrap_positive_clips,
    collect_audio_files,
    extract_centered_clip_from_file,
    format_hms,
    stream_prefilter_file,
)


def _normalize_label(value: Optional[str]) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _event_category(event: Dict[str, Any], event_type: str) -> str:
    shaman_ii = event.get("shaman_ii", {}) or {}
    if event_type.upper() == "BIRD":
        return _normalize_label(shaman_ii.get("species"))
    return _normalize_label(shaman_ii.get("label", "gunshot"))


def _filter_confirmed(events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    if event_type.upper() == "GUNSHOT":
        return [e for e in events if e.get("event_type") == "gunshot_confirmed"]
    if event_type.upper() == "BIRD":
        return [e for e in events if e.get("event_type") == "bird_confirmed" and (e.get("shaman_ii", {}) or {}).get("species")]
    raise ValueError(f"Unsupported event type: {event_type}")


def compare_events_to_ground_truth(
    predicted_events: List[Dict[str, Any]],
    log_json_path: str,
    event_type: str,
    tolerance_s: float = 3.0,
    require_category_match: bool = False,
) -> Dict[str, Any]:
    with open(log_json_path, "r") as f:
        gt = json.load(f)

    expected_events = [e for e in gt.get("events", []) if str(e.get("type", "")).upper() == event_type.upper()]
    preds = sorted(_filter_confirmed(predicted_events, event_type), key=lambda x: float(x.get("trigger_time_s", 0.0)))
    exps = sorted(expected_events, key=lambda x: float(x.get("timestamp_seconds", 0.0)))

    matched_pred = set()
    matched_exp = set()
    matches: List[Dict[str, Any]] = []
    for i, pred in enumerate(preds):
        pred_t = float(pred.get("trigger_time_s", 0.0))
        pred_cat = _event_category(pred, event_type)
        best_j = None
        best_d = None
        for j, exp in enumerate(exps):
            if j in matched_exp:
                continue
            exp_t = float(exp.get("timestamp_seconds", 0.0))
            d = abs(pred_t - exp_t)
            if d > tolerance_s:
                continue
            if require_category_match:
                exp_cat = _normalize_label(exp.get("category"))
                if pred_cat and exp_cat and pred_cat != exp_cat:
                    continue
            if best_d is None or d < best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            matched_pred.add(i)
            matched_exp.add(best_j)
            matches.append(
                {
                    "predicted_time_s": pred_t,
                    "predicted_category": pred_cat,
                    "expected_time_s": float(exps[best_j].get("timestamp_seconds", 0.0)),
                    "expected_category": exps[best_j].get("category"),
                    "abs_error_s": float(best_d),
                }
            )

    tp = len(matched_pred)
    fp = len(preds) - tp
    fn = len(exps) - len(matched_exp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "event_type": event_type.upper(),
        "tolerance_seconds": tolerance_s,
        "require_category_match": require_category_match,
        "n_predictions": len(preds),
        "n_expected": len(exps),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def _build_backend_payload(node_id: str, audio_path: str, gunshot_events: List[Dict[str, Any]], evaluation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "node_id": node_id,
        "audio_path": audio_path,
        "gunshot_timeline": gunshot_events,
        "bird_timeline": [],
        "combined_timeline": gunshot_events,
        "evaluation": evaluation or {},
    }


def build_longform_gunshot_timeline(
    audio_file: str,
    sr: int,
    shaman_i: BasicGunshotPrefilter,
    shaman_ii: TorchGunshotConfirm,
    clip_s: float = 3.0,
    block_seconds: float = 60.0,
) -> list[TimelineEvent]:
    triggers = stream_prefilter_file(audio_file, shaman_i, block_seconds=block_seconds, overlap_seconds=clip_s)
    events: list[TimelineEvent] = []
    for trig in triggers:
        clip, clip_sr, start_s, end_s = extract_centered_clip_from_file(audio_file, trig.trigger_time_s, clip_s=clip_s, target_sr=sr)
        conf = shaman_ii.confirm(clip, clip_sr, source_file=os.path.basename(audio_file), trigger_time_s=trig.trigger_time_s)
        events.append(
            TimelineEvent(
                event_type="gunshot_confirmed" if conf.is_gunshot else "gunshot_candidate",
                source_file=os.path.basename(audio_file),
                trigger_time_s=trig.trigger_time_s,
                trigger_time_formatted=format_hms(trig.trigger_time_s),
                clip_start_s=start_s,
                clip_end_s=end_s,
                shaman_i=asdict(trig),
                shaman_ii=asdict(conf),
            )
        )
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description="Long-form gunshot detector with separate positive/negative training.")
    ap.add_argument("--gunshot_dir", required=True, help="Training folder with gunshot clips.")
    ap.add_argument("--negative_dir", required=True, help="Training folder with non-gunshot clips.")
    ap.add_argument("--input_audio", required=True, help="Long-form WAV/MP3/FLAC audio file to scan.")
    ap.add_argument("--out_dir", default="gunshot_longform_outputs")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--clip_s", type=float, default=3.0)
    ap.add_argument("--block_seconds", type=float, default=60.0)
    ap.add_argument("--ground_truth_log", type=str, default=None, help="Optional Himesh generator log JSON for evaluation.")
    ap.add_argument("--ground_truth_json", type=str, default=None, help="Alias for --ground_truth_log.")
    ap.add_argument("--gunshot_tolerance_s", type=float, default=2.0)
    ap.add_argument("--node_id", type=str, default=None)
    ap.add_argument("--run_id", type=str, default=None)
    args = ap.parse_args()
    if args.ground_truth_json and not args.ground_truth_log:
        args.ground_truth_log = args.ground_truth_json

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gunshot_files = collect_audio_files(args.gunshot_dir)
    negative_files = collect_audio_files(args.negative_dir)
    if not gunshot_files:
        raise FileNotFoundError(f"No training gunshot clips found under {args.gunshot_dir}")
    if not negative_files:
        raise FileNotFoundError(f"No training negative clips found under {args.negative_dir}")

    positive_clips = bootstrap_positive_clips(gunshot_files, args.sr, top_k=3)
    negative_clips = bootstrap_negative_clips(negative_files, args.sr, max_per_file=3)

    shaman_ii = TorchGunshotConfirm(threshold=args.threshold)
    shaman_ii.fit(positive_clips, negative_clips, args.sr)
    shaman_i = BasicGunshotPrefilter()

    timeline = build_longform_gunshot_timeline(args.input_audio, args.sr, shaman_i, shaman_ii, clip_s=args.clip_s, block_seconds=args.block_seconds)
    out_timeline = [asdict(e) for e in timeline]

    with open(out_dir / "gunshot_timeline.json", "w") as f:
        json.dump(out_timeline, f, indent=2)
    with open(out_dir / "bird_timeline.json", "w") as f:
        json.dump([], f, indent=2)
    with open(out_dir / "combined_ai_event_timeline.json", "w") as f:
        json.dump(out_timeline, f, indent=2)

    evaluation: Dict[str, Any] = {}
    if args.ground_truth_log:
        evaluation["gunshot"] = compare_events_to_ground_truth(
            out_timeline,
            args.ground_truth_log,
            event_type="GUNSHOT",
            tolerance_s=args.gunshot_tolerance_s,
            require_category_match=False,
        )
        evaluation["overall"] = {
            "true_positives": evaluation["gunshot"]["true_positives"],
            "false_positives": evaluation["gunshot"]["false_positives"],
            "false_negatives": evaluation["gunshot"]["false_negatives"],
            "precision": evaluation["gunshot"]["precision"],
            "recall": evaluation["gunshot"]["recall"],
            "f1": evaluation["gunshot"]["f1"],
        }
        with open(out_dir / "evaluation.json", "w") as f:
            json.dump(evaluation, f, indent=2)

    node_id = args.node_id or Path(args.input_audio).stem
    payload = _build_backend_payload(node_id=node_id, audio_path=args.input_audio, gunshot_events=out_timeline, evaluation=evaluation)
    with open(out_dir / "backend_payload.json", "w") as f:
        json.dump(payload, f, indent=2)

    summary = {
        "n_gunshot_events": len(out_timeline),
        "n_bird_events": 0,
        "n_total_events": len(out_timeline),
        "ground_truth_log": args.ground_truth_log,
        "node_id": args.node_id,
        "run_id": args.run_id,
        "n_training_gunshot_files": len(gunshot_files),
        "n_training_negative_files": len(negative_files),
        "n_bootstrap_positive_clips": len(positive_clips),
        "n_bootstrap_negative_clips": len(negative_clips),
        "n_candidates": int(sum(1 for e in out_timeline if e["event_type"] == "gunshot_candidate")),
        "n_confirmed_gunshots": int(sum(1 for e in out_timeline if e["event_type"] == "gunshot_confirmed")),
    }
    if evaluation:
        summary["evaluation"] = evaluation
    with open(out_dir / "combined_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
