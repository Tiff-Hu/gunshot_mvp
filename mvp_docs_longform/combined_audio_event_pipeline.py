from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from audio_event_common_task49 import (
    BasicBirdPrefilter,
    BasicGunshotPrefilter,
    BirdNETConfirm,
    TorchGunshotConfirm,
    bootstrap_negative_clips,
    bootstrap_positive_clips,
    build_longform_bird_timeline,
    build_longform_gunshot_timeline,
    build_node_result_payload,
    collect_audio_files,
    compare_events_to_ground_truth,
    merge_timeline_events,
    save_node_event_timelines_sqlite,
    write_timeline_json,
)


def run_combined_pipeline(
    gunshot_dir: str,
    negative_dir: str,
    input_audio: str,
    out_dir: str,
    gunshot_sr: int = 16000,
    bird_sr: int = 48000,
    gunshot_threshold: float = 0.5,
    birdnet_threshold: float = 0.5,
    clip_s: float = 3.0,
    block_seconds: float = 60.0,
    ground_truth_log: Optional[str] = None,
    gunshot_tolerance_s: float = 2.0,
    bird_tolerance_s: float = 5.0,
    run_id: Optional[str] = None,
    node_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    gunshot_files = collect_audio_files(gunshot_dir)
    negative_files = collect_audio_files(negative_dir)
    positive_clips = bootstrap_positive_clips(gunshot_files, gunshot_sr, top_k=3)
    negative_clips = bootstrap_negative_clips(negative_files, gunshot_sr, max_per_file=3)

    gunshot_model = TorchGunshotConfirm(threshold=gunshot_threshold)
    gunshot_model.fit(positive_clips, negative_clips, gunshot_sr)
    gunshot_prefilter = BasicGunshotPrefilter()
    bird_prefilter = BasicBirdPrefilter()
    birdnet = BirdNETConfirm(min_confidence=birdnet_threshold)

    gun_events = build_longform_gunshot_timeline(
        input_audio,
        gunshot_sr,
        gunshot_prefilter,
        gunshot_model,
        clip_s=clip_s,
        block_seconds=block_seconds,
    )
    bird_events = build_longform_bird_timeline(
        input_audio,
        bird_sr,
        bird_prefilter,
        birdnet,
        clip_s=clip_s,
        block_seconds=block_seconds,
    )
    combined_events = merge_timeline_events(gun_events, bird_events)

    write_timeline_json(gun_events, str(out_dir_p / "gunshot_timeline.json"))
    write_timeline_json(bird_events, str(out_dir_p / "bird_timeline.json"))
    write_timeline_json(combined_events, str(out_dir_p / "combined_ai_event_timeline.json"))

    evaluation: Dict[str, Any] = {}
    if ground_truth_log:
        evaluation["gunshot"] = compare_events_to_ground_truth(
            gun_events,
            ground_truth_log,
            event_type="GUNSHOT",
            tolerance_s=gunshot_tolerance_s,
            require_category_match=False,
        )
        evaluation["bird"] = compare_events_to_ground_truth(
            bird_events,
            ground_truth_log,
            event_type="BIRD",
            tolerance_s=bird_tolerance_s,
            require_category_match=True,
        )
        overall_tp = evaluation["gunshot"]["true_positives"] + evaluation["bird"]["true_positives"]
        overall_fp = evaluation["gunshot"]["false_positives"] + evaluation["bird"]["false_positives"]
        overall_fn = evaluation["gunshot"]["false_negatives"] + evaluation["bird"]["false_negatives"]
        precision = overall_tp / max(1, overall_tp + overall_fp)
        recall = overall_tp / max(1, overall_tp + overall_fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        evaluation["overall"] = {
            "true_positives": overall_tp,
            "false_positives": overall_fp,
            "false_negatives": overall_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        with open(out_dir_p / "evaluation.json", "w") as f:
            json.dump(evaluation, f, indent=2)

    payload = build_node_result_payload(
        node_id=str(node_id or Path(input_audio).stem),
        audio_path=input_audio,
        gunshot_events=gun_events,
        bird_events=bird_events,
        combined_events=combined_events,
        evaluation=evaluation,
    )
    with open(out_dir_p / "backend_payload.json", "w") as f:
        json.dump(payload, f, indent=2)

    summary = {
        "n_gunshot_events": len(gun_events),
        "n_bird_events": len(bird_events),
        "n_total_events": len(combined_events),
        "ground_truth_log": ground_truth_log,
        "node_id": node_id,
        "run_id": run_id,
    }
    if evaluation:
        summary["evaluation"] = evaluation
    with open(out_dir_p / "combined_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if db_path and run_id and node_id:
        save_node_event_timelines_sqlite(
            db_path=db_path,
            run_id=run_id,
            node_id=str(node_id),
            audio_path=input_audio,
            gunshot_events=gun_events,
            bird_events=bird_events,
            evaluation=evaluation,
        )

    return {
        "gunshot_events": gun_events,
        "bird_events": bird_events,
        "combined_events": combined_events,
        "evaluation": evaluation,
        "summary": summary,
        "backend_payload": payload,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Combined long-form gunshot + bird pipeline with validation and backend-friendly outputs.")
    ap.add_argument("--gunshot_dir", required=True)
    ap.add_argument("--negative_dir", required=True)
    ap.add_argument("--input_audio", required=True)
    ap.add_argument("--out_dir", default="combined_audio_outputs")
    ap.add_argument("--gunshot_sr", type=int, default=16000)
    ap.add_argument("--bird_sr", type=int, default=48000)
    ap.add_argument("--gunshot_threshold", type=float, default=0.5)
    ap.add_argument("--birdnet_threshold", type=float, default=0.5)
    ap.add_argument("--clip_s", type=float, default=3.0)
    ap.add_argument("--block_seconds", type=float, default=60.0)
    ap.add_argument("--ground_truth_log")
    ap.add_argument("--gunshot_tolerance_s", type=float, default=2.0)
    ap.add_argument("--bird_tolerance_s", type=float, default=5.0)
    ap.add_argument("--run_id")
    ap.add_argument("--node_id")
    ap.add_argument("--db_path")
    args = ap.parse_args()

    result = run_combined_pipeline(
        gunshot_dir=args.gunshot_dir,
        negative_dir=args.negative_dir,
        input_audio=args.input_audio,
        out_dir=args.out_dir,
        gunshot_sr=args.gunshot_sr,
        bird_sr=args.bird_sr,
        gunshot_threshold=args.gunshot_threshold,
        birdnet_threshold=args.birdnet_threshold,
        clip_s=args.clip_s,
        block_seconds=args.block_seconds,
        ground_truth_log=args.ground_truth_log,
        gunshot_tolerance_s=args.gunshot_tolerance_s,
        bird_tolerance_s=args.bird_tolerance_s,
        run_id=args.run_id,
        node_id=args.node_id,
        db_path=args.db_path,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
