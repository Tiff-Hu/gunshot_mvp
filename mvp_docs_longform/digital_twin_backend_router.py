from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from combined_audio_event_pipeline_task49 import run_combined_pipeline


class NodeAudioConfig(BaseModel):
    node_id: str = Field(..., description="Node identifier used by the Digital Twin app")
    audio_path: str = Field(..., description="Local path to the node audio WAV on the same machine as the backend")
    ground_truth_log: Optional[str] = Field(None, description="Optional local path to Himesh's answer key JSON for this node")


class DetectionRunRequest(BaseModel):
    run_id: str
    gunshot_dir: str
    negative_dir: str
    out_root: str = "audio_detection_outputs"
    db_path: str = "digital_twin_audio.db"
    nodes: List[NodeAudioConfig]
    gunshot_sr: int = 16000
    bird_sr: int = 48000
    gunshot_threshold: float = 0.5
    birdnet_threshold: float = 0.5
    clip_s: float = 3.0
    block_seconds: float = 60.0
    gunshot_tolerance_s: float = 2.0
    bird_tolerance_s: float = 5.0


router = APIRouter(prefix="/audio-detection", tags=["audio-detection"])


@router.post("/runs/process")
def process_run_audio_detection(req: DetectionRunRequest) -> Dict[str, Any]:
    if not req.nodes:
        raise HTTPException(status_code=400, detail="At least one node audio path must be provided.")

    out_root = Path(req.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    node_results: List[Dict[str, Any]] = []
    for node in req.nodes:
        if not Path(node.audio_path).exists():
            raise HTTPException(status_code=400, detail=f"Audio path does not exist: {node.audio_path}")
        if node.ground_truth_log and not Path(node.ground_truth_log).exists():
            raise HTTPException(status_code=400, detail=f"Ground truth log does not exist: {node.ground_truth_log}")

        node_out_dir = out_root / req.run_id / str(node.node_id)
        result = run_combined_pipeline(
            gunshot_dir=req.gunshot_dir,
            negative_dir=req.negative_dir,
            input_audio=node.audio_path,
            out_dir=str(node_out_dir),
            gunshot_sr=req.gunshot_sr,
            bird_sr=req.bird_sr,
            gunshot_threshold=req.gunshot_threshold,
            birdnet_threshold=req.birdnet_threshold,
            clip_s=req.clip_s,
            block_seconds=req.block_seconds,
            ground_truth_log=node.ground_truth_log,
            gunshot_tolerance_s=req.gunshot_tolerance_s,
            bird_tolerance_s=req.bird_tolerance_s,
            run_id=req.run_id,
            node_id=node.node_id,
            db_path=req.db_path,
        )
        node_results.append({
            "node_id": node.node_id,
            "audio_path": node.audio_path,
            "out_dir": str(node_out_dir),
            "summary": result["summary"],
            "evaluation": result["evaluation"],
        })

    manifest = {"run_id": req.run_id, "nodes_processed": len(node_results), "results": node_results}
    with open(out_root / req.run_id / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
