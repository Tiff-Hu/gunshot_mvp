# Digital Twin MVP Audio Processing

This repository contains the local audio-processing work for the Digital Twin MVP. It now supports both the earlier standalone long-form detectors and the newer FastAPI-connected one-node workflow.

The current primary workflow is:

```text
single long node audio file
  -> stage 1 event candidate detection
  -> stage 2 TinyCNN bird-call gate
  -> stage 3 BirdNET species confirmation
  -> stage 4 human-presence adapter
  -> stage 5 backend timelines and dashboard tables
```

Each run processes one long audio file per node. To process multiple nodes, submit multiple node audio paths to the backend endpoint or rerun the standalone script once per node.

## Current Status

- The staged pipeline framework is implemented in `node_audio_workflow.py`.
- The TinyCNN architecture and mel preprocessing from `cnn_tiny.ipynb` / `TinyCNN_0.3_training.ipynb` are implemented in `tiny_cnn_birdcall.py`.
- The FastAPI backend integration is implemented in `backend/app/routes/audio_processing.py`.
- The backend registers the audio router in `backend/app/main.py`.
- The UI/backend field checklist lives in `REQUIRED_STAGE_INPUTS.md`.
- BirdNET is supported when `birdnetlib` and TensorFlow are installed.
- TinyCNN can execute without a weights file using a deterministic fallback, but real TinyCNN inference requires Griffen's trained `.pth` weights.
- Human presence is currently a schema-stable adapter until the final model artifact and exact feature order are provided.

## Important Files

### New staged workflow

- `node_audio_workflow.py`: Orchestrates one node audio file through all current stages.
- `tiny_cnn_birdcall.py`: TinyCNN model architecture, log-mel preprocessing, weight loading, and fallback gate.
- `simplify_pipeline_results.py`: Converts an existing pipeline output folder into simplified final human-presence results.
- `REQUIRED_STAGE_INPUTS.md`: Complete current list of inputs and metadata needed across the five stages.

### FastAPI backend

- `backend/app/main.py`: FastAPI app entry point.
- `backend/app/routes/audio_processing.py`: Audio workflow API endpoints.
- `backend/app/db_models.py`: Existing SQLAlchemy models, including `AIEventRow`.
- `backend/requirements.txt`: Core backend plus local audio workflow dependencies.
- `backend/requirements-audio-birdnet.txt`: Optional BirdNET/TensorFlow dependencies.

### Earlier standalone pipelines

- `combined_audio_event_pipeline.py`: Mixed bird and gunshot long-form detector.
- `bird_longform_birdnet_pipeline.py`: Bird-only long-form detector.
- `gunshot_mvp_separate_training_longform.py`: Gunshot-only long-form detector.
- `audio_event_common.py`: Shared audio/timeline utilities.

The `*_task49.py` files are Task 49 variants retained for compatibility.

## Environment Setup

Recommended conda environment:

```bash
conda create -n birdstates python=3.11 -y
conda activate birdstates
```

Install backend and audio workflow dependencies:

```bash
cd /Users/tiff1101/Desktop/digital_twin_mvp_docs
python3 -m pip install -r backend/requirements.txt
```

Optional BirdNET install:

```bash
python3 -m pip install -r backend/requirements-audio-birdnet.txt
```

On macOS, if you see an OpenMP duplicate runtime error, set these before running Python, Uvicorn, or the workflow:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

If audio decoding warns about `ffmpeg`, install it with Homebrew:

```bash
brew install ffmpeg
```

## TinyCNN Weights

The TinyCNN code is implemented, but no trained `.pth` file is currently present in this project folder.

The notebook references a saved model at:

```text
/Users/qian/KWF/best_tinycnn_thresh03.pth
```

That file is not included here. Until a trained weights file is provided, the TinyCNN stage uses a deterministic fallback and marks events like this:

```json
"model_status": "fallback_no_weights"
```

To use real TinyCNN inference, place the `.pth` file somewhere local, for example:

```text
/Users/tiff1101/Desktop/digital_twin_mvp_docs/models/best_tinycnn_thresh03.pth
```

Then pass it as:

```json
"tinycnn_weights": "/Users/tiff1101/Desktop/digital_twin_mvp_docs/models/best_tinycnn_thresh03.pth"
```

## Training TinyCNN Weights

If Griffen's trained weights are unavailable, `train_tinycnn.py` can train a compatible checkpoint from labeled folders:

```text
training_data/
|-- birdcall/
|   |-- bird_001.wav
|   `-- bird_002.wav
`-- not_birdcall/
    |-- noise_001.wav
    `-- noise_002.wav
```

Train from scratch:

```bash
python3 train_tinycnn.py \
  --positive_dir training_data/birdcall \
  --negative_dir training_data/not_birdcall \
  --output models/tinycnn_finetuned.pth \
  --epochs 25 \
  --batch_size 32 \
  --threshold 0.3
```

Fine-tune from an existing TinyCNN checkpoint:

```bash
python3 train_tinycnn.py \
  --positive_dir training_data/birdcall \
  --negative_dir training_data/not_birdcall \
  --initial_weights models/best_tinycnn_thresh03.pth \
  --output models/tinycnn_finetuned.pth \
  --epochs 10 \
  --batch_size 32 \
  --threshold 0.3
```

The output `.pth` stores `model_state_dict`, normalization values, training history, and dataset stats. The node workflow can load it directly with `--tinycnn_weights`.

## Standalone Node Workflow

Run this from the repository root:

```bash
cd /Users/tiff1101/Desktop/digital_twin_mvp_docs
conda activate birdstates

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

python3 node_audio_workflow.py \
  --input_audio "birdcalls/XC1085829 - Scarlet Macaw - Ara macao.wav" \
  --out_dir test_node_workflow_outputs \
  --node_id test_node_001 \
  --run_id test_run_001 \
  --block_seconds 10 \
  --skip_birdnet
```

Remove `--skip_birdnet` only after `birdnetlib` is installed and importable.

Expected output files:

- `stage_1_event_candidates.json`
- `stage_2_tinycnn_birdcall_timeline.json`
- `stage_3_birdnet_timeline.json`
- `stage_4_human_presence_timeline.json`
- `node_combined_timeline.json`
- `backend_payload.json`
- `node_run_summary.json`
- `simplified_results/final_human_presence_results.json`
- `simplified_results/final_human_presence_results.txt`

Inspect the summary:

```bash
cat test_node_workflow_outputs/node_run_summary.json
```

## FastAPI Backend Usage

Start the backend server in one terminal:

```bash
cd /Users/tiff1101/Desktop/digital_twin_mvp_docs
conda activate birdstates

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

python3 -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8001
```

Leave that terminal running. In a second terminal, test the audio endpoint:

```bash
cd /Users/tiff1101/Desktop/digital_twin_mvp_docs
conda activate birdstates

curl -X POST http://127.0.0.1:8001/api/audio/workflow/run \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "test_api_run_001",
    "nodes": [
      {
        "node_id": "test_node_001",
        "audio_path": "birdcalls/XC1085829 - Scarlet Macaw - Ara macao.wav"
      }
    ],
    "out_root": "test_api_audio_outputs",
    "block_seconds": 10,
    "skip_birdnet": true,
    "persist_to_db": false
  }'
```

Expected output folder:

```bash
open test_api_audio_outputs/test_api_run_001/test_node_001
```

The simplified final human-presence output will be in:

```bash
open test_api_audio_outputs/test_api_run_001/test_node_001/simplified_results
```

## FastAPI Web Views

With the backend running:

- API docs: <http://127.0.0.1:8001/docs>
- Health check: <http://127.0.0.1:8001/health>
- Required workflow inputs: <http://127.0.0.1:8001/api/audio/workflow/inputs>

## API Endpoints

### `POST /api/audio/workflow/run`

Runs local audio processing for one or more node audio files.

Example request:

```json
{
  "run_id": "run_001_audio",
  "db_run_id": 1,
  "nodes": [
    {
      "node_id": "N1",
      "audio_path": "/absolute/path/to/node_001.wav",
      "metadata_json": "/absolute/path/to/node_001_metadata.json"
    }
  ],
  "out_root": "audio_workflow_outputs",
  "tinycnn_weights": "/absolute/path/to/best_model.pth",
  "tinycnn_threshold": 0.3,
  "birdnet_threshold": 0.5,
  "human_presence_threshold": 0.5,
  "block_seconds": 60,
  "clip_s": 3.0,
  "skip_birdnet": false,
  "persist_to_db": true
}
```

If `db_run_id` is supplied and `persist_to_db` is true, confirmed audio events are written into the existing dashboard tables:

- `ai_events`
- `node_events`
- `detections_by_type`
- `run_metrics`
- `network_nodes.ai_det`

### `GET /api/audio/workflow/inputs`

Returns the current required stage-input checklist from `REQUIRED_STAGE_INPUTS.md` so the frontend can map UI fields to model inputs.

## Backend Output Shape

The workflow response includes:

- `run_id`
- `db_run_id`
- `nodes_processed`
- one result per node:
  - `node_id`
  - `audio_path`
  - `out_dir`
  - `summary`

The `backend_payload.json` written for each node includes:

- `stage_timelines.stage_1_event_candidates`
- `stage_timelines.stage_2_tinycnn_birdcall`
- `stage_timelines.stage_3_birdnet`
- `stage_timelines.stage_4_human_presence`
- `combined_timeline`
- `summary`

The `simplified_results/` folder written for each node includes:

- `final_human_presence_results.txt`: one line per final reported case, for example `Human presence detected at time 00:03:18.300.`
- `final_human_presence_results.json`: compact structured version with time, confidence, and boolean `human_presence_detected`

To generate simplified results for an existing output directory:

```bash
python3 simplify_pipeline_results.py combined_outputs_sample
```

## Earlier Long-Form Pipelines

The earlier standalone scripts are still available.

### Combined bird and gunshot

```bash
conda activate birdstates

python3 combined_audio_event_pipeline.py \
  --input_audio "/path/to/node_001_12hr_20260407_152944.wav" \
  --out_dir "/path/to/combined_outputs" \
  --ground_truth_log "/path/to/node_001_12hr_20260407_152944_log.json" \
  --gunshot_dir "/path/to/gunshot_data" \
  --negative_dir "/path/to/non_gunshot_data"
```

### Bird-only

```bash
conda activate birdstates

python3 bird_longform_birdnet_pipeline.py \
  --input_audio "/path/to/node_001_12hr_20260414_121347.wav" \
  --out_dir "/path/to/bird_only_outputs" \
  --ground_truth_log "/path/to/node_001_12hr_20260414_121347_log.json"
```

### Gunshot-only

```bash
conda activate birdstates

python3 gunshot_mvp_separate_training_longform.py \
  --gunshot_dir "/path/to/gunshot_data" \
  --negative_dir "/path/to/non_gunshot_data" \
  --input_audio "/path/to/node_001_12hr_20260414_121403.wav" \
  --out_dir "/path/to/gunshot_only_outputs" \
  --ground_truth_log "/path/to/node_001_12hr_20260414_121403_log.json"
```

Typical outputs from these older workflows:

- `bird_timeline.json`
- `gunshot_timeline.json`
- `combined_ai_event_timeline.json`
- `evaluation.json`
- `backend_payload.json`
- `combined_run_summary.json`
- `run_summary.json`

## Evaluation Logic

The older bird/gunshot pipelines can compare predictions against generator answer-key JSON logs.

Gunshots:

- event type: `GUNSHOT`
- default tolerance: `2.0` seconds
- category match is not required

Birds:

- event type: `BIRD`
- default tolerance: `5.0` seconds
- category/species match is required

## Open Items

- Get Griffen's final TinyCNN `.pth` weights and any training normalization constants.
- Confirm whether BirdNET should use location/date filtering in production.
- Replace the human-presence adapter with the final model artifact.
- Confirm the exact human-presence inference feature order.
- Confirm whether human speech is a separate detector or part of the human-presence model.

## Repository Layout

```text
digital_twin_mvp_docs/
|-- backend/
|   |-- app/
|   |   |-- main.py
|   |   |-- db_models.py
|   |   `-- routes/
|   |       |-- audio_processing.py
|   |       |-- ai.py
|   |       |-- network.py
|   |       `-- runs.py
|   |-- requirements.txt
|   `-- requirements-audio-birdnet.txt
|-- node_audio_workflow.py
|-- simplify_pipeline_results.py
|-- tiny_cnn_birdcall.py
|-- audio_event_common.py
|-- combined_audio_event_pipeline.py
|-- bird_longform_birdnet_pipeline.py
|-- gunshot_mvp_separate_training_longform.py
|-- REQUIRED_STAGE_INPUTS.md
|-- ARCHITECTURE.md
|-- README.md
|-- birdcalls/
|-- gunshots/
`-- test-audio_groundtruths/
```

## Summary

The project now has a backend-connected audio-processing path for the Digital Twin app, while preserving the earlier standalone bird/gunshot detectors. The staged node workflow can be tested today with `skip_birdnet: true`, and it is structured so Griffen's final TinyCNN weights and human-presence model can be plugged in with minimal changes.
