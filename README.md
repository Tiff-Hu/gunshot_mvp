# Shaman Long-Form Audio Event Detection

This repository contains three standalone long-form audio detection pipelines for the Digital Twin MVP workflow:

- `combined_audio_event_pipeline.py`
- `bird_longform_birdnet_pipeline.py`
- `gunshot_mvp_separate_training_longform.py`

These scripts are designed to process long `.wav` recordings generated from Himesh's audio generator, including recordings up to 12 hours long.

The pipelines support:

- mixed audio containing both bird calls and gunshots
- bird-only long recordings
- gunshot-only long recordings
- comparison against generator ground-truth answer-key JSON logs
- backend-friendly JSON outputs for later app integration

## What each script does

### 1. `combined_audio_event_pipeline.py`
Runs both the bird and gunshot pipelines on the same long-form `.wav` file and merges their detections into one combined structured timeline.

Use this for mixed recordings that may contain both bird calls and gunshots.

### 2. `bird_longform_birdnet_pipeline.py`
Runs the bird-only pipeline on a long-form `.wav` file.

This pipeline uses BirdNET via `birdnetlib` for bird confirmation and species labeling.

Use this for recordings generated in bird-only mode.

### 3. `gunshot_mvp_separate_training_longform.py`
Runs the gunshot-only pipeline on a long-form `.wav` file.

This pipeline keeps the two-stage structure:

- **Shaman I**: fast acoustic prefilter to find candidate transient events
- **Shaman II**: learned confirmation model trained from short gunshot and non-gunshot clips

Use this for recordings generated in gunshot-only mode.

## High-level workflow

### Combined pipeline
1. Load long `.wav` audio
2. Run gunshot detection
3. Run bird detection
4. Merge detections into a combined timeline
5. Optionally compare detections to the generator answer key
6. Write timeline, evaluation, and backend-friendly JSON outputs

### Bird-only pipeline
1. Load long `.wav` audio
2. Run acoustic trigger detection
3. Confirm bird events with BirdNET
4. Produce a bird timeline
5. Optionally evaluate against ground truth
6. Write the same output bundle style as the combined pipeline

### Gunshot-only pipeline
1. Load short gunshot and non-gunshot training clips
2. Bootstrap positive and negative subclips
3. Train the Shaman II confirmation model
4. Scan the long `.wav` audio for candidate transient events
5. Confirm gunshots
6. Optionally evaluate against ground truth
7. Write the same output bundle style as the combined pipeline

## Input files

### Audio input
All three scripts expect a long `.wav` audio file.

Examples:

- mixed mode: `node_001_12hr_20260407_152944.wav`
- bird-only mode: `node_001_12hr_20260414_121347.wav`
- gunshot-only mode: `node_001_12hr_20260414_121403.wav`

### Ground-truth answer key
All three scripts can compare detections to the generator answer bank JSON.

Examples:

- mixed mode: `node_001_12hr_20260407_152944_log.json`
- bird-only mode: `node_001_12hr_20260414_121347_log.json`
- gunshot-only mode: `node_001_12hr_20260414_121403_log.json`

The scripts accept either:

- `--ground_truth_log`
- `--ground_truth_json`

### Gunshot training inputs
The gunshot pipeline requires:

- `--gunshot_dir`: directory of short gunshot clips
- `--negative_dir`: directory of short non-gunshot clips

The combined pipeline also requires these because it internally runs the gunshot detector.

## Output files

All three scripts now write the same bundle style so they are easy to compare and easy to hand to a backend later.

Typical output directory contents:

- `bird_timeline.json`
- `gunshot_timeline.json`
- `combined_ai_event_timeline.json`
- `evaluation.json`
- `backend_payload.json`
- `combined_run_summary.json`
- `run_summary.json`

### Output file descriptions

#### `bird_timeline.json`
Bird detections only.

Each event includes fields like:

- `event_type`
- `source_file`
- `trigger_time_s`
- `trigger_time_formatted`
- `clip_start_s`
- `clip_end_s`
- `shaman_i`
- `shaman_ii`

For confirmed bird detections, `shaman_ii` includes:

- detected species
- confidence
- BirdNET raw label information

#### `gunshot_timeline.json`
Gunshot detections only.

Each event includes:

- `event_type`
- `source_file`
- `trigger_time_s`
- `trigger_time_formatted`
- `clip_start_s`
- `clip_end_s`
- `shaman_i`
- `shaman_ii`

For confirmed gunshots, `shaman_ii` includes:

- `is_gunshot`
- confidence
- label

#### `combined_ai_event_timeline.json`
Combined merged event timeline across event types.

For bird-only runs, this will contain bird events.

For gunshot-only runs, this will contain gunshot events.

For mixed runs, this will contain both.

#### `evaluation.json`
Comparison between predictions and the generator answer key.

This includes metrics such as:

- `n_predictions`
- `n_expected`
- `true_positives`
- `false_positives`
- `false_negatives`
- `precision`
- `recall`
- `f1`
- matched event timestamp errors

#### `backend_payload.json`
Backend-friendly structured payload for later integration into the Digital Twin app.

#### `combined_run_summary.json`
High-level run summary including counts of total events and evaluation results.

#### `run_summary.json`
Convenience alias summary for standalone runs.

## Evaluation logic

The scripts compare predictions against the generator answer key.

### Gunshots
- event type: `GUNSHOT`
- default tolerance: 2.0 seconds
- category match is not required by default

### Birds
- event type: `BIRD`
- default tolerance: 5.0 seconds
- category or species match is required

## Example commands

### Combined pipeline
```bash
conda activate birdstates

python combined_audio_event_pipeline.py \
  --input_audio "/path/to/node_001_12hr_20260407_152944.wav" \
  --out_dir "/path/to/task49_combined_outputs" \
  --ground_truth_log "/path/to/node_001_12hr_20260407_152944_log.json" \
  --gunshot_dir "/path/to/gunshot_data" \
  --negative_dir "/path/to/non_gunshot_data"
```

### Bird-only pipeline
```bash
conda activate birdstates

python bird_longform_birdnet_pipeline.py \
  --input_audio "/path/to/node_001_12hr_20260414_121347.wav" \
  --out_dir "/path/to/bird_only_outputs_121347" \
  --ground_truth_log "/path/to/node_001_12hr_20260414_121347_log.json"
```

### Gunshot-only pipeline
```bash
conda activate birdstates

python gunshot_mvp_separate_training_longform.py \
  --gunshot_dir "/path/to/gunshot_data" \
  --negative_dir "/path/to/non_gunshot_data" \
  --input_audio "/path/to/node_001_12hr_20260414_121403.wav" \
  --out_dir "/path/to/gunshot_only_outputs_121403" \
  --ground_truth_log "/path/to/node_001_12hr_20260414_121403_log.json"
```

## Python environment and dependencies

Recommended environment:

```bash
conda create -n birdstates python=3.11 -y
conda activate birdstates
```

Install packages:

```bash
python -m pip install --upgrade pip
python -m pip install numpy scipy soundfile librosa torch scikit-learn pydub birdnetlib tensorflow
```

If BirdNET or `pydub` warns that `ffmpeg` is missing, install it locally.

On macOS with Homebrew:

```bash
brew install ffmpeg
```

## Notes on current behavior

### Bird-only pipeline
The bird pipeline is working end-to-end and can localize many correct events, but it may still over-detect and produce repeated nearby confirmations for a single ground-truth event.

Potential next improvements:

- higher BirdNET confirmation threshold
- deduplication or non-max suppression across nearby bird detections
- stricter candidate filtering before BirdNET confirmation

### Gunshot-only pipeline
The gunshot pipeline is currently much cleaner and performs strongly on gunshot-only long recordings.

Potential next improvement:

- merge nearby duplicate confirmations from the same underlying gunshot event

## Backend integration note

These scripts currently work as standalone local programs.

They are structured so they can later be called by a local backend using node file paths supplied by the app. The `backend_payload.json` file is intended to make that later integration easier.

## Suggested repository layout

```text
project_root/
├── combined_audio_event_pipeline.py
├── bird_longform_birdnet_pipeline.py
├── gunshot_mvp_separate_training_longform.py
├── audio_event_common.py
├── README.md
├── Data/
│   ├── gunshot_data/
│   └── non_gunshot_data/
└── outputs/
    ├── combined/
    ├── bird_only/
    └── gunshot_only/
```

## Summary

This project now supports three aligned long-form workflows:

- mixed bird + gunshot detection
- bird-only detection
- gunshot-only detection

All three produce a consistent set of outputs, can be evaluated against Himesh's answer-key JSON logs, and are organized in a way that can later be connected to the Digital Twin backend.
