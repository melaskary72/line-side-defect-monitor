# Line-Side Surface Defect Monitor

A production-style deployment artifact built on [Roboflow](https://roboflow.com): a line-side camera feed is monitored for surface defects on hot-rolled steel, a rolling defect rate is maintained, and an operator alert fires when the rate crosses a tuned threshold. Every inference is written to an append-only audit log so any alert can be traced to the exact frames and detections that caused it.

I build and operate systems like this for a living: at GeoAct I install sensors and camera sensors on industrial machines and run the predictive maintenance models on what they capture. This repo is that pattern, implemented end to end on Roboflow in one evening.

## Representative scenario

A steel processor runs a camera over a hot-rolled strip line. QA today is periodic manual inspection, which means a bad coil can run for minutes before anyone notices. The ask: detect the six common surface defect types in real time, alert the line operator when defect density climbs, and keep an audit trail QA can review, all deployable on an edge box with unreliable connectivity.

## Architecture

```
 line camera (simulated: frame directory)
        |
        v
 +----------------------+     hosted mode: serverless.roboflow.com
 |  Roboflow model      | <-- edge mode:   Roboflow Inference server in
 |  (trained on NEU     |                  Docker on localhost:9001
 |   surface defects)   |
 +----------------------+
        |
        v
 confidence filter (0.40 floor)
        |
        v
 rolling window (20 frames) -> defect rate
        |
        +--> audit.jsonl      every frame, every detection, latency
        +--> annotated/       frames with boxes drawn, for QA review
        +--> ALERT + alerts.jsonl   when rate >= 30% with cooldown
```

A parallel Roboflow Workflow (model -> detections filter -> email notification block) covers the managed-cloud version of the same alert path.

## Model

Trained on Roboflow from a fork of the NEU Metal Surface Defects database ([Universe dataset](https://universe.roboflow.com/sujitsa/metallic-surface-defect-detector), 1.4k images, CC / public domain). Six classes: scratches, patches, crazing, inclusion, pitted surface, rolled-in scale.

Model: `melaskary72-gmail-com/metallic-surface-defect-detector-39eee/1` (Roboflow 3.0 Object Detection, Fast). Metrics from the Roboflow training report: mAP@50 67.1%, precision 66.2%, recall 66.6%.

## Acceptance criteria

1. Detects all six defect classes on held-out test frames at a 0.40 confidence floor.
2. Raises an alert within one window (20 frames) of defect rate reaching 30%.
3. No repeat alert inside the 15-frame cooldown, so operators are not spammed into ignoring it.
4. Every alert traceable to its frames and detections via `audit.jsonl`.
5. Runs identically in hosted and edge mode with one flag change.

## Alert tuning rationale

An unactioned alert and an alert nobody trusts fail the same way. The threshold is a rate over a window, not a single-frame trigger, because single defective frames are normal on any real line and per-frame alarms train operators to ignore the system. The 20-frame window and 30% rate are starting points to be tuned on the line with the operators who live with the alarms; the cooldown prevents alert storms during a sustained bad run, when the operator already knows and needs the system to shut up and log.

## Run it

```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY=your_key   # workspace settings -> API key. Never commit it.

# Hosted inference
python line_monitor.py --frames ./test_images --model melaskary72-gmail-com/metallic-surface-defect-detector-39eee/1

# Edge inference: production-representative, survives connectivity loss
# after weights are cached locally
docker run -d --name rf-inference -p 9001:9001 roboflow/roboflow-inference-server-cpu
python line_monitor.py --frames ./test_images --model melaskary72-gmail-com/metallic-surface-defect-detector-39eee/1 --mode edge
```

`test_images/` is the test split of the dataset version, downloaded from Roboflow.

## Runbook

* **Alert fires:** open `alerts.jsonl`, pull the listed window frames from `annotated/`, confirm or dismiss. Dismissals are feedback: recurring false positives on one class mean the confidence floor or training data for that class needs work.
* **Inference errors in the log:** in edge mode check the Docker container (`docker logs rf-inference`); in hosted mode check connectivity. Failed frames are logged, never dropped, because a monitoring gap on a live line is an incident, not noise.
* **Retraining:** confirmed misses go back into the Roboflow dataset as new labeled images; retrain, bump the model version, redeploy by changing one flag. Model version is pinned per run in the audit log, so results are attributable to a specific model.

## What I would do differently on a real line

Real camera ingest (RTSP) instead of a frame directory; camera calibration and lighting normalization at install, which is where most line-side vision projects actually fail; per-class thresholds, since a scratch and an inclusion do not carry equal cost; alert delivery to the operator HMI or SMS rather than stdout; and a weekly precision review against operator dismissals, the same loop I run for predictive maintenance alerting today.

---

Mohamed El Askary · [github.com/melaskary72](https://github.com/melaskary72) · Dataset credit: NEU Metal Surface Defects via Roboflow Universe.
