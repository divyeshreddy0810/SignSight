# SignSight

Real-time American Sign Language recognition from MediaPipe skeletal landmarks.
Two tasks: an 8-word vocabulary and the 26-letter fingerspelling alphabet.
Served by five microservices behind a browser front end. No video leaves the
device; only landmarks are sent to the backend.

## How to run

Requires Python 3.11 and a webcam. Use Chrome or Brave.

```bash
python3.11 -m venv .venv-ml
source .venv-ml/bin/activate
pip install -r requirements.txt      # exact versions: requirements.lock.txt
./start_all.sh
```

Open **http://localhost:8080/index.html** in the browser.
Do not open the HTML file directly. The camera only works over
`http://localhost`, not `file://`.

Stop everything:

```bash
pkill -f uvicorn && pkill -f http.server
```

Ports: 8000 gateway · 8001 preprocessing · 8002 vision · 8003 grammar · 8080 frontend.

### Retraining (optional; trained models ship in `ml/models/`)

```bash
# Word model
python ml/build_dataset.py           # WLASL download + landmark extraction
python ml/evaluate.py                # full model sweep -> results/
python ml/export_final_model.py      # writes ml/models/random_forest.joblib

# Fingerspelling model
python ml/build_alphabet_dataset.py  # landmark extraction from the image set
python ml/train_letters.py           # writes ml/models/letter_rf.joblib
                                     # add --calibration to include webcam samples

python ml/test_smoke.py              # quick pipeline sanity check
```

## Directory layout

```
.
├── README.md
├── requirements.txt              # dependencies
├── requirements.lock.txt         # exact pinned versions
├── start_all.sh                  # launches all five services
│
├── frontend/index.html           # React UI: camera + hands-only landmark view
│
├── gateway/main.py               # API gateway (8000)
├── preprocessing-service/app.py  # quality gate, DBSCAN, smoothing (8001)
├── vision-service/app.py         # model inference (8002)
├── nlp-service/app.py            # sentence composition (8003)
│
├── ml/
│   ├── features.py               # feature engineering (shared train + serve)
│   ├── stgcn.py                  # spatio-temporal graph network
│   ├── build_dataset.py          # WLASL -> word landmark dataset
│   ├── build_alphabet_dataset.py # images -> letter landmark dataset
│   ├── evaluate.py               # repeated-CV sweep + significance tests
│   ├── train_baseline.py         # RF / SVM baseline check
│   ├── train_lstm.py             # GRU / LSTM sequence models
│   ├── train_letters.py          # fingerspelling model (+ calibration)
│   ├── train_validator.py        # SVM frame validator
│   ├── export_final_model.py     # exports the deployed word model
│   ├── measure_system.py         # latency / FPS measurement
│   ├── test_smoke.py             # pipeline sanity checks
│   └── models/                   # trained models (RF word, RF letter, SVM)
│
├── data/
│   ├── wlasl_subset.json         # WLASL index for the 8 glosses
│   └── processed/                # word landmark dataset (104 .npy + labels)
│
└── results/
    ├── figures/                  # result graphs (PNG)
    ├── decision_viz.html         # interactive decision visualisation
    └── *.csv                     # evaluation, significance, latency results
```

Generated locally and not tracked: `data/raw_videos/` (WLASL source clips),
`data/processed_alphabet/` (letter landmarks), `data/calibration_samples.jsonl`
(personal webcam frames), and the `.venv-ml/` environment.
