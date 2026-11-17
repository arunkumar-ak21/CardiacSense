# CardiacSense

A longitudinal cardiac companion. A patient returns after a hospital visit, uploads a new 12-lead ECG (`.hdf5`, `.npy` or `.csv`, CODE-15% format) with blood results, and CardiacSense shows:

- ECG **heart age** and the gap to the real age, from a pre-trained ResNet1d
- a **risk score (0-100%)** with a severity tier: Low, Moderate, High or Critical
- arrhythmia probabilities (1dAVb, RBBB, LBBB, SB, AF, ST) and a 12-lead saliency heatmap
- **Before vs. Now** changes since the previous visit
- an adaptive **diet and exercise plan**, a narrative summary and a chat assistant
- a staff approval step for the care team

> Research prototype for a college project. It is not clinically validated and must not be used for diagnosis.

## Folder structure

```
CardiacSense/
├── application/   the web app: React + TypeScript frontend (src/) and FastAPI backend (backend/)
├── training/      the original authors' code that trained the ECG-age model (ecg-age-prediction)
├── model/         the trained weights used by the app: model.pth, config.json, history.csv
└── docs/          model training explanation and the sample-ECG test guide
```

## Run it

You need Python 3.13 and Node.js 20+.

**1. Backend** (port 8000)

```bash
cd application/backend
pip install -r requirements.txt
cp .env.example .env          # optional: add a GEMINI_API_KEY for AI-written text
python -m uvicorn main:app --port 8000
```

**2. Frontend** (port 5173)

```bash
cd application
npm install
npm run dev
```

Open http://localhost:5173, log in (demo mode accepts any login) and choose a patient. Two demo patients are created on first start. Sample ECGs are in `application/sample_code15_ecgs/`.

Without a `GEMINI_API_KEY` the app still works. The text is then built from rule-based templates instead of an LLM.

**Check that the model is connected:** open http://localhost:8000/health. It should show `"weights_loaded": true`.

## The model

`model/model.pth` is the pre-trained ECG-age network published by the authors of *Deep neural network-estimated electrocardiographic age as a mortality predictor* (Lima, Ribeiro, Paixao et al., Nature Communications 2021). It is a 1D ResNet trained on the CODE cohort. We use it for inference only.

- Download page: https://zenodo.org/records/4892365
- Training code: `training/` (copy of https://github.com/antonior92/ecg-age-prediction)
- How it was trained and what is and is not learned in this app: [docs/MODEL_TRAINING_AND_INTEGRATION.md](docs/MODEL_TRAINING_AND_INTEGRATION.md)

The risk score, severity tiers, arrhythmia probabilities and diet plan are rule-based. Only the heart age comes from a trained network.

## Tests

```bash
cd application/backend
pip install -r requirements-dev.txt
python -m pytest tests -q
python verify_code15_multimodal.py
```

## Dataset

CODE-15%: https://zenodo.org/records/4916206 (345,779 exams, 233,770 patients). It is the authors' held-out test set.

## Acknowledgements

- The frontend and backend started from the open-source VitalSense project by derrickrajkumar10 (https://github.com/derrickrajkumar10/VitalSense) and were extended into this longitudinal ECG companion.
- `training/` and `model/` come from the ecg-age-prediction project by Emilly M. Lima, Antonio H. Ribeiro and colleagues (MIT licence, see `training/LICENSE`).
