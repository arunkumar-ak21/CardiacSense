# How the ECG model was trained, and which model CardiacSense uses

Prepared for the project review. Every number below was read from a file in this project or from the Zenodo / paper pages linked at the end. Nothing here was re-trained by us.

---

## 1. The short answer

| Question | Answer |
|---|---|
| Which model does the new CODE-15% companion flow use? | **`model/model.pth`**, the original author's trained ResNet1d. |
| Is it really the author's model? | Yes. All three files in `model/` are byte-identical (same MD5) to `model.zip` published by the authors at Zenodo record 4892365 (`model.pth` MD5 `F6C7FA15623607AADA4388ED700C2701`; zip MD5 `00aad2c0...94cd5` matches the Zenodo page). |
| Where is it loaded? | `backend/ml/code15_engine.py` loads the weights from the top-level `model/` folder. `/health` reports `weights_loaded: true`. |
| Did we train it? | No. We load the authors' published weights and run inference only. |
| Does the app also contain the original VitalSense developer's models? | Yes, they are still in `backend/models/`, but only the old quick-vitals endpoints use them (see section 5). The new flow does not. |

---

## 2. What the model does

- **Task:** regression. It reads a raw 12-lead ECG and predicts the person's **age in years** ("ECG-age").
- **Use in CardiacSense:** `heart age gap = ECG-age - real age`. A positive gap means the ECG looks older than the patient.
- **Input:** tensor `(N, 12, 4096)`: 12 leads, 4096 samples at 400 Hz (about 10 s). Short signals are zero-padded.
- **Output:** `(N, 1)`, the predicted age.
- **Lead order:** DI, DII, DIII, AVR, AVL, AVF, V1-V6.

## 3. How it was trained (the authors' method)

Source: the authors' repository `ecg-age-prediction` (a copy is in `training/`), the paper by Lima, Ribeiro, Paixao et al., *Nature Communications* 12, 5117 (2021), and `model/config.json` + `history.csv`.

### 3.1 Data

- **Training data:** the `CODE` cohort (Telehealth Network of Minas Gerais, Brazil, 1,558,415 patients). About **85% of the patients** were used to develop the network. That full set is available only on request, so we do not have it.
- **Test data:** the other **15% of the patients** form **CODE-15%**, the dataset you linked (Zenodo 4916206): 345,779 exams from 233,770 patients, collected 2010-2016.
- **What this means for us:** the model never saw CODE-15% during training. CODE-15% is the authors' held-out test set, which makes it a fair set to demo and test on.
- **Split rule** (`formulate_problem.py`): splits are made **by patient**, so one patient's exams never appear in both train and test. Ages are balanced across the splits. `config.json` records `valid_split 0.05` and `test_split 0.15`.

### 3.2 Algorithm and architecture

- **Type:** a deep residual convolutional neural network for 1-D signals (1D-ResNet), written in PyTorch (`resnet.py`). It follows the ECG-classification network of Ribeiro et al. (*Nature Communications*, 2020), with the final sigmoid removed so it outputs a number.
- **Layout:**
  - one input convolution and batch normalisation;
  - 5 residual blocks. Filters grow 64 -> 128 -> 196 -> 256 -> 320 while the time axis shrinks 4096 -> 1024 -> 256 -> 64 -> 16;
  - each block has two 1-D convolutions (kernel size 17), batch normalisation, ReLU, dropout 0.8, and a skip connection (max-pool plus a 1x1 convolution when the shape changes);
  - the output is flattened (320 x 16 = 5120 values) into one linear layer that gives the age.
- **Size:** 72 weight tensors, **6,928,700 parameters**.

### 3.3 Training settings (from `train.py` and `config.json`)

| Setting | Value |
|---|---|
| Loss | Weighted squared error. Each exam is weighted by 1 / (number of exams with that age), so rare ages count as much as common ones. |
| Optimiser | Adam, learning rate 0.001 |
| Learning-rate schedule | `ReduceLROnPlateau`: multiply by 0.1 after 7 epochs without improvement, minimum 1e-7 |
| Batch size / epochs | 32 / up to 70 |
| Dropout / kernel size | 0.8 / 17 |
| Seed | 2 |
| Model selection | The checkpoint with the lowest validation loss is saved |

### 3.4 What this particular checkpoint reached (`history.csv`)

- 70 epochs were logged. The saved checkpoint is from **epoch 44**, with validation loss 136.15.
- At that epoch the validation **MAE is 8.52 years** and RMSE 11.10 years. (The weighted versions are 9.02 and 11.67.)
- The learning rate first dropped at epoch 25 and ended at 1e-7.
- `history.csv` has the whole curve. `training/plot_learning_curves.py` plots it.

In plain words: the predicted age is typically about 8-9 years away from the real age on the validation data. That is why CardiacSense treats the **gap** as a rough health signal and not as a diagnosis.

---

## 4. Which parts of the new CardiacSense flow are trained, and which are not

A reviewer may ask this directly, so this table is meant to be shown.

| Part | File | Trained? |
|---|---|---|
| ECG heart age (ResNet1d) | `ml/resnet1d.py` + `model-1/model.pth` | **Yes.** The authors' trained model. |
| ECG measurements: heart rate, QRS, PR, ST, rhythm irregularity | `ml/clinical_measurements.py` | No. Signal-processing rules. |
| Arrhythmia probabilities (1dAVb, RBBB, LBBB, SB, AF, ST) | `ml/code15_scoring.py` | **No.** Rules on the DSP measurements and the ticked flags. A small term from the ResNet features is added, but its mixing weights are random and fixed (`W_DEEP`, seed 2024), not learned. |
| Cross-attention fusion (risk adjustment, +/-5 points at most) | `ml/multimodal_fusion.py` | **No.** Weights come from a fixed random seed (1515). The code says so: "The weights are not trained". |
| Overall risk score | `ml/code15_scoring.py` | No. A hand-written formula with fixed weights (`RISK_SCALE 0.6`, `BASE_CONST 18`), tuned so the seeded demo patients land near their stored scores. It is not fitted to patient outcomes. |
| Diet and exercise plan | `genai/diet_planner.py` | No. Clinical rules plus a text model (Gemini) for the wording. |
| Saliency heatmap | `ml/saliency.py` | Not a model. It is the gradient of the ResNet output, weighted by the DSP findings. |

So the only learned component in the new flow is the ECG-age network. The CardiacSense code around it is rule-based and has **not been clinically validated**.

---

## 5. Other trained models in the project (the original VitalSense developer's)

These are still loaded by `main.py` for the **old** endpoints (`/predict`, `/predict/ecg`, `/ecg/demos`). The patient-journey screens (`/patients`, `/vitals`, `/predictions`, `/history`, `/insights`, `/chat`) do not use them.

| Model | File | Algorithm | Data |
|---|---|---|---|
| Vitals ensemble | `backend/models/ensemble.pkl` (+ `scaler.pkl`, `label_encoders.pkl`) | Per condition (hypertension, arrhythmia, hypoxia, tachycardia, bradycardia): **XGBoost (weight 0.45) + Random Forest (0.35) + small neural net `VitalsNet` (0.20)** | UCI Heart Disease (dataset id 45) features plus 4 vitals. See `train_ensemble.py`. |
| Anomaly detector | `backend/models/iso_forest.pkl` | Isolation Forest, 100 trees | same data |
| 12-lead ECG classifier | `backend/models/ecg_resnet.pt` | 1D-ResNet, 1,436,357 parameters, 5 classes (NORM, MI, STTC, CD, HYP) | PTB-XL 1.0.3 at 100 Hz, 21,799 records. Best validation macro-AUC 0.9309, test macro-AUC 0.925 (`ecg_model_report.md`). |

The PTB-XL demo data is not present on this machine, so the `/ecg/demos` endpoints return 503 at startup. That does not affect the new flow.

---

## 6. Can the training be reproduced?

- **Code:** yes. `training/train.py`, `resnet.py`, `dataloader.py`, `formulate_problem.py`, `evaluate.py` are the authors' original scripts.
- **Command (from the authors' README):** `python train.py PATH_TO_HDF5 PATH_TO_CSV`.
- **Data:** the full 85% training set is available only on request from the authors (doi 10.17044/scilifelab.15169716). CODE-15% alone is the test set.
- **Known issues if you run `train.py` today:**
  - It uses `np.Inf` and `DataFrame.append`, which newer NumPy and pandas versions removed. The script needs small edits to run.
  - The repo's `train.py` has no `valid_split` / `test_split` options, but `config.json` of `model-1` lists them. So the published weights came from a slightly newer version of the authors' script than the one in the repo. The settings that matter (optimiser, loss, epochs, learning-rate schedule, architecture) are identical.
- **Evaluation:** `python evaluate.py PATH_TO_MODEL_FOLDER PATH_TO_HDF5` writes predicted ages to a CSV.

We did not re-train anything. Re-training needs the full CODE data and a GPU, and would not be required to use the published model.

---

## 7. Likely reviewer questions

**Did you train the model?**
No. We reuse the pre-trained ECG-age network published by the original authors (Zenodo 4892365) and verified our copy is identical to it. Our own work is the application around it: signal processing, risk scoring, longitudinal comparison, diet/exercise planning and the user interface.

**Which algorithm is it?**
A 1-D residual convolutional neural network (ResNet) trained with Adam on a weighted squared-error loss to regress age from the raw 12-lead ECG.

**Which dataset trained it?**
The CODE cohort from Brazil (about 85% of its patients). The dataset on Zenodo, CODE-15%, is the held-out 15% used for testing.

**How accurate is it?**
On the authors' validation split the average error is about 8.5 years (MAE 8.52, RMSE 11.10 at the saved epoch).

**Is the risk score a trained prediction?**
No. It is a transparent rule-based score that uses the trained ECG age as one input. It has not been validated against patient outcomes and must not be used for diagnosis.

**Why use the ECG-age gap?**
The paper links ECG-age that is higher than real age with higher mortality, so the gap is a rough indicator of cardiovascular health.

---

## 8. Sources

- CODE-15% dataset: https://zenodo.org/records/4916206
- Authors' pre-trained models: https://zenodo.org/records/4892365
- Authors' training code: https://github.com/antonior92/ecg-age-prediction (copy in `training/`)
- Paper: Lima EM, Ribeiro AH, Paixao GMM et al. *Deep neural network-estimated electrocardiographic age as a mortality predictor.* Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7
- Local files used for the figures above: `model/config.json`, `model/history.csv`, `training/{train,resnet,evaluate,formulate_problem}.py`, `application/backend/models/{train_ensemble.py,ecg_model_report.md}`.
