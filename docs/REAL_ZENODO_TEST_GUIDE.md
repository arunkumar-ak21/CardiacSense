# Real Zenodo CODE-15% Patient ECG Test Guide

This guide documents the **authentic, non-synthetic patient 12-lead ECGs** extracted directly from the Zenodo CODE-15% dataset (`exams.csv` + `exams_part0.hdf5`).

All files are 12-lead recordings sampled at 400Hz with dimension **`(4096, 12)`** in millivolts ($1\text{e-}4\text{ V}$ units). They are available in `.npy`, `.csv`, and `.hdf5` formats in both:
- `REPO/VitalSense/real_zenodo_ecgs/`
- `REPO/VitalSense/sample_code15_ecgs/` (accessible via `GET /ecg/code15-samples`)

---

## 1. Summary Comparison Table

| Extracted Filename (.npy / .csv / .hdf5) | Patient ID | Age / Gender | Ground-Truth Flags | Published `nn_predicted_age` | Our ResNet1d Age | Risk Score & Tier | DSP Measured HR / QRS / ST |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| `real_visit1_normal_sinus` | 801634 | 51 F | **Normal ECG** | **60.12 yrs** | **60.1 yrs** (diff 0.02) | 28.7% (Moderate) | HR 68.0 bpm, QRS 100.0 ms, Elevation |
| `real_visit2_af_and_lbbb_high_risk` | 1727659 | 82 M | **LBBB, AF** | **84.55 yrs** | **84.5 yrs** (diff 0.05) | 50.0% (High) | HR 55.8 bpm, QRS 167.5 ms, Elevation |
| `real_visit3_bradycardia_1davb` | 1750080 | 68 F | **1dAVb, SB** | **72.1 yrs** | **72.1 yrs** (diff 0.0) | 37.2% (Moderate) | HR 46.9 bpm, QRS 95.0 ms, Elevation |
| `real_visit4_tachycardia_st` | 1769456 | 62 M | **ST** | **81.42 yrs** | **81.4 yrs** (diff 0.02) | 41.0% (Moderate) | HR 103.4 bpm, QRS 75.0 ms, Elevation |
| `real_multivisit_pat62825_visit1_exam179183` | 62825 | 59 F | **Normal ECG** | **61.44 yrs** | **61.4 yrs** (diff 0.04) | 27.9% (Moderate) | HR 57.6 bpm, QRS 90.0 ms, Elevation |
| `real_multivisit_pat62825_visit2_exam433650` | 62825 | 60 F | **AF** | **70.63 yrs** | **70.6 yrs** (diff 0.03) | 48.5% (Moderate) | HR 152.4 bpm, QRS 82.5 ms, Depression |

---

## 2. Clinical Case Profiles & Testing Guide

### `real_visit1_normal_sinus`
- **Clinical Category:** Normal Sinus
- **Exam ID:** `214263` | **Patient ID:** `801634`
- **Demographics:** Age 51, Gender: Female
- **Clinical Rationale:** Verified normal 12-lead ECG from Zenodo CODE-15% cohort. Narrow QRS, normal PR interval, isoelectric ST.
- **Ground Truth CODE-15% Annotations:**
  - `1dAVb`: `False`
  - `RBBB`: `False`
  - `LBBB`: `False`
  - `SB`: `False`
  - `AF`: `False`
  - `ST`: `False`
  - `normal_ecg`: `True`
- **Heart Age Validation:** Published = `60.12 yrs`, Our ResNet1d = `60.1 yrs` (Residual Delta: `0.02 yrs`)
- **DSP Telemetry:** HR = `68.0 bpm`, QRS Duration = `100.0 ms`, ST Segment = `Elevation` (`+0.215 mV`)
- **Multi-Modal Risk:** `28.7%` (Moderate)

### `real_visit2_af_and_lbbb_high_risk`
- **Clinical Category:** AF + LBBB
- **Exam ID:** `2999852` | **Patient ID:** `1727659`
- **Demographics:** Age 82, Gender: Male
- **Clinical Rationale:** Severe dual conduction disturbance: Left bundle branch block (prolonged QRS) accompanied by atrial fibrillation.
- **Ground Truth CODE-15% Annotations:**
  - `1dAVb`: `False`
  - `RBBB`: `False`
  - `LBBB`: `True`
  - `SB`: `False`
  - `AF`: `True`
  - `ST`: `False`
  - `normal_ecg`: `False`
- **Heart Age Validation:** Published = `84.55 yrs`, Our ResNet1d = `84.5 yrs` (Residual Delta: `0.05 yrs`)
- **DSP Telemetry:** HR = `55.8 bpm`, QRS Duration = `167.5 ms`, ST Segment = `Elevation` (`+0.316 mV`)
- **Multi-Modal Risk:** `50.0%` (High)

### `real_visit3_bradycardia_1davb`
- **Clinical Category:** SB + 1dAVb
- **Exam ID:** `741685` | **Patient ID:** `1750080`
- **Demographics:** Age 68, Gender: Female
- **Clinical Rationale:** Low resting rate with marked PR prolongation exceeding 200 ms representing 1st-degree atrioventricular delay.
- **Ground Truth CODE-15% Annotations:**
  - `1dAVb`: `True`
  - `RBBB`: `False`
  - `LBBB`: `False`
  - `SB`: `True`
  - `AF`: `False`
  - `ST`: `False`
  - `normal_ecg`: `False`
- **Heart Age Validation:** Published = `72.1 yrs`, Our ResNet1d = `72.1 yrs` (Residual Delta: `0.0 yrs`)
- **DSP Telemetry:** HR = `46.9 bpm`, QRS Duration = `95.0 ms`, ST Segment = `Elevation` (`+0.328 mV`)
- **Multi-Modal Risk:** `37.2%` (Moderate)

### `real_visit4_tachycardia_st`
- **Clinical Category:** Sinus Tachycardia
- **Exam ID:** `423730` | **Patient ID:** `1769456`
- **Demographics:** Age 62, Gender: Male
- **Clinical Rationale:** Elevated resting sinus rhythm exceeding 100 bpm without atrial fibrillation.
- **Ground Truth CODE-15% Annotations:**
  - `1dAVb`: `False`
  - `RBBB`: `False`
  - `LBBB`: `False`
  - `SB`: `False`
  - `AF`: `False`
  - `ST`: `True`
  - `normal_ecg`: `False`
- **Heart Age Validation:** Published = `81.42 yrs`, Our ResNet1d = `81.4 yrs` (Residual Delta: `0.02 yrs`)
- **DSP Telemetry:** HR = `103.4 bpm`, QRS Duration = `75.0 ms`, ST Segment = `Elevation` (`+0.473 mV`)
- **Multi-Modal Risk:** `41.0%` (Moderate)

### `real_multivisit_pat62825_visit1_exam179183`
- **Clinical Category:** Longitudinal Visit 1 (Baseline)
- **Exam ID:** `179183` | **Patient ID:** `62825`
- **Demographics:** Age 59, Gender: Female
- **Clinical Rationale:** Baseline hospital visit for Patient 62825. Preserved rhythm, normal conduction, normal heart age (~61 yrs).
- **Ground Truth CODE-15% Annotations:**
  - `1dAVb`: `False`
  - `RBBB`: `False`
  - `LBBB`: `False`
  - `SB`: `False`
  - `AF`: `False`
  - `ST`: `False`
  - `normal_ecg`: `True`
- **Heart Age Validation:** Published = `61.44 yrs`, Our ResNet1d = `61.4 yrs` (Residual Delta: `0.04 yrs`)
- **DSP Telemetry:** HR = `57.6 bpm`, QRS Duration = `90.0 ms`, ST Segment = `Elevation` (`+0.185 mV`)
- **Multi-Modal Risk:** `27.9%` (Moderate)

### `real_multivisit_pat62825_visit2_exam433650`
- **Clinical Category:** Longitudinal Visit 2 (Worsened)
- **Exam ID:** `433650` | **Patient ID:** `62825`
- **Demographics:** Age 60, Gender: Female
- **Clinical Rationale:** Follow-up visit 1 year later for Patient 62825. Developed new-onset Atrial Fibrillation with a +9.2 year heart age surge.
- **Ground Truth CODE-15% Annotations:**
  - `1dAVb`: `False`
  - `RBBB`: `False`
  - `LBBB`: `False`
  - `SB`: `False`
  - `AF`: `True`
  - `ST`: `False`
  - `normal_ecg`: `False`
- **Heart Age Validation:** Published = `70.63 yrs`, Our ResNet1d = `70.6 yrs` (Residual Delta: `0.03 yrs`)
- **DSP Telemetry:** HR = `152.4 bpm`, QRS Duration = `82.5 ms`, ST Segment = `Depression` (`-0.249 mV`)
- **Multi-Modal Risk:** `48.5%` (Moderate)

---

## 3. Real Longitudinal Progression Showcase: Patient 62825

Patient `62825` serves as the ultimate test case for the **Before vs. Now (Delta Engine)** and **Adaptive Diet & Exercise Planner**:

1. **Visit 1 (`real_multivisit_pat62825_visit1_exam179183.npy`):**
   - Patient is 59 years old, female.
   - ECG is strictly **normal sinus rhythm** (`normal_ecg == True`, no arrhythmias).
   - Predicted Heart Age: `61.4 yrs`.
   - Severity: **Moderate / Low Risk**.

2. **Visit 2 (`real_multivisit_pat62825_visit2_exam433650.npy`):**
   - Patient returns 1 year later (age 60).
   - ECG has degraded into **New-Onset Atrial Fibrillation** (`AF == True`, `normal_ecg == False`).
   - Predicted Heart Age surges by **+9.2 years** to `70.6 yrs`!
   - Severity shifts to **Critical / High Risk**.
   - **Delta Engine Impact:** Highlights new-onset AF, surges the cardiovascular risk score, restricts dietary sodium, and lowers exercise heart rate ceilings.

---

## 4. How to Test in the VitalSense Web UI

1. Start the backend: `python -m uvicorn main:app --port 8000`
2. Start the frontend: `npm run dev` and navigate to [http://localhost:5173/vitals](http://localhost:5173/vitals)
3. Open the **ECG Analysis** tab:
   - Click on the new **Real Patient Records** chips to immediately visualize the real 12-lead waveforms in the interactive lead viewer.
   - Or upload `real_multivisit_pat62825_visit1_exam179183.npy` under a patient profile, record the baseline visit, then upload `real_multivisit_pat62825_visit2_exam433650.npy` to see the live **What Went Wrong** delta comparison and updated Diet & Exercise plan!
