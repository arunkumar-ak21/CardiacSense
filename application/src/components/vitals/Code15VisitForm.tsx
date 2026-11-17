import { useEffect, useRef, useState, type DragEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FileUp, HeartPulse, Loader2, Stethoscope, Upload } from 'lucide-react';
import ECGLeadViewer from './ECGLeadViewer';
import { useVitals } from '../../context/VitalsContext';
import { DIET_OPTIONS } from '../PatientProfileSwitcher';
import {
  CODE15_CODES, CODE15_LABELS, fetchCode15SamplePreview, previewCode15Upload,
  type Code15Code, type Code15Preview, type DietPreference,
} from '../../lib/api';

type Flags = Record<Code15Code, boolean>;
const NO_FLAGS: Flags = { '1dAVb': false, RBBB: false, LBBB: false, SB: false, AF: false, ST: false };

interface ClinicalValues {
  hr: string; bp_sys: string; bp_dia: string; spo2: string; rr: string; temp: string;
  glucose: string; cholesterol: string; creatinine: string; hrv: string;
}
const EMPTY_VALUES: ClinicalValues = { hr: '', bp_sys: '', bp_dia: '', spo2: '', rr: '', temp: '', glucose: '', cholesterol: '', creatinine: '', hrv: '' };

const PRESETS: { id: string; label: string; file: string; dot: string; flags: Partial<Flags>; values: ClinicalValues; notes: string }[] = [
  {
    id: 'normal', label: '1. Normal Sinus', file: '1_normal_sinus_4096x12.npy', dot: 'bg-sage-dark', flags: {},
    values: { hr: '72', bp_sys: '122', bp_dia: '78', spo2: '98', rr: '14', temp: '36.7', glucose: '95', cholesterol: '176', creatinine: '0.9', hrv: '52' },
    notes: 'Walked 30 minutes daily and cooked at home.',
  },
  {
    id: 'af_lbbb', label: '2. AF + LBBB High Risk', file: '2_atrial_fib_lbbb_high_risk_4096x12.npy', dot: 'bg-rose-dark', flags: { AF: true, LBBB: true },
    values: { hr: '92', bp_sys: '156', bp_dia: '94', spo2: '95', rr: '18', temp: '36.8', glucose: '168', cholesterol: '242', creatinine: '1.7', hrv: '28' },
    notes: 'Skipped evening walks for 5 weeks and ate high-salt restaurant food frequently.',
  },
  {
    id: 'sb_avb', label: '3. Sinus Bradycardia + 1dAVb', file: '3_sinus_bradycardia_1davb_4096x12.npy', dot: 'bg-amber-dark', flags: { SB: true, '1dAVb': true },
    values: { hr: '48', bp_sys: '134', bp_dia: '84', spo2: '97', rr: '15', temp: '36.6', glucose: '112', cholesterol: '205', creatinine: '1.1', hrv: '38' },
    notes: 'Occasional dizziness on standing.',
  },
  {
    id: 'st_dep', label: '4. Critical ST Depression', file: '4_critical_st_depression_tachycardia_4096x12.npy', dot: 'bg-rose-dark', flags: { ST: true },
    values: { hr: '118', bp_sys: '162', bp_dia: '98', spo2: '93', rr: '22', temp: '37.1', glucose: '185', cholesterol: '258', creatinine: '1.5', hrv: '22' },
    notes: 'Chest tightness while climbing stairs; stopped the low-salt diet.',
  },
];

const ADHERENCE_CHIPS = ['Ate salty / restaurant food often', 'Skipped exercise most days', 'Missed some medications', 'Walked daily as advised'];

type Adherence = 'fully' | 'partially' | 'not' | '';
const ADHERENCE_OPTIONS: { value: Exclude<Adherence, ''>; label: string; prefix: string; cls: string }[] = [
  { value: 'fully', label: '✅ Followed fully', prefix: 'Followed the previous diet & exercise plan fully.', cls: 'border-sage-dark/30 bg-sage-light text-sage-dark' },
  { value: 'partially', label: '🟡 Partially followed', prefix: 'Partially followed the previous diet & exercise plan.', cls: 'border-amber-dark/30 bg-amber-light text-amber-dark' },
  { value: 'not', label: '❌ Did not follow', prefix: 'Did not follow the previous diet & exercise plan.', cls: 'border-rose-dark/30 bg-rose-light/70 text-rose-dark' },
];
const PRESET_ADHERENCE: Record<string, Adherence> = { normal: 'fully', af_lbbb: 'partially', sb_avb: 'partially', st_dep: 'not' };

const FIELD_DEFS: { key: keyof ClinicalValues; label: string; unit: string; step?: string; required?: boolean }[] = [
  { key: 'bp_sys', label: 'Systolic BP', unit: 'mmHg', required: true },
  { key: 'bp_dia', label: 'Diastolic BP', unit: 'mmHg', required: true },
  { key: 'hr', label: 'Heart Rate', unit: 'bpm', required: true },
  { key: 'spo2', label: 'SpO2', unit: '%' },
  { key: 'glucose', label: 'Fasting Blood Glucose', unit: 'mg/dL', required: true },
  { key: 'cholesterol', label: 'Total Cholesterol', unit: 'mg/dL', required: true },
  { key: 'creatinine', label: 'Serum Creatinine', unit: 'mg/dL', step: '0.01', required: true },
  { key: 'hrv', label: 'Heart Rate Variability', unit: 'ms' },
  { key: 'rr', label: 'Respiratory Rate', unit: '/min' },
  { key: 'temp', label: 'Temperature', unit: '°C', step: '0.1' },
];

function todayISO(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

export default function Code15VisitForm() {
  const navigate = useNavigate();
  const { activeProfile, isStaffMode, recordNewVisit, visits } = useVitals();
  const lastVisit = visits[0] ?? null;
  const lastPlan = lastVisit?.diet_plan;
  const [adherence, setAdherence] = useState<Adherence>('');
  const [values, setValues] = useState<ClinicalValues>(EMPTY_VALUES);
  const [flags, setFlags] = useState<Flags>(NO_FLAGS);
  const [dietPref, setDietPref] = useState<DietPreference>('vegetarian');
  const [hospital, setHospital] = useState('City General Hospital - Cardiology OPD');
  const [notes, setNotes] = useState('');
  const [visitDate, setVisitDate] = useState(todayISO());
  const [file, setFile] = useState<File | null>(null);
  const [sampleFile, setSampleFile] = useState<string | null>(null);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [preview, setPreview] = useState<Code15Preview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (activeProfile?.diet_preference && DIET_OPTIONS.some(d => d.value === activeProfile.diet_preference)) {
      setDietPref(activeProfile.diet_preference as DietPreference);
    }
  }, [activeProfile]);

  const loadPreview = async (loader: () => Promise<Code15Preview>) => {
    setPreviewLoading(true);
    setError(null);
    try {
      setPreview(await loader());
    } catch (e) {
      setPreview(null);
      setError(e instanceof Error ? e.message : 'Could not load the ECG preview');
    } finally {
      setPreviewLoading(false);
    }
  };

  const applyPreset = (p: (typeof PRESETS)[number]) => {
    setActivePreset(p.id);
    setFile(null);
    setSampleFile(p.file);
    setValues(p.values);
    setFlags({ ...NO_FLAGS, ...p.flags });
    setNotes(p.notes);
    setAdherence(lastVisit ? PRESET_ADHERENCE[p.id] ?? '' : '');
    void loadPreview(() => fetchCode15SamplePreview(p.file));
  };

  const acceptFile = (f: File | undefined | null) => {
    if (!f) return;
    if (!/\.(npy|csv|hdf5|h5)$/i.test(f.name)) {
      setError('Please upload a .npy, .csv, .hdf5 or .h5 12-lead ECG file.');
      return;
    }
    setFile(f);
    setSampleFile(null);
    setActivePreset(null);
    void loadPreview(() => previewCode15Upload(f));
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    acceptFile(e.dataTransfer.files?.[0]);
  };

  const submit = async () => {
    if (!activeProfile) { setError('Select or create a patient profile first.'); return; }
    const missing = FIELD_DEFS.filter(f => f.required && !values[f.key].trim()).map(f => f.label);
    if (missing.length) { setError(`Please fill in: ${missing.join(', ')}`); return; }
    if (Number(values.bp_dia) >= Number(values.bp_sys)) { setError('Diastolic BP must be lower than systolic BP.'); return; }

    const form = new FormData();
    form.append('patient_id', activeProfile.id);
    form.append('visit_date', visitDate);
    form.append('hospital_name', hospital);
    const prefix = ADHERENCE_OPTIONS.find(o => o.value === adherence)?.prefix;
    const adherenceText = [prefix, notes.trim()].filter(Boolean).join(' ')
      || (lastVisit ? '' : 'Baseline visit: no previous plan on record.');
    form.append('adherence_notes', adherenceText);
    form.append('diet_preference', dietPref);
    (Object.keys(values) as (keyof ClinicalValues)[]).forEach(k => { if (values[k].trim()) form.append(k, values[k].trim()); });
    CODE15_CODES.forEach(c => form.append(c, flags[c] ? 'true' : 'false'));
    if (file) form.append('file', file);
    else if (sampleFile) form.append('sample_filename', sampleFile);

    setSubmitting(true);
    setError(null);
    try {
      await recordNewVisit(form);
      navigate('/predictions');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Visit analysis failed. Is the backend running on port 8000?');
    } finally {
      setSubmitting(false);
    }
  };

  const sourceLabel = file ? `Uploaded: ${file.name}` : sampleFile ? `CODE-15% preset: ${sampleFile}` : 'No ECG file — a waveform will be synthesized from the vitals and flags';
  const inputCls = 'w-full rounded-lg border border-black/10 bg-paper px-3 py-2 text-sm text-ink-main outline-none transition focus:border-lavender-dark';

  return (
    <div className="flex flex-col gap-6">
      {/* ── Step 1: ECG ─────────────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-ink-main">
          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-ink-main text-[10px] text-paper">1</span>
          12-Lead CODE-15% ECG
        </h3>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map(p => (
            <button
              key={p.id}
              onClick={() => applyPreset(p)}
              className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition ${activePreset === p.id ? 'border-ink-main bg-ink-main text-paper' : 'border-black/5 bg-paper text-ink-muted hover:border-black/15 hover:text-ink-main'}`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${p.dot}`} /> {p.label}
            </button>
          ))}
        </div>
        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`flex cursor-pointer items-center gap-4 rounded-xl border-2 border-dashed px-5 py-4 transition ${dragOver ? 'border-lavender-dark bg-lavender-light/40' : 'border-black/10 bg-paper hover:border-black/20'}`}
        >
          <input ref={fileInputRef} type="file" accept=".npy,.csv,.hdf5,.h5" className="hidden" onChange={e => acceptFile(e.target.files?.[0])} />
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cream text-ink-muted">
            {file ? <FileUp className="h-5 w-5" /> : <Upload className="h-5 w-5" />}
          </div>
          <div className="min-w-0">
            <div className="text-sm font-medium text-ink-main">Drag & drop a 12-lead ECG (.npy, .csv, .hdf5)</div>
            <div className="truncate text-[11px] text-ink-soft">{sourceLabel}</div>
          </div>
        </div>
        {previewLoading ? (
          <div className="flex items-center justify-center gap-2 rounded-2xl border border-black/5 bg-paper py-10 text-sm text-ink-muted">
            <Loader2 className="h-4 w-4 animate-spin" /> Reading 12-lead ECG…
          </div>
        ) : preview ? (
          <ECGLeadViewer
            leadsPreview={preview.leads_preview}
            dsp={preview.dsp_measurements}
            isStaffMode={isStaffMode}
            durationSec={preview.duration_s}
            title="ECG preview"
            subtitle="Saliency heatmap appears after analysis"
          />
        ) : null}
      </section>

      {/* ── Step 2: CODE-15% flags ─────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-ink-main">
          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-ink-main text-[10px] text-paper">2</span>
          CODE-15% clinical arrhythmia findings
          <span className="text-[11px] font-normal text-ink-soft">(from the hospital ECG report)</span>
        </h3>
        <div className="flex flex-wrap gap-2">
          {CODE15_CODES.map(code => (
            <motion.button
              key={code}
              whileTap={{ scale: 0.96 }}
              onClick={() => setFlags(f => ({ ...f, [code]: !f[code] }))}
              aria-pressed={flags[code]}
              className={`rounded-xl border px-3 py-2 text-left text-xs transition ${flags[code] ? 'border-rose-dark/30 bg-rose-light/60 text-rose-dark' : 'border-black/5 bg-paper text-ink-muted hover:border-black/15'}`}
            >
              <div className="font-mono font-bold">{code}</div>
              <div className="text-[10px]">{CODE15_LABELS[code]}</div>
            </motion.button>
          ))}
        </div>
      </section>

      {/* ── Step 3: Blood report & vitals ──────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-ink-main">
          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-ink-main text-[10px] text-paper">3</span>
          Routine blood report & vitals
        </h3>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
          {FIELD_DEFS.map(f => (
            <label key={f.key} className="text-[11px] font-medium text-ink-muted">
              {f.label}{f.required && <span className="text-rose-dark"> *</span>}
              <div className="relative mt-1">
                <input
                  type="number" step={f.step ?? '1'} inputMode="decimal"
                  className={`${inputCls} pr-12`} value={values[f.key]}
                  onChange={e => setValues(v => ({ ...v, [f.key]: e.target.value }))}
                />
                <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-ink-soft">{f.unit}</span>
              </div>
            </label>
          ))}
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <label className="text-[11px] font-medium text-ink-muted">Diet preference
            <select className={`${inputCls} mt-1`} value={dietPref} onChange={e => setDietPref(e.target.value as DietPreference)}>
              {DIET_OPTIONS.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
            </select>
          </label>
          <label className="text-[11px] font-medium text-ink-muted">Hospital name
            <input className={`${inputCls} mt-1`} maxLength={200} value={hospital} onChange={e => setHospital(e.target.value)} />
          </label>
          <label className="text-[11px] font-medium text-ink-muted">Visit date
            <input type="date" className={`${inputCls} mt-1`} value={visitDate} onChange={e => setVisitDate(e.target.value)} />
          </label>
        </div>
      </section>

      {/* ── Step 4: adherence to the previous plan ─────────────────────── */}
      <section className="flex flex-col gap-3">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-ink-main">
          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-ink-main text-[10px] text-paper">4</span>
          Did the patient follow the previous diet & exercise plan?
        </h3>
        {lastPlan ? (
          <div className="rounded-xl border border-black/5 bg-paper px-4 py-3 text-xs text-ink-muted">
            <span className="font-semibold text-ink-main">Plan given on {lastVisit?.visit_date}:</span>{' '}
            sodium under {lastPlan.daily_targets.sodium_mg} mg/day, {lastPlan.daily_targets.fluid_liters} L fluid,{' '}
            {lastPlan.exercise_plan.allowed_activity.toLowerCase()} {lastPlan.exercise_plan.duration_mins} min (max HR {lastPlan.exercise_plan.max_hr_ceiling} bpm)
            {lastVisit?.approved_by_staff ? ' · ✅ hospital verified' : ''}
          </div>
        ) : (
          <div className="rounded-xl border border-black/5 bg-paper px-4 py-3 text-xs text-ink-muted">
            No earlier visit for this patient — this report becomes the baseline visit, so you can skip this step.
          </div>
        )}
        {lastPlan && (
          <div className="flex flex-wrap gap-2">
            {ADHERENCE_OPTIONS.map(o => (
              <button key={o.value} onClick={() => setAdherence(a => (a === o.value ? '' : o.value))} aria-pressed={adherence === o.value}
                className={`rounded-xl border px-3 py-2 text-xs font-semibold transition ${adherence === o.value ? o.cls : 'border-black/5 bg-paper text-ink-muted hover:border-black/15'}`}>
                {o.label}
              </button>
            ))}
          </div>
        )}
        <label className="text-[11px] font-medium text-ink-muted">Notes (optional)
          <textarea
            rows={2} maxLength={1800} className={`${inputCls} mt-1 resize-none`} value={notes}
            placeholder="e.g. skipped evening walks for 5 weeks and ate salty restaurant food"
            onChange={e => setNotes(e.target.value)}
          />
        </label>
        <div className="flex flex-wrap gap-1.5">
          {ADHERENCE_CHIPS.map(c => (
            <button key={c} onClick={() => setNotes(n => (n ? `${n} ${c}.` : `${c}.`))} className="rounded-full border border-black/5 bg-paper px-2.5 py-1 text-[11px] text-ink-muted hover:text-ink-main">
              + {c}
            </button>
          ))}
        </div>
      </section>

      {error && <div className="rounded-xl border border-rose-dark/20 bg-rose-light/50 px-4 py-3 text-xs text-rose-dark">{error}</div>}

      <motion.button
        whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.98 }}
        disabled={submitting || !activeProfile}
        onClick={submit}
        className="flex items-center justify-center gap-2 rounded-xl bg-ink-main py-3.5 text-sm font-medium text-paper shadow-[0_4px_14px_rgba(44,41,38,0.2)] transition hover:bg-ink-main/90 disabled:opacity-60"
      >
        {submitting
          ? <><Loader2 className="h-4 w-4 animate-spin" /> Running CODE-15% multimodal analysis…</>
          : <>{isStaffMode ? <Stethoscope className="h-4 w-4" /> : <HeartPulse className="h-4 w-4" />} Analyze & Save Visit</>}
      </motion.button>
    </div>
  );
}
