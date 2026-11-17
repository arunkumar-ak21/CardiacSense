const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export async function predictVitals(vitals: Record<string, unknown>) {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(vitals),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Prediction failed' }))
    throw new Error(err.detail || err.error || 'Prediction failed')
  }
  return res.json()
}

export async function parsePDF(file: File) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE_URL}/parse-pdf`, { method: 'POST', body: form })
  if (!res.ok) throw new Error('PDF parsing failed')
  return res.json()
}

export async function voiceToVitals(audioBlob: Blob, mimeType: string) {
  const form = new FormData()
  form.append('audio', audioBlob, 'recording.webm')
  // Override content type header is set by browser for FormData
  void mimeType
  const res = await fetch(`${BASE_URL}/voice-to-vitals`, { method: 'POST', body: form })
  if (!res.ok) throw new Error('Voice parsing failed')
  return res.json()
}

export async function* streamNarrative(payload: object): AsyncGenerator<string> {
  const res = await fetch(`${BASE_URL}/summarize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok || !res.body) return
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n\n')
    buffer = lines.pop() || ''
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6))
          if (data.token) yield data.token
          if (data.done) return
        } catch {
          // skip malformed chunk
        }
      }
    }
  }
}

export async function* streamChat(
  messages: Array<{ role: string; content: string }>,
  vitalsContext: object,
  predictionsContext: object,
  longitudinal?: object | null,
): AsyncGenerator<string> {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      vitals_context: vitalsContext,
      predictions_context: predictionsContext,
      ...(longitudinal ?? {}),
    }),
  })
  if (!res.ok || !res.body) return
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n\n')
    buffer = lines.pop() || ''
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6))
          if (data.token) yield data.token
          if (data.done) return
        } catch {
          // skip malformed chunk
        }
      }
    }
  }
}

export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`)
  if (!res.ok) throw new Error('Backend unreachable')
  return res.json()
}

export async function storeHistory(patientId: string, predictionResult: object, timestamp: string) {
  const res = await fetch(`${BASE_URL}/patient/history`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ patient_id: patientId, prediction_result: predictionResult, timestamp }),
  })
  if (!res.ok) return null
  return res.json()
}

export async function getHistory(patientId: string, limit = 20) {
  const res = await fetch(`${BASE_URL}/patient/${patientId}/history?limit=${limit}`)
  if (!res.ok) return { records: [] as HistoryRecord[], total: 0 }
  return res.json() as Promise<{ records: HistoryRecord[]; total: number }>
}

export interface HistoryRecord {
  record_id: string
  timestamp: string
  overall_risk_score: number
  overall_risk_label: string
  primary_condition: string
  vitals: { hr: number; bp_systolic: number; bp_diastolic: number; spo2: number; rr: number; temp: number }
}

export async function getRecommendations(vitals: object, conditions: object[], shap: object[], narrative?: string) {
  const res = await fetch(`${BASE_URL}/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ vitals, conditions, shap, narrative: narrative || '' }),
  })
  if (!res.ok) return null
  return res.json() as Promise<Array<{ label: string; badge: string; type: 'urgent' | 'moderate' | 'routine' }>>
}

// ── ECG Analysis types ────────────────────────────────────────────────────────
export interface EcgDemo {
  ecg_id: number
  superclass: string
  label: string
  age: number
  sex: string
  description: string
}

export interface EcgSignalData {
  ecg_id: number
  lead_names: string[]
  signals: number[][]  // [12 leads][1000 samples]
  fs: number
}

export interface EcgFinding {
  class: string
  label: string
  probability: number
}

export interface EcgPredictionResult {
  ecg_findings: EcgFinding[]
  primary_finding: EcgFinding
  model: string
  macro_auc: number
  inference_time_ms: number
}

// ── ECG API functions ─────────────────────────────────────────────────────────
export async function fetchEcgDemos(): Promise<EcgDemo[]> {
  const res = await fetch(`${BASE_URL}/ecg/demos`)
  if (!res.ok) throw new Error('Failed to load demo records')
  const data = await res.json()
  return data.demos
}

export async function fetchEcgDemoSignal(ecgId: number): Promise<EcgSignalData> {
  const res = await fetch(`${BASE_URL}/ecg/demo/${ecgId}/signal`)
  if (!res.ok) throw new Error('Failed to load ECG signal')
  return res.json()
}

export async function runEcgDemoPredict(ecgId: number): Promise<EcgPredictionResult> {
  const res = await fetch(`${BASE_URL}/ecg/demo/${ecgId}/predict`)
  if (!res.ok) throw new Error('ECG prediction failed')
  return res.json()
}

export async function runEcgFilePredict(file: File): Promise<EcgPredictionResult> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE_URL}/predict/ecg`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'ECG prediction failed' }))
    throw new Error(typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail))
  }
  return res.json()
}

// ══════════════════════════════════════════════════════════════════════════════
// Longitudinal companion (companion.db + CODE-15% engine) — Step 3
// ══════════════════════════════════════════════════════════════════════════════
export type DietPreference = 'vegetarian' | 'non_vegetarian' | 'vegan' | 'diabetic' | 'dash'
export type SeverityTier = 'Low' | 'Moderate' | 'High' | 'Critical'
export const CODE15_CODES = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'] as const
export type Code15Code = (typeof CODE15_CODES)[number]
export const CODE15_LABELS: Record<Code15Code, string> = {
  '1dAVb': '1st Degree AV Block',
  RBBB: 'Right Bundle Branch Block',
  LBBB: 'Left Bundle Branch Block',
  SB: 'Sinus Bradycardia',
  AF: 'Atrial Fibrillation',
  ST: 'Sinus Tachycardia',
}
export const LEAD_NAMES = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] as const

export interface PatientProfile {
  id: string
  name: string
  mrn: string
  age: number
  gender: 'M' | 'F' | 'Other' | string
  diet_preference: DietPreference | string
  known_conditions: string[]
  created_at: string | null
  visit_count?: number
  latest_visit_date?: string | null
  latest_risk_score?: number | null
  latest_severity_tier?: SeverityTier | string | null
  latest_plan_approved?: boolean
}

export interface NewPatientInput {
  name: string
  mrn: string
  age: number
  gender: 'M' | 'F' | 'Other'
  diet_preference: DietPreference
  known_conditions?: string[]
}

export interface DspMeasurements {
  heart_rate?: number | null
  qrs_ms?: number | null
  pr_ms?: number | null
  st_amp?: number | null
  st_status?: string | null
  rr_irregularity?: number | null
  measurable?: boolean
  [key: string]: unknown
}

export interface EcgMetadata {
  filename?: string | null
  source?: string
  format?: string
  sample_rate?: number
  n_samples?: number
  lead_names?: string[]
  units?: string
  dsp_measurements?: DspMeasurements
  preview_sample_rate?: number
  leads_preview?: Record<string, number[]>
  [key: string]: unknown
}

export interface VisitPredictions {
  overall_risk_score: number
  severity_tier: SeverityTier | string
  predicted_heart_age: number
  chronological_age: number
  heart_age_gap?: number
  heart_age_source?: string
  arrhythmias: Record<string, number>
  risk_components?: { clinical_risk_component: number; neural_risk_adjustment: number }
  cross_attention_weights?: number[][] | null
  clinical_validation_disclaimer?: string
  [key: string]: unknown
}

export interface ShapDriver {
  feature: string
  display_name: string
  value: number
  shap_score: number
  direction: 'positive' | 'negative' | string
}

export interface BiomarkerDelta {
  key: string
  label: string
  unit: string
  previous: number | null
  current: number | null
  delta: number | null
  delta_pct?: number | null
  status: 'worsened' | 'improved' | 'stable' | string
  clinical_note?: string | null
  display?: string
  [key: string]: unknown
}

export interface ArrhythmiaChange {
  code: string
  name: string
  previous_prob: number
  current_prob: number
}

export interface DeltaSummary {
  is_baseline: boolean
  previous_visit_id: string | null
  previous_visit_date: string | null
  current_visit_date?: string | null
  days_between_visits?: number | null
  previous_risk_score: number | null
  current_risk_score: number | null
  previous_severity_tier?: string | null
  current_severity_tier?: string | null
  risk_delta: number | null
  previous_heart_age?: number | null
  current_heart_age?: number | null
  heart_age_delta: number | null
  overall_trend: string
  biomarker_deltas: BiomarkerDelta[]
  worsened: string[]
  improved: string[]
  stable: string[]
  new_arrhythmias: ArrhythmiaChange[]
  resolved_arrhythmias: ArrhythmiaChange[]
  persistent_arrhythmias?: ArrhythmiaChange[]
  ecg_morphology_deltas?: Array<{
    key: string; label: string; unit: string | null
    previous: number | string | null; current: number | string | null; delta: number | null; status: string
  }>
  what_went_wrong_summary: string
  what_improved_summary: string
  [key: string]: unknown
}

export interface MealSlot { time: string; items: string[]; notes?: string }

export interface DietExercisePlan {
  plan_title?: string
  diet_preference?: string
  daily_targets: {
    sodium_mg: number
    protein_g: number
    fluid_liters: number
    calories: number
    potassium_mg?: number
    fiber_g?: number
    saturated_fat_g?: number
    [key: string]: number | undefined
  }
  meals: Record<string, MealSlot>
  foods_to_avoid: string[]
  exercise_plan: {
    allowed_activity: string
    duration_mins: number
    sessions_per_day?: number
    frequency?: string
    intensity?: string
    max_hr_ceiling: number
    warnings: string[]
  }
  why_plan_changed_summary: string
  clinical_rules_applied?: string[]
  generated_by?: string
  staff_review?: { approved_by_staff: boolean; staff_notes: string | null; reviewed_at: string }
  [key: string]: unknown
}

export interface LongitudinalVisit {
  id: string
  patient_id: string
  visit_date: string
  hospital_name: string
  adherence_notes: string | null
  vitals: { hr?: number | null; bp_sys?: number | null; bp_dia?: number | null; spo2?: number | null; rr?: number | null; temp?: number | null }
  blood: { glucose?: number | null; cholesterol?: number | null; creatinine?: number | null; hrv?: number | null }
  code15_flags: Record<string, boolean>
  ecg_metadata: EcgMetadata
  predictions: VisitPredictions
  shap: ShapDriver[]
  heatmap: number[][] | null
  delta_summary: DeltaSummary | null
  diet_plan: DietExercisePlan
  approved_by_staff: boolean
  staff_notes: string | null
  created_at: string | null
}

export interface Code15Sample {
  filename: string
  format: string
  size_bytes: number
  label?: string | null
  description?: string | null
  expected_code15_flags?: Record<string, boolean> | null
}

async function apiError(res: Response, fallback: string): Promise<Error> {
  const body = await res.json().catch(() => null)
  const detail = body?.detail
  const msg = typeof detail === 'string' ? detail : detail?.detail || detail?.error || fallback
  return new Error(msg)
}

async function loadFallback() {
  const mod = await import('../data/longitudinalFallback')
  return mod.FALLBACK_LONGITUDINAL
}

/** True when the last longitudinal read came from local seed data (backend unreachable). */
export let usingLongitudinalFallback = false

export async function fetchPatients(): Promise<PatientProfile[]> {
  try {
    const res = await fetch(`${BASE_URL}/patients`)
    if (!res.ok) throw await apiError(res, 'Failed to load patients')
    usingLongitudinalFallback = false
    return (await res.json()) as PatientProfile[]
  } catch {
    usingLongitudinalFallback = true
    return (await loadFallback()).patients
  }
}

export async function createPatient(data: NewPatientInput): Promise<PatientProfile> {
  const res = await fetch(`${BASE_URL}/patients`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw await apiError(res, 'Could not create patient')
  return res.json()
}

export async function fetchPatientVisits(patientId: string): Promise<LongitudinalVisit[]> {
  try {
    const res = await fetch(`${BASE_URL}/patients/${encodeURIComponent(patientId)}/visits`)
    if (!res.ok) throw await apiError(res, 'Failed to load visits')
    usingLongitudinalFallback = false
    return ((await res.json()) as { visits: LongitudinalVisit[] }).visits
  } catch {
    usingLongitudinalFallback = true
    return (await loadFallback()).visits[patientId] ?? []
  }
}

export async function fetchCode15Samples(): Promise<Code15Sample[]> {
  const res = await fetch(`${BASE_URL}/ecg/code15-samples`)
  if (!res.ok) throw await apiError(res, 'Failed to load sample ECGs')
  return ((await res.json()) as { samples: Code15Sample[] }).samples
}

export async function submitCode15Visit(formData: FormData): Promise<LongitudinalVisit> {
  const res = await fetch(`${BASE_URL}/predict/code15-multimodal`, { method: 'POST', body: formData })
  if (!res.ok) throw await apiError(res, 'Visit analysis failed')
  return res.json()
}

export async function approveVisitPlan(
  visitId: string,
  payload: { approved_by_staff: boolean; staff_notes: string; diet_plan?: DietExercisePlan },
): Promise<LongitudinalVisit> {
  const res = await fetch(`${BASE_URL}/visits/${encodeURIComponent(visitId)}/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw await apiError(res, 'Approval failed')
  return res.json()
}

export interface Code15Preview {
  leads_preview: Record<string, number[]>
  preview_sample_rate: number
  duration_s: number
  dsp_measurements: DspMeasurements
}

/** 12-lead preview + DSP for a bundled CODE-15% sample (no visit is saved). */
export async function fetchCode15SamplePreview(filename: string): Promise<Code15Preview> {
  const res = await fetch(`${BASE_URL}/ecg/code15-samples/${encodeURIComponent(filename)}/preview`)
  if (!res.ok) throw await apiError(res, 'Could not load sample ECG')
  return res.json()
}

/** 12-lead preview + DSP for an uploaded .npy / .csv / .hdf5 file (no visit is saved). */
export async function previewCode15Upload(file: File, sampleRate = 400): Promise<Code15Preview> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE_URL}/ecg/code15-preview?sample_rate=${sampleRate}`, { method: 'POST', body: form })
  if (!res.ok) throw await apiError(res, 'Could not read the ECG file')
  return res.json()
}
