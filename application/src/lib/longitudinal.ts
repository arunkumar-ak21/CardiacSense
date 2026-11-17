import type { DeltaSummary, LongitudinalVisit, PatientProfile } from './api';

export function fmt(value: number | null | undefined, digits = 1): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

export function signed(value: number | null | undefined, digits = 1): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return `${value > 0 ? '+' : ''}${value.toFixed(digits)}`;
}

export type DeltaStatus = 'worsened' | 'improved' | 'stable';

export const STATUS_META: Record<DeltaStatus, { label: string; chip: string; text: string }> = {
  worsened: { label: '🔴 Worsened', chip: 'bg-rose-light/70 text-rose-dark border border-rose-dark/15', text: 'text-rose-dark' },
  improved: { label: '🟢 Improved', chip: 'bg-sage-light text-sage-dark border border-sage-dark/15', text: 'text-sage-dark' },
  stable:   { label: '⚪ Stable',   chip: 'bg-sand-light text-ink-muted border border-black/5',      text: 'text-ink-muted' },
};

export function statusOf(value: string | undefined | null): DeltaStatus {
  return value === 'worsened' || value === 'improved' ? value : 'stable';
}

/** Lower-is-better comparison used when the stored delta summary has no entry. */
export function compareLowerBetter(prev: number | null | undefined, curr: number | null | undefined, stablePct = 3): DeltaStatus {
  if (typeof prev !== 'number' || typeof curr !== 'number') return 'stable';
  if (prev === 0) return curr === 0 ? 'stable' : curr > 0 ? 'worsened' : 'improved';
  const pct = (Math.abs(curr - prev) / Math.abs(prev)) * 100;
  if (pct < stablePct) return 'stable';
  return curr > prev ? 'worsened' : 'improved';
}

export function biomarker(delta: DeltaSummary | null | undefined, key: string) {
  return delta?.biomarker_deltas?.find(b => b.key === key) ?? null;
}

/** Visit without the heavy waveform/heatmap arrays — for chat/summary payloads. */
export function slimVisit(v: LongitudinalVisit | null): Record<string, unknown> | null {
  if (!v) return null;
  const { leads_preview: _lp, ...meta } = v.ecg_metadata ?? {};
  void _lp;
  const { cross_attention_weights: _cw, ...preds } = v.predictions ?? ({} as LongitudinalVisit['predictions']);
  void _cw;
  return { ...v, heatmap: null, ecg_metadata: meta, predictions: preds };
}

export function buildLongitudinalPayload(
  profile: PatientProfile | null, active: LongitudinalVisit | null, previous: LongitudinalVisit | null, isStaffMode: boolean,
) {
  if (!profile || !active) return null;
  return {
    patient_id: profile.id,
    patient_profile: profile,
    current_visit: slimVisit(active),
    previous_visit: slimVisit(previous),
    deltas: active.delta_summary,
    diet_plan: active.diet_plan,
    is_staff_mode: isStaffMode,
  };
}

/** 1-based chronological visit number (#1 = oldest). */
export function visitNumber(visits: LongitudinalVisit[], id: string | undefined): number {
  const idx = visits.findIndex(v => v.id === id);
  return idx < 0 ? 0 : visits.length - idx;
}
