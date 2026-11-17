import { motion } from 'framer-motion';
import { Activity, BrainCircuit, HeartPulse, Info, Scale } from 'lucide-react';
import { useVitals } from '../../context/VitalsContext';
import ECGLeadViewer from '../vitals/ECGLeadViewer';
import LongitudinalShell, { Card, RiskDeltaBadge, TierBadge, VisitSelector } from './LongitudinalShell';
import { CODE15_CODES, CODE15_LABELS } from '../../lib/api';
import { dietLabel } from '../PatientProfileSwitcher';
import { fmt, signed } from '../../lib/longitudinal';

const GAUGE_LEN = 283; // half-circle path length

function riskColor(score: number): string {
  if (score >= 75) return '#8a4b4b';
  if (score >= 50) return '#c0705f';
  if (score >= 25) return '#c99a3c';
  return '#6f8f6f';
}

export default function Code15PredictionsView() {
  const { activeProfile, activeVisit, previousVisit, isStaffMode } = useVitals();
  if (!activeVisit || !activeProfile) return null;

  const p = activeVisit.predictions;
  const prevP = previousVisit?.predictions;
  const delta = activeVisit.delta_summary;
  const gap = p.heart_age_gap ?? p.predicted_heart_age - p.chronological_age;
  const dsp = activeVisit.ecg_metadata?.dsp_measurements;
  const shap = [...(activeVisit.shap ?? [])].sort((a, b) => Math.abs(b.shap_score) - Math.abs(a.shap_score));
  const maxShap = Math.max(0.01, ...shap.map(s => Math.abs(s.shap_score)));
  const leads = activeVisit.ecg_metadata?.leads_preview;

  return (
    <LongitudinalShell
      title="CODE-15% Risk, Heart Age & 12-Lead Explainability"
      subtitle={<>{activeProfile.name} · {activeProfile.age}{activeProfile.gender} · {dietLabel(activeProfile.diet_preference)} · MRN {activeProfile.mrn}</>}
      actions={<VisitSelector />}
    >
      {/* ── Top summary banner ─────────────────────────────────────────── */}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <Card>
          <div className="flex flex-wrap items-center gap-6">
            <div className="relative h-[120px] w-[220px] shrink-0">
              <svg viewBox="0 0 200 110" className="h-full w-full">
                <path d="M10 100 A90 90 0 0 1 190 100" fill="none" stroke="#ece5da" strokeWidth="14" strokeLinecap="round" />
                <motion.path
                  d="M10 100 A90 90 0 0 1 190 100" fill="none" stroke={riskColor(p.overall_risk_score)} strokeWidth="14" strokeLinecap="round"
                  strokeDasharray={GAUGE_LEN}
                  initial={{ strokeDashoffset: GAUGE_LEN }}
                  animate={{ strokeDashoffset: GAUGE_LEN * (1 - p.overall_risk_score / 100) }}
                  transition={{ duration: 1.2, ease: 'easeOut' }}
                />
              </svg>
              <div className="absolute inset-x-0 bottom-0 text-center">
                <div className="font-serif text-4xl leading-none text-ink-main">{fmt(p.overall_risk_score)}%</div>
                <div className="font-mono text-[9px] uppercase tracking-widest text-ink-soft">SCD / cardiac risk</div>
              </div>
            </div>
            <div className="flex min-w-0 flex-1 flex-col gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <TierBadge tier={p.severity_tier} />
                <span className="text-xs text-ink-soft">Visit {activeVisit.visit_date} · {activeVisit.hospital_name}</span>
              </div>
              <RiskDeltaBadge delta={delta} large />
              {prevP ? (
                <div className="text-sm text-ink-main">
                  Previously <span className="font-semibold">{fmt(prevP.overall_risk_score)}% {prevP.severity_tier} Risk</span>
                  {' → '}Now <span className="font-semibold">{fmt(p.overall_risk_score)}% {p.severity_tier} Risk</span>
                </div>
              ) : (
                <div className="text-sm text-ink-muted">First visit on record — the next visit will be compared with this one.</div>
              )}
              {delta && !delta.is_baseline && (
                <div className="flex flex-wrap gap-1.5">
                  {(delta.new_arrhythmias ?? []).map(a => (
                    <span key={a.code} className="rounded-full bg-rose-light/70 px-2 py-0.5 text-[11px] font-semibold text-rose-dark">🆕 {a.name}</span>
                  ))}
                  {(delta.biomarker_deltas ?? []).filter(b => b.status === 'worsened').slice(0, 4).map(b => (
                    <span key={b.key} className="rounded-full bg-rose-light/40 px-2 py-0.5 text-[11px] text-rose-dark">▲ {b.label} {b.previous} → {b.current}</span>
                  ))}
                  {(delta.biomarker_deltas ?? []).filter(b => b.status === 'improved').slice(0, 3).map(b => (
                    <span key={b.key} className="rounded-full bg-sage-light px-2 py-0.5 text-[11px] text-sage-dark">▼ {b.label} {b.previous} → {b.current}</span>
                  ))}
                </div>
              )}
              {isStaffMode && p.risk_components && (
                <div className="font-mono text-[11px] text-ink-soft">
                  Clinical component {fmt(p.risk_components.clinical_risk_component)} + neural cross-attention adj. {signed(p.risk_components.neural_risk_adjustment)} pts
                </div>
              )}
            </div>
          </div>
        </Card>

        {/* ── ECG heart age card ───────────────────────────────────────── */}
        <Card title="CODE-15% ECG Heart Age" icon={<HeartPulse className="h-4 w-4 text-rose-dark" />}>
          <div className="flex items-end gap-6">
            <div>
              <div className="font-serif text-4xl text-ink-main">{fmt(p.predicted_heart_age)}<span className="text-base text-ink-soft"> yrs</span></div>
              <div className="text-xs text-ink-muted">Predicted biological ECG heart age</div>
            </div>
            <div>
              <div className="font-serif text-2xl text-ink-muted">{p.chronological_age}<span className="text-sm"> yrs</span></div>
              <div className="text-xs text-ink-muted">Actual age</div>
            </div>
          </div>
          <div className={`mt-3 rounded-xl px-3 py-2 text-sm font-semibold ${gap > 2 ? 'bg-rose-light/60 text-rose-dark' : gap < -2 ? 'bg-sage-light text-sage-dark' : 'bg-sand-light text-ink-muted'}`}>
            {signed(gap)} yr {gap > 2 ? 'Accelerated Cardiac Aging' : gap < -2 ? 'Younger-than-age heart' : 'In line with actual age'}
          </div>
          {delta?.heart_age_delta != null && (
            <div className="mt-2 text-xs text-ink-muted">
              Since last visit: {fmt(delta.previous_heart_age)} → {fmt(delta.current_heart_age)} yrs ({signed(delta.heart_age_delta)} yrs)
            </div>
          )}
          {isStaffMode && <div className="mt-2 font-mono text-[10px] text-ink-soft">Source: {p.heart_age_source ?? 'resnet1d'} · ResNet1d (MODELS/model-1)</div>}
        </Card>
      </div>

      {/* ── Arrhythmias + SHAP ─────────────────────────────────────────── */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="CODE-15% Arrhythmia Probabilities" icon={<Activity className="h-4 w-4 text-lavender-dark" />}>
          <div className="flex flex-col gap-3">
            {CODE15_CODES.map(code => {
              const prob = p.arrhythmias?.[code] ?? 0;
              const prev = prevP?.arrhythmias?.[code];
              const present = prob >= 0.5 || !!activeVisit.code15_flags?.[code];
              const isNew = present && !!delta?.new_arrhythmias?.some(a => a.code === code);
              return (
                <div key={code}>
                  <div className="mb-1 flex items-center gap-2 text-xs">
                    <span className="w-12 font-mono font-bold text-ink-main">{code}</span>
                    <span className="text-ink-muted">{CODE15_LABELS[code]}</span>
                    <span className="ml-auto font-mono text-ink-main">{(prob * 100).toFixed(0)}%</span>
                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${present ? 'bg-rose-light/70 text-rose-dark' : 'bg-sage-light text-sage-dark'}`}>
                      {isNew ? 'NEW · Detected' : present ? 'Detected' : 'Not detected'}
                    </span>
                  </div>
                  <div className="relative h-2.5 overflow-hidden rounded-full bg-sand-light">
                    <motion.div
                      className={`h-full rounded-full ${present ? 'bg-rose-dark/80' : prob >= 0.3 ? 'bg-amber-main' : 'bg-sage-main'}`}
                      initial={{ width: 0 }} animate={{ width: `${prob * 100}%` }} transition={{ duration: 0.9 }}
                    />
                    {typeof prev === 'number' && <div className="absolute top-0 h-full w-0.5 bg-ink-main/50" style={{ left: `${prev * 100}%` }} title={`Previous visit ${(prev * 100).toFixed(0)}%`} />}
                  </div>
                </div>
              );
            })}
            {prevP && <div className="text-[10px] text-ink-soft">Thin marker = previous visit ({previousVisit?.visit_date})</div>}
          </div>
        </Card>

        <Card title="SHAP Clinical & Blood Biomarker Drivers" icon={<Scale className="h-4 w-4 text-amber-dark" />}>
          <div className="flex flex-col gap-2">
            {shap.length === 0 && <p className="text-sm text-ink-muted">All inputs are at their healthy reference values.</p>}
            {shap.map(s => {
              const pos = s.shap_score >= 0;
              const w = (Math.abs(s.shap_score) / maxShap) * 50;
              return (
                <div key={s.feature} className="grid grid-cols-[minmax(0,180px)_1fr_56px] items-center gap-2 text-xs">
                  <span className="truncate text-ink-main" title={s.display_name}>{s.display_name}</span>
                  <div className="relative h-4">
                    <div className="absolute left-1/2 top-0 h-full w-px bg-black/15" />
                    <motion.div
                      className={`absolute top-0.5 h-3 rounded ${pos ? 'bg-rose-main/90' : 'bg-sage-main/90'}`}
                      style={pos ? { left: '50%' } : { right: '50%' }}
                      initial={{ width: 0 }} animate={{ width: `${w}%` }} transition={{ duration: 0.8 }}
                    />
                  </div>
                  <span className={`text-right font-mono ${pos ? 'text-rose-dark' : 'text-sage-dark'}`}>{signed(s.shap_score * 100)}</span>
                </div>
              );
            })}
            <div className="mt-1 flex justify-between text-[10px] text-ink-soft">
              <span>◀ Protective (lowers risk)</span><span>Risk-increasing ▶ (risk points)</span>
            </div>
          </div>
        </Card>
      </div>

      {/* ── 12-lead viewer + heatmap ───────────────────────────────────── */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-ink-muted">
          <BrainCircuit className="h-4 w-4" /> 12-Lead ECG & Cross-Attention Saliency Heatmap
        </div>
        <ECGLeadViewer
          leadsPreview={leads}
          heatmap={activeVisit.heatmap}
          dsp={dsp}
          isStaffMode={isStaffMode}
          title={activeVisit.ecg_metadata?.filename ? String(activeVisit.ecg_metadata.filename) : '12-Lead ECG'}
          subtitle={`${activeVisit.visit_date} · ${String(activeVisit.ecg_metadata?.source ?? 'CODE-15%')} · amber/red = segments driving the model`}
        />
      </div>

      {p.clinical_validation_disclaimer && (
        <div className="flex items-start gap-2 rounded-xl border border-black/5 bg-cream px-4 py-3 text-[11px] text-ink-muted">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {p.clinical_validation_disclaimer}
        </div>
      )}

    </LongitudinalShell>
  );
}
