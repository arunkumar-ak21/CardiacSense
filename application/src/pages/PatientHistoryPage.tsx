import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Activity, AlertTriangle, CalendarClock, CheckCircle2, LineChart, TrendingUp } from 'lucide-react';
import { useVitals } from '../context/VitalsContext';
import LongitudinalShell, { Card, EmptyVisits, RiskDeltaBadge, TierBadge } from '../components/longitudinal/LongitudinalShell';
import { CODE15_LABELS, type Code15Code, type LongitudinalVisit } from '../lib/api';
import {
  STATUS_META, biomarker, compareLowerBetter, fmt, signed, statusOf, visitNumber, type DeltaStatus,
} from '../lib/longitudinal';

// ── Delta grid card ───────────────────────────────────────────────────────────
function DeltaCard({ label, prev, curr, unit, delta, status }: {
  label: string; prev: string; curr: string; unit: string; delta: string; status: DeltaStatus;
}) {
  const meta = STATUS_META[status];
  return (
    <div className="rounded-xl border border-black/5 bg-ivory p-4">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft">{label}</span>
        <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${meta.chip}`}>{meta.label}</span>
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="text-sm text-ink-muted line-through decoration-black/20">{prev}</span>
        <span className="text-ink-soft">→</span>
        <span className="font-serif text-2xl text-ink-main">{curr}</span>
        <span className="text-[11px] text-ink-soft">{unit}</span>
      </div>
      <div className={`mt-1 font-mono text-xs ${meta.text}`}>{delta}</div>
    </div>
  );
}

// ── Small-multiple SVG trend chart ────────────────────────────────────────────
function TrendChart({ visits, label, unit, color, pick, refLine, digits = 0 }: {
  visits: LongitudinalVisit[]; label: string; unit: string; color: string;
  pick: (v: LongitudinalVisit) => number | null | undefined; refLine?: number; digits?: number;
}) {
  const W = 300, H = 120, PX = 24, PY = 18;
  const pts = visits.map(v => ({ v, y: pick(v) })).filter((p): p is { v: LongitudinalVisit; y: number } => typeof p.y === 'number');
  const values = pts.map(p => p.y).concat(refLine != null ? [refLine] : []);
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const x = (i: number) => (pts.length <= 1 ? W / 2 : PX + (i / (pts.length - 1)) * (W - 2 * PX));
  const y = (val: number) => H - PY - ((val - min) / span) * (H - 2 * PY);
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.y).toFixed(1)}`).join(' ');
  return (
    <div className="rounded-xl border border-black/5 bg-ivory p-3">
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="font-medium text-ink-main">{label}</span>
        <span className="font-mono text-ink-soft">{unit}</span>
      </div>
      {pts.length === 0 ? <div className="py-8 text-center text-xs text-ink-soft">No data</div> : (
        <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full">
          {refLine != null && (
            <>
              <line x1={PX} x2={W - PX} y1={y(refLine)} y2={y(refLine)} stroke="#b9ad9a" strokeDasharray="3 4" />
              <text x={W - PX} y={y(refLine) - 4} textAnchor="end" fontSize="9" fill="#8a8073">target {refLine}</text>
            </>
          )}
          <motion.path d={d} fill="none" stroke={color} strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round"
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1 }} />
          {pts.map((p, i) => (
            <g key={p.v.id}>
              <circle cx={x(i)} cy={y(p.y)} r={4} fill="#fff" stroke={color} strokeWidth={2} />
              <text x={x(i)} y={y(p.y) - 8} textAnchor="middle" fontSize="10" fontWeight={600} fill="#2C2926">{p.y.toFixed(digits)}</text>
              <text x={x(i)} y={H - 3} textAnchor="middle" fontSize="9" fill="#8a8073">{p.v.visit_date.slice(5)}</text>
            </g>
          ))}
        </svg>
      )}
    </div>
  );
}

export default function PatientHistoryPage() {
  const navigate = useNavigate();
  const { activeProfile, visits, activeVisit, previousVisit, selectVisit, isStaffMode } = useVitals();

  if (!activeVisit || !activeProfile) {
    return <LongitudinalShell title="Before vs. Now — Patient History"><EmptyVisits /></LongitudinalShell>;
  }

  const delta = activeVisit.delta_summary;
  const chronological = [...visits].reverse();
  const curV = activeVisit.vitals, curB = activeVisit.blood;
  const prvV = previousVisit?.vitals, prvB = previousVisit?.blood;

  const bioStatus = (key: string, prev?: number | null, curr?: number | null) =>
    statusOf(biomarker(delta, key)?.status ?? compareLowerBetter(prev, curr));
  const riskStatus: DeltaStatus = delta?.overall_trend === 'worsened' ? 'worsened' : delta?.overall_trend === 'improved' ? 'improved' : 'stable';
  const hr = biomarker(delta, 'hr');

  const cards = previousVisit ? [
    { label: 'SCD Risk %', prev: `${fmt(previousVisit.predictions.overall_risk_score)}%`, curr: `${fmt(activeVisit.predictions.overall_risk_score)}%`, unit: '', delta: `${signed(delta?.risk_delta)} pts`, status: riskStatus },
    { label: 'Predicted Heart Age', prev: fmt(previousVisit.predictions.predicted_heart_age), curr: fmt(activeVisit.predictions.predicted_heart_age), unit: 'yrs',
      delta: `${signed(delta?.heart_age_delta)} yrs`, status: compareLowerBetter(previousVisit.predictions.predicted_heart_age, activeVisit.predictions.predicted_heart_age) },
    { label: 'Blood Pressure', prev: `${prvV?.bp_sys ?? '—'}/${prvV?.bp_dia ?? '—'}`, curr: `${curV.bp_sys ?? '—'}/${curV.bp_dia ?? '—'}`, unit: 'mmHg',
      delta: `${signed((curV.bp_sys ?? 0) - (prvV?.bp_sys ?? 0), 0)} / ${signed((curV.bp_dia ?? 0) - (prvV?.bp_dia ?? 0), 0)} mmHg`, status: bioStatus('bp_sys', prvV?.bp_sys, curV.bp_sys) },
    { label: 'Serum Creatinine', prev: fmt(prvB?.creatinine, 2), curr: fmt(curB.creatinine, 2), unit: 'mg/dL',
      delta: `${signed((curB.creatinine ?? 0) - (prvB?.creatinine ?? 0), 2)} mg/dL`, status: bioStatus('creatinine', prvB?.creatinine, curB.creatinine) },
    { label: 'Fasting Blood Glucose', prev: fmt(prvB?.glucose, 0), curr: fmt(curB.glucose, 0), unit: 'mg/dL',
      delta: `${signed((curB.glucose ?? 0) - (prvB?.glucose ?? 0), 0)} mg/dL`, status: bioStatus('glucose', prvB?.glucose, curB.glucose) },
    { label: 'Total Cholesterol', prev: fmt(prvB?.cholesterol, 0), curr: fmt(curB.cholesterol, 0), unit: 'mg/dL',
      delta: `${signed((curB.cholesterol ?? 0) - (prvB?.cholesterol ?? 0), 0)} mg/dL`, status: bioStatus('cholesterol', prvB?.cholesterol, curB.cholesterol) },
    { label: 'Heart Rate', prev: fmt(prvV?.hr, 0), curr: fmt(curV.hr, 0), unit: 'bpm',
      delta: `${signed((curV.hr ?? 0) - (prvV?.hr ?? 0), 0)} bpm`, status: statusOf(hr?.status) },
  ] : [];

  const changes: { code: string; kind: 'new' | 'resolved' | 'persistent' }[] = [
    ...(delta?.new_arrhythmias ?? []).map(a => ({ code: a.code, kind: 'new' as const })),
    ...(delta?.resolved_arrhythmias ?? []).map(a => ({ code: a.code, kind: 'resolved' as const })),
    ...(delta?.persistent_arrhythmias ?? []).map(a => ({ code: a.code, kind: 'persistent' as const })),
  ];
  const morph = (delta?.ecg_morphology_deltas ?? []).filter(m => isStaffMode || m.status !== 'stable');

  return (
    <LongitudinalShell
      title="Before vs. Now — What Went Wrong / What Improved"
      subtitle={<>{activeProfile.name} · {visits.length} hospital visit{visits.length === 1 ? '' : 's'} · comparing Visit #{visitNumber(visits, activeVisit.id)} ({activeVisit.visit_date}){previousVisit ? ` with Visit #${visitNumber(visits, previousVisit.id)} (${previousVisit.visit_date})` : ''}</>}
      actions={<button onClick={() => navigate('/vitals')} className="rounded-xl bg-ink-main px-4 py-2.5 text-sm font-medium text-paper hover:bg-ink-main/90">+ Add New Visit Report</button>}
    >
      <div className="grid gap-6 lg:grid-cols-[280px_minmax(0,1fr)]">
        {/* ── Interactive timeline ─────────────────────────────────────── */}
        <Card title="Visit Timeline" icon={<CalendarClock className="h-4 w-4 text-ink-muted" />} className="h-fit">
          <ol className="relative ml-2 border-l border-black/10">
            {visits.map(v => {
              const active = v.id === activeVisit.id;
              return (
                <li key={v.id} className="mb-4 ml-4">
                  <span className={`absolute -left-[7px] mt-1.5 h-3 w-3 rounded-full border-2 border-paper ${active ? 'bg-ink-main' : 'bg-sand-dark/50'}`} />
                  <button onClick={() => selectVisit(v.id)}
                    className={`w-full rounded-xl border p-3 text-left transition ${active ? 'border-ink-main/30 bg-cream shadow-sm' : 'border-black/5 bg-ivory hover:bg-cream'}`}>
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[11px] font-semibold text-ink-main">Visit #{visitNumber(visits, v.id)}</span>
                      <TierBadge tier={v.predictions.severity_tier} className="!text-[9px]" />
                    </div>
                    <div className="mt-1 text-xs text-ink-muted">{v.visit_date} · {fmt(v.predictions.overall_risk_score)}%</div>
                    <div className="truncate text-[10px] text-ink-soft">{v.hospital_name}</div>
                  </button>
                </li>
              );
            })}
          </ol>
        </Card>

        <div className="flex flex-col gap-6">
          {/* ── Trajectory summary banner ──────────────────────────────── */}
          <Card>
            <div className="flex flex-wrap items-center gap-3">
              <RiskDeltaBadge delta={delta} large />
              {previousVisit && <span className="text-xs text-ink-muted">{previousVisit.predictions.severity_tier} → {activeVisit.predictions.severity_tier}</span>}
            </div>
            {previousVisit ? (
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <div className="rounded-xl border border-rose-dark/10 bg-rose-light/25 p-4">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-rose-dark"><AlertTriangle className="h-3.5 w-3.5" /> What went wrong</div>
                  <p className="text-sm leading-relaxed text-ink-main">{delta?.what_went_wrong_summary}</p>
                </div>
                <div className="rounded-xl border border-sage-dark/10 bg-sage-light/40 p-4">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-sage-dark"><CheckCircle2 className="h-3.5 w-3.5" /> What improved</div>
                  <p className="text-sm leading-relaxed text-ink-main">{delta?.what_improved_summary}</p>
                </div>
              </div>
            ) : (
              <p className="mt-3 text-sm text-ink-muted">This is the first recorded visit. Changes will be tracked from the next visit onward.</p>
            )}
            {activeVisit.adherence_notes && (
              <div className="mt-3 rounded-xl border border-black/5 bg-cream px-4 py-3 text-sm">
                <span className="font-mono text-[10px] uppercase tracking-widest text-ink-soft">Adherence to previous plan · </span>
                <span className="italic text-ink-main">“{activeVisit.adherence_notes}”</span>
              </div>
            )}
          </Card>

          {/* ── Delta grid ─────────────────────────────────────────────── */}
          {previousVisit && (
            <Card title="Biomarker & Vitals Delta" icon={<TrendingUp className="h-4 w-4 text-ink-muted" />}>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {cards.map(c => <DeltaCard key={c.label} {...c} />)}
              </div>
            </Card>
          )}

          {/* ── Arrhythmia / morphology shift ──────────────────────────── */}
          {previousVisit && (
            <Card title="12-Lead ECG Arrhythmia & Morphology Shift" icon={<Activity className="h-4 w-4 text-rose-dark" />}>
              {changes.length === 0 && morph.length === 0 ? (
                <p className="text-sm text-ink-muted">No CODE-15% arrhythmia or ECG morphology change between these visits.</p>
              ) : (
                <div className="flex flex-col gap-3">
                  <div className="flex flex-wrap gap-2">
                    {changes.map(c => (
                      <span key={`${c.kind}-${c.code}`} className={`rounded-xl border px-3 py-1.5 text-xs font-semibold ${
                        c.kind === 'new' ? 'border-rose-dark/20 bg-rose-light/60 text-rose-dark' : c.kind === 'resolved' ? 'border-sage-dark/20 bg-sage-light text-sage-dark' : 'border-amber-dark/20 bg-amber-light text-amber-dark'}`}>
                        {c.kind === 'new' ? '🆕 Newly appeared' : c.kind === 'resolved' ? '✅ Resolved' : '⏺ Persistent'}: {c.code} — {CODE15_LABELS[c.code as Code15Code] ?? c.code}
                      </span>
                    ))}
                  </div>
                  {morph.length > 0 && (
                    <table className="w-full text-left text-xs">
                      <thead className="font-mono text-[10px] uppercase tracking-widest text-ink-soft">
                        <tr><th className="py-1">Measure</th><th>Previous</th><th>Current</th><th>Change</th></tr>
                      </thead>
                      <tbody>
                        {morph.map(m => (
                          <tr key={m.key} className="border-t border-black/5">
                            <td className="py-1.5 font-medium text-ink-main">{m.label}</td>
                            <td className="text-ink-muted">{m.previous ?? '—'} {m.unit ?? ''}</td>
                            <td className="text-ink-main">{m.current ?? '—'} {m.unit ?? ''}</td>
                            <td className={STATUS_META[statusOf(m.status)].text}>{m.status}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              )}
            </Card>
          )}

          {/* ── Multi-visit trends ─────────────────────────────────────── */}
          <Card title="Multi-Visit Trends" icon={<LineChart className="h-4 w-4 text-ink-muted" />}>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <TrendChart visits={chronological} label="Cardiac Risk" unit="%" color="#8a4b4b" pick={v => v.predictions.overall_risk_score} digits={1} />
              <TrendChart visits={chronological} label="Systolic BP" unit="mmHg" color="#6b5fa8" pick={v => v.vitals.bp_sys} refLine={130} />
              <TrendChart visits={chronological} label="Fasting Glucose" unit="mg/dL" color="#c99a3c" pick={v => v.blood.glucose} refLine={100} />
              <TrendChart visits={chronological} label="Total Cholesterol" unit="mg/dL" color="#6f8f6f" pick={v => v.blood.cholesterol} refLine={200} />
            </div>
          </Card>
        </div>
      </div>
    </LongitudinalShell>
  );
}
