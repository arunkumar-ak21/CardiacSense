import type { LongitudinalVisit } from '../../lib/api';
import { CODE15_LABELS, type Code15Code } from '../../lib/api';
import { STATUS_META, fmt, signed, statusOf } from '../../lib/longitudinal';

/** Printable Before-vs-Now comparison + Diet & Exercise plan (rendered inside the PDF area). */
export default function ReportLongitudinalSections({ visit, previous }: { visit: LongitudinalVisit; previous: LongitudinalVisit | null }) {
  const delta = visit.delta_summary;
  const plan = visit.diet_plan;
  const ex = plan.exercise_plan;
  const rows = previous ? [
    { label: 'Cardiac Risk', prev: `${fmt(previous.predictions.overall_risk_score)}% (${previous.predictions.severity_tier})`, curr: `${fmt(visit.predictions.overall_risk_score)}% (${visit.predictions.severity_tier})`, change: `${signed(delta?.risk_delta)} pts`, status: delta?.overall_trend },
    { label: 'ECG Heart Age', prev: `${fmt(previous.predictions.predicted_heart_age)} yrs`, curr: `${fmt(visit.predictions.predicted_heart_age)} yrs`, change: `${signed(delta?.heart_age_delta)} yrs`, status: (delta?.heart_age_delta ?? 0) > 1 ? 'worsened' : (delta?.heart_age_delta ?? 0) < -1 ? 'improved' : 'stable' },
    ...(delta?.biomarker_deltas ?? []).filter(b => ['bp_sys', 'bp_dia', 'hr', 'glucose', 'cholesterol', 'creatinine', 'spo2'].includes(b.key)).map(b => ({
      label: b.label, prev: `${b.previous ?? '—'} ${b.unit}`, curr: `${b.current ?? '—'} ${b.unit}`, change: `${signed(b.delta, b.key === 'creatinine' ? 2 : 0)} ${b.unit}`, status: b.status,
    })),
  ] : [];
  const arrChanges = [
    ...(delta?.new_arrhythmias ?? []).map(a => `New: ${a.code} (${CODE15_LABELS[a.code as Code15Code] ?? a.name})`),
    ...(delta?.resolved_arrhythmias ?? []).map(a => `Resolved: ${a.code} (${CODE15_LABELS[a.code as Code15Code] ?? a.name})`),
  ];

  return (
    <>
      <section className="flex flex-col gap-4">
        <h3 className="font-mono text-[11px] uppercase tracking-widest text-ink-muted">
          Before vs. Now {previous ? `— ${previous.visit_date} → ${visit.visit_date}` : '— baseline visit'}
        </h3>
        {previous ? (
          <>
            <div className="overflow-hidden rounded-2xl border border-black/5 bg-paper">
              <table className="w-full text-left text-sm">
                <thead className="border-b border-black/5 bg-ivory/40 font-mono text-[10px] uppercase tracking-widest text-ink-muted">
                  <tr><th className="px-6 py-3 font-normal">Measure</th><th className="font-normal">Previous</th><th className="font-normal">Now</th><th className="font-normal">Change</th><th className="pr-6 text-right font-normal">Status</th></tr>
                </thead>
                <tbody className="divide-y divide-black/5">
                  {rows.map(r => {
                    const meta = STATUS_META[statusOf(r.status)];
                    return (
                      <tr key={r.label}>
                        <td className="px-6 py-3 font-medium">{r.label}</td>
                        <td className="text-ink-muted">{r.prev}</td>
                        <td className="font-medium text-ink-main">{r.curr}</td>
                        <td className={`font-mono text-xs ${meta.text}`}>{r.change}</td>
                        <td className="pr-6 text-right"><span className={`rounded px-2 py-0.5 text-[10px] font-bold ${meta.chip}`}>{meta.label}</span></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {arrChanges.length > 0 && <div className="text-sm text-ink-main"><span className="font-semibold">12-lead ECG changes:</span> {arrChanges.join(' · ')}</div>}
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-2xl border border-rose-dark/10 bg-rose-light/20 p-4 text-sm leading-relaxed"><div className="mb-1 font-mono text-[10px] uppercase tracking-widest text-rose-dark">What went wrong</div>{delta?.what_went_wrong_summary}</div>
              <div className="rounded-2xl border border-sage-dark/10 bg-sage-light/30 p-4 text-sm leading-relaxed"><div className="mb-1 font-mono text-[10px] uppercase tracking-widest text-sage-dark">What improved</div>{delta?.what_improved_summary}</div>
            </div>
            {visit.adherence_notes && <div className="text-sm italic text-ink-muted">Adherence to previous plan: “{visit.adherence_notes}”</div>}
          </>
        ) : (
          <p className="text-sm text-ink-muted">This is the first recorded visit for this patient.</p>
        )}
      </section>

      <section className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h3 className="font-mono text-[11px] uppercase tracking-widest text-ink-muted">Diet & Exercise Plan</h3>
          <span className={`rounded-md border px-3 py-1 text-[10px] font-bold uppercase tracking-widest ${visit.approved_by_staff ? 'border-sage-dark/20 bg-sage-light text-sage-dark' : 'border-amber-dark/20 bg-amber-light text-amber-dark'}`}>
            {visit.approved_by_staff ? '✓ Approved by Hospital Care Team' : 'AI draft — pending staff review'}
          </span>
        </div>
        {visit.approved_by_staff && visit.staff_notes && <div className="text-xs text-ink-muted">Staff notes: {visit.staff_notes}</div>}
        <div className="grid grid-cols-6 gap-2">
          {([['Sodium', plan.daily_targets.sodium_mg, 'mg'], ['Protein', plan.daily_targets.protein_g, 'g'], ['Fluid', plan.daily_targets.fluid_liters, 'L'],
             ['Calories', plan.daily_targets.calories, 'kcal'], ['Potassium', plan.daily_targets.potassium_mg, 'mg'], ['Fiber', plan.daily_targets.fiber_g, 'g']] as const).map(([l, v, u]) => (
            <div key={l} className="rounded-xl border border-black/5 bg-paper px-3 py-2">
              <div className="text-[10px] uppercase text-ink-soft">{l}</div>
              <div className="font-serif text-lg">{v ?? '—'} <span className="text-xs text-ink-soft">{u}</span></div>
            </div>
          ))}
        </div>
        <div className="grid grid-cols-4 gap-3">
          {(['breakfast', 'lunch', 'snack', 'dinner'] as const).map(slot => (
            <div key={slot} className="rounded-xl border border-black/5 bg-paper p-3 text-xs">
              <div className="mb-1 font-semibold capitalize text-ink-main">{slot === 'snack' ? 'Evening snack' : slot} <span className="font-mono font-normal text-ink-soft">{plan.meals[slot]?.time}</span></div>
              <ul className="list-disc space-y-0.5 pl-4">{(plan.meals[slot]?.items ?? []).map(i => <li key={i}>{i}</li>)}</ul>
            </div>
          ))}
        </div>
        <div className="text-xs text-ink-main"><span className="font-semibold">Avoid:</span> {plan.foods_to_avoid.join(' · ')}</div>
        <div className="rounded-xl border border-black/5 bg-paper p-4 text-sm">
          <span className="font-semibold">Exercise:</span> {ex.allowed_activity} · {ex.duration_mins} min{ex.sessions_per_day && ex.sessions_per_day > 1 ? ` × ${ex.sessions_per_day}/day` : ''} · {ex.frequency} · <span className="font-semibold text-rose-dark">max HR {ex.max_hr_ceiling} bpm</span>
          <ul className="mt-2 list-disc space-y-0.5 pl-5 text-xs text-ink-muted">{ex.warnings.map(w => <li key={w}>{w}</li>)}</ul>
        </div>
        <div className="rounded-xl border border-amber-dark/15 bg-amber-light/40 p-4 text-sm leading-relaxed"><span className="font-semibold">Why the plan changed:</span> {plan.why_plan_changed_summary}</div>
      </section>
    </>
  );
}
