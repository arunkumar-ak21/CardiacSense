import { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { CheckCircle2, Loader2, Stethoscope, X } from 'lucide-react';
import { useVitals } from '../../context/VitalsContext';
import type { DietExercisePlan } from '../../lib/api';

const SLOTS: { key: string; label: string }[] = [
  { key: 'breakfast', label: 'Breakfast' },
  { key: 'lunch', label: 'Lunch' },
  { key: 'snack', label: 'Evening Snack' },
  { key: 'dinner', label: 'Dinner' },
];

const TARGETS: { key: keyof DietExercisePlan['daily_targets']; label: string; unit: string; step: string }[] = [
  { key: 'sodium_mg', label: 'Sodium', unit: 'mg', step: '50' },
  { key: 'protein_g', label: 'Protein', unit: 'g', step: '1' },
  { key: 'fluid_liters', label: 'Fluid', unit: 'L', step: '0.1' },
  { key: 'calories', label: 'Calories', unit: 'kcal', step: '50' },
  { key: 'potassium_mg', label: 'Potassium', unit: 'mg', step: '100' },
  { key: 'saturated_fat_g', label: 'Saturated fat', unit: 'g', step: '1' },
  { key: 'fiber_g', label: 'Fiber', unit: 'g', step: '1' },
];

const lines = (text: string) => text.split('\n').map(s => s.trim()).filter(Boolean);

/** Editable copy of the active visit's plan + "Approve & Publish to Patient Profile". */
export function StaffPlanEditor({ onDone }: { onDone?: () => void }) {
  const { activeVisit, activeProfile, approveActiveVisitPlan } = useVitals();
  const [plan, setPlan] = useState<DietExercisePlan | null>(null);
  const [notes, setNotes] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!activeVisit) return;
    setPlan(structuredClone(activeVisit.diet_plan));
    setNotes(activeVisit.staff_notes && !/pending/i.test(activeVisit.staff_notes) ? activeVisit.staff_notes : '');
    setDone(false);
    setError(null);
    // Reset only when a different visit is selected (not after our own approval updates it).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeVisit?.id]);

  if (!plan || !activeVisit) return null;

  const setTarget = (key: string, value: string) =>
    setPlan(p => p && ({ ...p, daily_targets: { ...p.daily_targets, [key]: value === '' ? undefined : Number(value) } }));
  const setExercise = (patch: Partial<DietExercisePlan['exercise_plan']>) =>
    setPlan(p => p && ({ ...p, exercise_plan: { ...p.exercise_plan, ...patch } }));
  const setMeal = (slot: string, text: string) =>
    setPlan(p => p && ({ ...p, meals: { ...p.meals, [slot]: { ...(p.meals[slot] ?? { time: '' }), items: lines(text) } } }));

  const approve = async () => {
    setSaving(true);
    setError(null);
    try {
      const cleanTargets = Object.fromEntries(Object.entries(plan.daily_targets).filter(([, v]) => typeof v === 'number' && v > 0));
      await approveActiveVisitPlan(
        notes.trim() || 'Verified and approved by hospital care team.',
        { ...plan, daily_targets: cleanTargets as DietExercisePlan['daily_targets'] },
      );
      setDone(true);
      onDone?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Approval failed');
    } finally {
      setSaving(false);
    }
  };

  const input = 'w-full rounded-lg border border-black/10 bg-paper px-2.5 py-1.5 text-sm text-ink-main outline-none focus:border-lavender-dark';
  return (
    <div className="flex flex-col gap-5">
      <div className="text-xs text-ink-muted">
        Editing the AI draft for <span className="font-semibold text-ink-main">{activeProfile?.name}</span> · visit {activeVisit.visit_date}
        {plan.clinical_rules_applied?.length ? <> · rules: <span className="font-mono">{plan.clinical_rules_applied.join(', ')}</span></> : null}
      </div>

      <div>
        <h4 className="mb-2 font-mono text-[10px] uppercase tracking-widest text-ink-soft">Daily targets</h4>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {TARGETS.map(t => (
            <label key={t.key as string} className="text-[11px] font-medium text-ink-muted">{t.label} ({t.unit})
              <input type="number" min={0} step={t.step} className={`${input} mt-1`}
                value={plan.daily_targets[t.key] ?? ''} onChange={e => setTarget(t.key as string, e.target.value)} />
            </label>
          ))}
        </div>
      </div>

      <div>
        <h4 className="mb-2 font-mono text-[10px] uppercase tracking-widest text-ink-soft">Exercise protocol</h4>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <label className="col-span-2 text-[11px] font-medium text-ink-muted">Allowed activity
            <input className={`${input} mt-1`} value={plan.exercise_plan.allowed_activity} onChange={e => setExercise({ allowed_activity: e.target.value })} />
          </label>
          <label className="text-[11px] font-medium text-ink-muted">Duration (min)
            <input type="number" min={1} className={`${input} mt-1`} value={plan.exercise_plan.duration_mins} onChange={e => setExercise({ duration_mins: Number(e.target.value) })} />
          </label>
          <label className="text-[11px] font-medium text-ink-muted">Max HR ceiling (bpm)
            <input type="number" min={40} className={`${input} mt-1`} value={plan.exercise_plan.max_hr_ceiling} onChange={e => setExercise({ max_hr_ceiling: Number(e.target.value) })} />
          </label>
          <label className="col-span-2 text-[11px] font-medium text-ink-muted">Frequency
            <input className={`${input} mt-1`} value={plan.exercise_plan.frequency ?? ''} onChange={e => setExercise({ frequency: e.target.value })} />
          </label>
          <label className="col-span-2 text-[11px] font-medium text-ink-muted">Stop warnings (one per line)
            <textarea rows={3} className={`${input} mt-1 resize-none`} value={plan.exercise_plan.warnings.join('\n')} onChange={e => setExercise({ warnings: lines(e.target.value) })} />
          </label>
        </div>
      </div>

      <div>
        <h4 className="mb-2 font-mono text-[10px] uppercase tracking-widest text-ink-soft">Meals (one item per line)</h4>
        <div className="grid gap-3 sm:grid-cols-2">
          {SLOTS.map(s => (
            <label key={s.key} className="text-[11px] font-medium text-ink-muted">{s.label}
              <textarea rows={3} className={`${input} mt-1 resize-none`} value={(plan.meals[s.key]?.items ?? []).join('\n')} onChange={e => setMeal(s.key, e.target.value)} />
            </label>
          ))}
          <label className="text-[11px] font-medium text-ink-muted sm:col-span-2">Foods to strictly avoid (one per line)
            <textarea rows={3} className={`${input} mt-1 resize-none`} value={plan.foods_to_avoid.join('\n')} onChange={e => setPlan({ ...plan, foods_to_avoid: lines(e.target.value) })} />
          </label>
        </div>
      </div>

      <label className="text-[11px] font-medium text-ink-muted">Staff notes
        <textarea rows={2} maxLength={2000} className={`${input} mt-1 resize-none`} value={notes}
          placeholder="Verified against renal-cardiac protocol. Approved for patient."
          onChange={e => setNotes(e.target.value)} />
      </label>

      {error && <div className="rounded-lg bg-rose-light/60 px-3 py-2 text-xs text-rose-dark">{error}</div>}
      {done && <div className="flex items-center gap-2 rounded-lg bg-sage-light px-3 py-2 text-xs text-sage-dark"><CheckCircle2 className="h-4 w-4" /> Approved and published to the patient profile.</div>}

      <button onClick={approve} disabled={saving}
        className="flex items-center justify-center gap-2 rounded-xl bg-lavender-dark py-3 text-sm font-semibold text-white shadow-sm transition hover:opacity-90 disabled:opacity-60">
        {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Stethoscope className="h-4 w-4" />}
        Approve & Publish to Patient Profile
      </button>
    </div>
  );
}

export default function StaffApprovalModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          className="fixed inset-0 z-[90] flex items-center justify-center bg-ink-main/40 p-4 backdrop-blur-sm" onClick={onClose}>
          <motion.div initial={{ y: 24, scale: 0.98 }} animate={{ y: 0, scale: 1 }} exit={{ y: 24, scale: 0.98 }}
            onClick={e => e.stopPropagation()}
            className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-2xl bg-paper p-6 shadow-float">
            <div className="mb-4 flex items-center gap-2">
              <h3 className="font-serif text-xl text-ink-main">🏥 Review, Edit & Approve Diet/Exercise Plan</h3>
              <button onClick={onClose} className="ml-auto rounded-lg p-1 text-ink-soft hover:bg-cream"><X className="h-4 w-4" /></button>
            </div>
            <StaffPlanEditor onDone={() => setTimeout(onClose, 900)} />
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
