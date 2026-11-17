import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Ban, Clock, Dumbbell, HeartPulse, Lightbulb, Salad, ShieldAlert, Sparkles } from 'lucide-react';
import { useVitals } from '../../context/VitalsContext';
import LongitudinalShell, { Card, RiskDeltaBadge, VisitSelector } from './LongitudinalShell';
import StaffApprovalModal, { StaffPlanEditor } from '../dashboard/StaffApprovalModal';
import { dietLabel } from '../PatientProfileSwitcher';
import { streamNarrative, type DietExercisePlan } from '../../lib/api';
import { buildLongitudinalPayload } from '../../lib/longitudinal';

const SLOTS = [
  { key: 'breakfast', label: 'Breakfast' },
  { key: 'lunch', label: 'Lunch' },
  { key: 'snack', label: 'Evening Snack' },
  { key: 'dinner', label: 'Dinner' },
];

const TARGET_ROWS: { key: string; label: string; unit: string }[] = [
  { key: 'sodium_mg', label: 'Sodium', unit: 'mg' },
  { key: 'protein_g', label: 'Protein', unit: 'g' },
  { key: 'fluid_liters', label: 'Fluid', unit: 'L' },
  { key: 'calories', label: 'Calories', unit: 'kcal' },
  { key: 'potassium_mg', label: 'Potassium', unit: 'mg' },
  { key: 'saturated_fat_g', label: 'Sat. fat', unit: 'g' },
  { key: 'fiber_g', label: 'Fiber', unit: 'g' },
];

function PlanColumn({ plan, heading, tone, followNote }: { plan: DietExercisePlan | undefined; heading: string; tone: 'old' | 'new'; followNote?: string | null }) {
  if (!plan) {
    return (
      <div className="rounded-xl border border-dashed border-black/10 p-4 text-sm text-ink-muted">
        No earlier plan — this is the baseline plan.
      </div>
    );
  }
  const ex = plan.exercise_plan;
  return (
    <div className={`rounded-xl border p-4 ${tone === 'new' ? 'border-lavender-dark/20 bg-lavender-light/25' : 'border-black/5 bg-ivory'}`}>
      <div className="mb-1 font-mono text-[10px] uppercase tracking-widest text-ink-soft">{heading}</div>
      <div className="mb-3 text-sm font-semibold text-ink-main">{plan.plan_title ?? 'Diet & exercise plan'}</div>
      <div className="grid grid-cols-3 gap-2">
        {TARGET_ROWS.slice(0, 3).map(t => (
          <div key={t.key} className="rounded-lg border border-black/5 bg-paper px-2 py-1.5">
            <div className="text-[10px] text-ink-soft">{t.label}</div>
            <div className="font-mono text-sm font-semibold text-ink-main">{plan.daily_targets[t.key] ?? '—'} {t.unit}</div>
          </div>
        ))}
      </div>
      <div className="mt-3 text-xs text-ink-main">
        <span className="font-semibold">Exercise:</span> {ex.allowed_activity} · {ex.duration_mins} min
        {ex.sessions_per_day && ex.sessions_per_day > 1 ? ` × ${ex.sessions_per_day}/day` : ''} · max HR {ex.max_hr_ceiling} bpm
      </div>
      {followNote !== undefined && (
        <div className="mt-2 text-xs text-ink-muted"><span className="font-semibold">Did the patient follow it?</span> {followNote || 'Not recorded'}</div>
      )}
    </div>
  );
}

export default function DietCompanionView() {
  const { activeProfile, activeVisit, previousVisit, isStaffMode } = useVitals();
  const [modalOpen, setModalOpen] = useState(false);
  const [explanation, setExplanation] = useState('');
  const [explaining, setExplaining] = useState(false);

  // Longitudinal report explanation (LLM when configured, otherwise the backend's rule-based fallback).
  useEffect(() => {
    const payload = buildLongitudinalPayload(activeProfile, activeVisit, previousVisit, isStaffMode);
    if (!payload) return;
    let cancelled = false;
    setExplanation('');
    setExplaining(true);
    (async () => {
      try {
        let text = '';
        for await (const t of streamNarrative(payload)) {
          if (cancelled) return;
          text += t;
          setExplanation(text);
        }
      } catch {
        if (!cancelled) setExplanation('');
      } finally {
        if (!cancelled) setExplaining(false);
      }
    })();
    return () => { cancelled = true; };
  }, [activeProfile, activeVisit, previousVisit, isStaffMode]);

  if (!activeVisit || !activeProfile) return null;
  const plan = activeVisit.diet_plan;
  const prevPlan = previousVisit?.diet_plan;
  const ex = plan.exercise_plan;
  const approved = activeVisit.approved_by_staff;

  return (
    <LongitudinalShell
      title="Report Explanation & Adaptive Diet / Exercise Companion"
      subtitle={<>{activeProfile.name} · {dietLabel(plan.diet_preference ?? activeProfile.diet_preference)} · visit {activeVisit.visit_date}</>}
      actions={<VisitSelector />}
    >
      {/* ── Verification banner ─────────────────────────────────────────── */}
      <div className={`flex flex-wrap items-center gap-4 rounded-2xl border p-4 ${approved ? 'border-sage-dark/20 bg-sage-light/60' : 'border-amber-dark/20 bg-amber-light/60'}`}>
        <div className="min-w-0 flex-1">
          <div className={`text-sm font-semibold ${approved ? 'text-sage-dark' : 'text-amber-dark'}`}>
            {approved ? '✅ Verified & Approved by Hospital Care Team / Dietitian' : '⏳ AI Draft Plan — Pending Hospital Dietitian / Doctor Review'}
          </div>
          <div className="text-xs text-ink-muted">
            {approved
              ? (activeVisit.staff_notes || 'Approved.') + (plan.staff_review?.reviewed_at ? ` · ${new Date(plan.staff_review.reviewed_at).toLocaleString()}` : '')
              : 'Follow your care team’s advice until this plan is signed off.'}
            {plan.generated_by && <span className="ml-1 font-mono text-[10px] text-ink-soft">({plan.generated_by})</span>}
          </div>
        </div>
        {isStaffMode && (
          <button onClick={() => setModalOpen(true)} className="rounded-xl bg-lavender-dark px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:opacity-90">
            🏥 Review, Edit & Approve Diet/Exercise Plan
          </button>
        )}
      </div>

      {/* ── Report explanation ─────────────────────────────────────────── */}
      <Card title={isStaffMode ? 'Clinical longitudinal summary' : 'Your report, explained simply'} icon={<Sparkles className="h-4 w-4 text-lavender-dark" />}
        right={<RiskDeltaBadge delta={activeVisit.delta_summary} />}>
        <p className="whitespace-pre-line text-sm leading-relaxed text-ink-main">
          {explanation || (explaining ? 'Preparing your explanation…' : activeVisit.delta_summary?.what_went_wrong_summary ?? '')}
        </p>
      </Card>

      {/* ── Previous vs new plan ───────────────────────────────────────── */}
      <Card title="Previous Plan vs. New Adapted Plan" icon={<Clock className="h-4 w-4 text-ink-muted" />}>
        <div className="grid gap-4 md:grid-cols-2">
          <PlanColumn plan={prevPlan} heading={`What was suggested previously${previousVisit ? ` (${previousVisit.visit_date})` : ''}`} tone="old"
            followNote={previousVisit ? activeVisit.adherence_notes : undefined} />
          <PlanColumn plan={plan} heading={`What you must do now (${activeVisit.visit_date})`} tone="new" />
        </div>
        <div className="mt-4 rounded-xl border border-amber-dark/20 bg-amber-light/50 p-4">
          <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-amber-dark"><Lightbulb className="h-4 w-4" /> Why your plan changed</div>
          <p className="text-sm leading-relaxed text-ink-main">{plan.why_plan_changed_summary}</p>
        </div>
      </Card>

      {/* ── Nutrition ──────────────────────────────────────────────────── */}
      <Card title="Daily Nutrition & Meal Schedule" icon={<Salad className="h-4 w-4 text-sage-dark" />}>
        <div className="mb-4 grid grid-cols-3 gap-2 sm:grid-cols-7">
          {TARGET_ROWS.map(t => {
            const now = plan.daily_targets[t.key];
            const before = prevPlan?.daily_targets[t.key];
            const changed = typeof before === 'number' && typeof now === 'number' && before !== now;
            return (
              <div key={t.key} className={`rounded-xl border px-3 py-2 ${changed ? 'border-lavender-dark/25 bg-lavender-light/30' : 'border-black/5 bg-ivory'}`}>
                <div className="text-[10px] uppercase tracking-wider text-ink-soft">{t.label}</div>
                <div className="font-serif text-xl text-ink-main">{now ?? '—'}<span className="text-xs text-ink-soft"> {t.unit}</span></div>
                {changed && <div className="font-mono text-[10px] text-lavender-dark">was {before}</div>}
              </div>
            );
          })}
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {SLOTS.map((s, i) => {
            const meal = plan.meals[s.key];
            return (
              <motion.div key={s.key} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }}
                className="rounded-xl border border-black/5 bg-ivory p-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold text-ink-main">{s.label}</span>
                  <span className="font-mono text-[10px] text-ink-soft">{meal?.time}</span>
                </div>
                <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-ink-main">
                  {(meal?.items ?? []).map(item => <li key={item}>{item}</li>)}
                </ul>
                {meal?.notes && <p className="mt-2 text-[11px] italic text-ink-muted">{meal.notes}</p>}
              </motion.div>
            );
          })}
        </div>
        <div className="mt-4 rounded-xl border border-rose-dark/15 bg-rose-light/25 p-4">
          <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-rose-dark"><Ban className="h-4 w-4" /> Foods to strictly avoid</div>
          <div className="flex flex-wrap gap-1.5">
            {plan.foods_to_avoid.map(f => <span key={f} className="rounded-full border border-rose-dark/10 bg-paper px-2.5 py-1 text-[11px] text-ink-main">{f}</span>)}
          </div>
        </div>
      </Card>

      {/* ── Exercise ───────────────────────────────────────────────────── */}
      <Card title="Safe Cardiac Exercise Protocol" icon={<Dumbbell className="h-4 w-4 text-lavender-dark" />}>
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-xl border border-black/5 bg-ivory p-4">
            <div className="text-[10px] uppercase tracking-wider text-ink-soft">Allowed activity</div>
            <div className="mt-1 text-sm font-semibold text-ink-main">{ex.allowed_activity}</div>
            {ex.intensity && <div className="text-xs text-ink-muted">Intensity: {ex.intensity.replace('_', ' ')}</div>}
          </div>
          <div className="rounded-xl border border-black/5 bg-ivory p-4">
            <div className="text-[10px] uppercase tracking-wider text-ink-soft">Duration & frequency</div>
            <div className="mt-1 text-sm font-semibold text-ink-main">
              {ex.duration_mins} min{ex.sessions_per_day && ex.sessions_per_day > 1 ? ` × ${ex.sessions_per_day} sessions/day` : ''}
            </div>
            <div className="text-xs text-ink-muted">{ex.frequency}</div>
          </div>
          <div className="rounded-xl border border-rose-dark/20 bg-rose-light/40 p-4">
            <div className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-rose-dark"><HeartPulse className="h-3 w-3" /> Strict max heart rate</div>
            <div className="mt-1 font-serif text-3xl text-rose-dark">{ex.max_hr_ceiling}<span className="text-sm"> bpm</span></div>
          </div>
        </div>
        <div className="mt-3 rounded-xl border border-black/5 bg-cream p-4">
          <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-ink-main"><ShieldAlert className="h-4 w-4 text-rose-dark" /> Red-flag stop warnings</div>
          <ul className="list-disc space-y-1 pl-5 text-xs text-ink-main">
            {ex.warnings.map(w => <li key={w}>{w}</li>)}
          </ul>
        </div>
      </Card>

      {/* ── Inline staff editor ────────────────────────────────────────── */}
      {isStaffMode && (
        <Card title="🏥 Staff plan editor (inline)" className="border-lavender-dark/30">
          <StaffPlanEditor />
        </Card>
      )}

      <StaffApprovalModal open={modalOpen} onClose={() => setModalOpen(false)} />
    </LongitudinalShell>
  );
}
