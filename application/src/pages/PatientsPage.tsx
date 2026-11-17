import { useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { FilePlus2, LineChart, Plus, UserRound } from 'lucide-react';
import { useVitals } from '../context/VitalsContext';
import LongitudinalShell, { TierBadge } from '../components/longitudinal/LongitudinalShell';
import { NewPatientModal, dietLabel } from '../components/PatientProfileSwitcher';

/** Step 1 — choose an existing patient profile or create a new one. */
export default function PatientsPage() {
  const navigate = useNavigate();
  const { profiles, activeProfile, selectProfile, loadingProfiles } = useVitals();
  const [creating, setCreating] = useState(false);

  const open = async (id: string, to: string) => {
    await selectProfile(id);
    navigate(to);
  };

  return (
    <LongitudinalShell
      title="Choose a patient profile"
      subtitle="Pick the patient whose new hospital report you want to add. All of their past visits load automatically."
      actions={
        <button onClick={() => setCreating(true)} className="flex items-center gap-2 rounded-xl bg-ink-main px-4 py-2.5 text-sm font-medium text-paper hover:bg-ink-main/90">
          <Plus className="h-4 w-4" /> New Patient Profile
        </button>
      }
    >
      {loadingProfiles && <div className="text-sm text-ink-muted">Loading patients…</div>}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {profiles.map((p, i) => {
          const selected = p.id === activeProfile?.id;
          const visits = p.visit_count ?? 0;
          return (
            <motion.div
              key={p.id}
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}
              className={`flex flex-col gap-4 rounded-2xl border bg-paper p-5 shadow-soft ${selected ? 'border-ink-main/30 ring-2 ring-lavender-dark/20' : 'border-black/5'}`}
            >
              <div className="flex items-start gap-3">
                <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-lavender-main to-lavender-dark text-base font-bold text-white">
                  {p.name.split(' ').map(s => s[0]).slice(0, 2).join('')}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-base font-semibold text-ink-main">{p.name}</div>
                  <div className="text-xs text-ink-muted">{p.age} yrs · {p.gender === 'M' ? 'Male' : p.gender === 'F' ? 'Female' : p.gender} · {dietLabel(p.diet_preference)}</div>
                  <div className="font-mono text-[10px] text-ink-soft">MRN {p.mrn}</div>
                </div>
                {selected && <span className="rounded-full bg-lavender-light px-2 py-0.5 text-[10px] font-semibold text-lavender-dark">Selected</span>}
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs text-ink-muted">
                <span>{visits} hospital visit{visits === 1 ? '' : 's'}</span>
                {p.latest_visit_date && <span>· last {p.latest_visit_date}</span>}
                {p.latest_risk_score != null && (
                  <>
                    <span className="font-semibold text-ink-main">{p.latest_risk_score.toFixed(1)}%</span>
                    <TierBadge tier={p.latest_severity_tier} className="!text-[10px]" />
                  </>
                )}
                {visits > 0 && (p.latest_plan_approved
                  ? <span className="rounded-full bg-sage-light px-2 py-0.5 text-[10px] font-semibold text-sage-dark">✅ Hospital Verified plan</span>
                  : <span className="rounded-full bg-amber-light px-2 py-0.5 text-[10px] font-semibold text-amber-dark">⏳ Plan pending review</span>)}
              </div>
              <div className="mt-auto flex gap-2">
                <button onClick={() => open(p.id, '/vitals')} className="flex flex-1 items-center justify-center gap-1.5 rounded-xl bg-ink-main px-3 py-2.5 text-xs font-medium text-paper hover:bg-ink-main/90">
                  <FilePlus2 className="h-4 w-4" /> Add new hospital report
                </button>
                {visits > 0 && (
                  <button onClick={() => open(p.id, '/predictions')} className="flex items-center justify-center gap-1.5 rounded-xl border border-black/10 bg-paper px-3 py-2.5 text-xs font-medium text-ink-main hover:bg-cream">
                    <LineChart className="h-4 w-4" /> Latest results
                  </button>
                )}
              </div>
            </motion.div>
          );
        })}
        <button onClick={() => setCreating(true)}
          className="flex min-h-[180px] flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-black/10 bg-ivory p-5 text-sm font-medium text-ink-muted transition hover:border-lavender-dark/40 hover:text-lavender-dark">
          <UserRound className="h-6 w-6" /> + New Patient Profile
          <span className="text-xs font-normal">Name, age, gender and diet preference</span>
        </button>
      </div>
      <NewPatientModal open={creating} onClose={() => setCreating(false)} onCreated={() => navigate('/vitals')} />
    </LongitudinalShell>
  );
}
