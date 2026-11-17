/* eslint-disable react-refresh/only-export-components */
import { useEffect, useRef, useState, type FormEvent } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Check, ChevronDown, Plus, UserRound, X } from 'lucide-react';
import { useVitals } from '../context/VitalsContext';
import type { DietPreference, NewPatientInput } from '../lib/api';

export const DIET_OPTIONS: { value: DietPreference; label: string }[] = [
  { value: 'vegetarian', label: 'Vegetarian' },
  { value: 'non_vegetarian', label: 'Non-Vegetarian' },
  { value: 'vegan', label: 'Vegan' },
  { value: 'diabetic', label: 'Diabetic-Friendly' },
  { value: 'dash', label: 'Low-Sodium DASH' },
];

export function dietLabel(value?: string | null): string {
  return DIET_OPTIONS.find(d => d.value === value)?.label ?? (value ?? '—');
}

export function tierClasses(tier?: string | null): string {
  switch (tier) {
    case 'Critical': return 'bg-rose-dark text-white';
    case 'High': return 'bg-rose-light text-rose-dark border border-rose-main';
    case 'Moderate': return 'bg-amber-light text-amber-dark border border-amber-main';
    case 'Low': return 'bg-sage-light text-sage-dark border border-sage-main';
    default: return 'bg-sand-light text-ink-muted';
  }
}

/** `🧑‍💼 Patient Companion` | `🏥 Hospital Staff / Dietitian` pill. */
export function RoleModeSwitch({ compact = false }: { compact?: boolean }) {
  const { isStaffMode, setIsStaffMode } = useVitals();
  const opts = [
    { staff: false, icon: '🧑‍💼', label: 'Patient Companion' },
    { staff: true, icon: '🏥', label: compact ? 'Staff / Dietitian' : 'Hospital Staff / Dietitian' },
  ];
  return (
    <div className="relative flex rounded-full border border-black/5 bg-cream p-0.5 text-[11px] font-semibold" role="radiogroup" aria-label="View mode">
      {opts.map(o => {
        const active = isStaffMode === o.staff;
        return (
          <button
            key={o.label}
            role="radio"
            aria-checked={active}
            onClick={() => setIsStaffMode(o.staff)}
            className={`relative z-10 flex items-center gap-1 whitespace-nowrap rounded-full px-3 py-1.5 transition-colors ${active ? (o.staff ? 'text-white' : 'text-ink-main') : 'text-ink-muted hover:text-ink-main'}`}
          >
            {active && (
              <motion.span
                layoutId={`role-pill-${compact ? 'c' : 'f'}`}
                className={`absolute inset-0 -z-10 rounded-full ${o.staff ? 'bg-lavender-dark' : 'bg-paper shadow-sm'}`}
                transition={{ type: 'spring', stiffness: 420, damping: 34 }}
              />
            )}
            <span aria-hidden>{o.icon}</span> {o.label}
          </button>
        );
      })}
    </div>
  );
}

/** Modal form: Name, MRN, Age, Gender, Diet Preference. */
export function NewPatientModal({ open, onClose, onCreated }: { open: boolean; onClose: () => void; onCreated?: () => void }) {
  const { createNewProfile } = useVitals();
  const [form, setForm] = useState<NewPatientInput>({ name: '', mrn: '', age: 50, gender: 'M', diet_preference: 'vegetarian' });
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (open) {
      setError(null);
      setForm(f => ({ ...f, mrn: f.mrn || `MRN-${new Date().getFullYear()}-${Math.floor(100 + Math.random() * 900)}` }));
    }
  }, [open]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await createNewProfile({ ...form, name: form.name.trim(), mrn: form.mrn.trim() });
      setForm({ name: '', mrn: '', age: 50, gender: 'M', diet_preference: 'vegetarian' });
      onClose();
      onCreated?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create profile');
    } finally {
      setSaving(false);
    }
  };

  const input = 'w-full rounded-lg border border-black/10 bg-ivory px-3 py-2 text-sm text-ink-main outline-none focus:border-lavender-dark';
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          className="fixed inset-0 z-[90] flex items-center justify-center bg-ink-main/40 p-4 backdrop-blur-sm"
          onClick={onClose}
        >
          <motion.form
            initial={{ y: 20, scale: 0.98 }} animate={{ y: 0, scale: 1 }} exit={{ y: 20, scale: 0.98 }}
            onSubmit={submit}
            onClick={e => e.stopPropagation()}
            className="w-full max-w-md rounded-2xl bg-paper p-6 shadow-float"
          >
            <div className="mb-4 flex items-center justify-between">
              <h3 className="font-serif text-xl text-ink-main">New Patient Profile</h3>
              <button type="button" onClick={onClose} className="rounded-lg p-1 text-ink-soft hover:bg-cream"><X className="h-4 w-4" /></button>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <label className="col-span-2 text-xs font-medium text-ink-muted">Full name
                <input required maxLength={120} className={input} value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} placeholder="e.g. Anita Sharma" />
              </label>
              <label className="text-xs font-medium text-ink-muted">MRN
                <input required maxLength={50} className={input} value={form.mrn} onChange={e => setForm({ ...form, mrn: e.target.value })} />
              </label>
              <label className="text-xs font-medium text-ink-muted">Age
                <input required type="number" min={0} max={120} className={input} value={form.age} onChange={e => setForm({ ...form, age: Number(e.target.value) })} />
              </label>
              <label className="text-xs font-medium text-ink-muted">Gender
                <select className={input} value={form.gender} onChange={e => setForm({ ...form, gender: e.target.value as NewPatientInput['gender'] })}>
                  <option value="M">Male</option><option value="F">Female</option><option value="Other">Other</option>
                </select>
              </label>
              <label className="text-xs font-medium text-ink-muted">Diet preference
                <select className={input} value={form.diet_preference} onChange={e => setForm({ ...form, diet_preference: e.target.value as DietPreference })}>
                  {DIET_OPTIONS.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
                </select>
              </label>
            </div>
            {error && <p className="mt-3 rounded-lg bg-rose-light/50 px-3 py-2 text-xs text-rose-dark">{error}</p>}
            <button disabled={saving} className="mt-5 w-full rounded-xl bg-ink-main py-2.5 text-sm font-medium text-paper transition hover:bg-ink-main/90 disabled:opacity-60">
              {saving ? 'Creating…' : 'Create profile'}
            </button>
          </motion.form>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/** Dropdown: "Ramesh Kumar • 58M • 2 Visits" with one-click switching + "New Patient Profile". */
export default function PatientProfileSwitcher({ align = 'left', fullWidth = false }: { align?: 'left' | 'right'; fullWidth?: boolean }) {
  const { profiles, activeProfile, visits, selectProfile, loadingProfiles } = useVitals();
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: globalThis.MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  const visitCount = activeProfile ? (visits.length || activeProfile.visit_count || 0) : 0;
  const summary = activeProfile
    ? `${activeProfile.name} • ${activeProfile.age}${activeProfile.gender === 'Other' ? '' : activeProfile.gender} • ${visitCount} Visit${visitCount === 1 ? '' : 's'}`
    : loadingProfiles ? 'Loading patients…' : 'Select patient';

  return (
    <div ref={ref} className={`relative ${fullWidth ? 'w-full' : ''}`}>
      <button
        onClick={() => setOpen(o => !o)}
        className={`flex items-center gap-2 rounded-xl border border-black/5 bg-paper px-3 py-2 text-xs font-semibold text-ink-main shadow-sm transition hover:border-black/10 ${fullWidth ? 'w-full' : ''}`}
      >
        <UserRound className="h-4 w-4 text-lavender-dark" />
        <span className="truncate">{summary}</span>
        <ChevronDown className={`ml-auto h-3.5 w-3.5 text-ink-soft transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
            className={`absolute z-[70] mt-2 w-80 rounded-2xl border border-black/5 bg-paper p-2 shadow-float ${align === 'right' ? 'right-0' : 'left-0'}`}
          >
            <div className="px-2 pb-1 pt-1 font-mono text-[10px] uppercase tracking-widest text-ink-soft">Patient profiles</div>
            <div className="max-h-72 overflow-y-auto">
              {profiles.map(p => (
                <button
                  key={p.id}
                  onClick={() => { setOpen(false); void selectProfile(p.id); }}
                  className={`flex w-full items-center gap-3 rounded-xl px-2 py-2 text-left transition hover:bg-cream ${p.id === activeProfile?.id ? 'bg-lavender-light/40' : ''}`}
                >
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-lavender-main to-lavender-dark text-[11px] font-bold text-white">
                    {p.name.split(' ').map(s => s[0]).slice(0, 2).join('')}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-xs font-semibold text-ink-main">{p.name}</div>
                    <div className="text-[10px] text-ink-soft">{p.age}{p.gender} · {dietLabel(p.diet_preference)} · {p.visit_count ?? 0} visits</div>
                  </div>
                  {p.latest_risk_score != null && (
                    <span className={`rounded-full px-1.5 py-0.5 text-[9px] font-bold ${tierClasses(p.latest_severity_tier)}`}>{p.latest_risk_score.toFixed(1)}%</span>
                  )}
                  {p.id === activeProfile?.id && <Check className="h-3.5 w-3.5 text-lavender-dark" />}
                </button>
              ))}
            </div>
            <button
              onClick={() => { setOpen(false); setCreating(true); }}
              className="mt-1 flex w-full items-center gap-2 rounded-xl border border-dashed border-black/10 px-3 py-2 text-xs font-semibold text-lavender-dark hover:bg-lavender-light/30"
            >
              <Plus className="h-3.5 w-3.5" /> New Patient Profile
            </button>
          </motion.div>
        )}
      </AnimatePresence>
      <NewPatientModal open={creating} onClose={() => setCreating(false)} />
    </div>
  );
}
