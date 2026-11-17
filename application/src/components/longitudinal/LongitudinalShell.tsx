import type { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowDownRight, ArrowUpRight, Minus } from 'lucide-react';
import { pageVariants } from '../../lib/animations';
import { useVitals } from '../../context/VitalsContext';
import PatientProfileSwitcher, { RoleModeSwitch, tierClasses } from '../PatientProfileSwitcher';
import { fmt, visitNumber } from '../../lib/longitudinal';
import { usingLongitudinalFallback, type DeltaSummary } from '../../lib/api';
import JourneyStepper, { JourneyNext } from './JourneyStepper';

/** Green "Hospital Verified" badge once staff approved the active visit's plan. */
export function VerifiedBadge() {
  const { activeVisit } = useVitals();
  if (!activeVisit) return null;
  return activeVisit.approved_by_staff ? (
    <span className="rounded-full border border-sage-dark/20 bg-sage-light px-2.5 py-1 text-[11px] font-semibold text-sage-dark">? Hospital Verified</span>
  ) : (
    <span className="rounded-full border border-amber-dark/20 bg-amber-light px-2.5 py-1 text-[11px] font-semibold text-amber-dark">? Plan awaiting staff review</span>
  );
}

const EXTRA_NAV = [
  { label: 'Dashboard', path: '/dashboard' },
  { label: 'PDF Report', path: '/reports' },
];

export function TierBadge({ tier, className = '' }: { tier?: string | null; className?: string }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-bold ${tierClasses(tier)} ${className}`}>
      {tier ?? '—'} Risk
    </span>
  );
}

/** "▲ +40.4% worsened since 2026-08-10 (34.2% → 74.6%)" */
export function RiskDeltaBadge({ delta, large = false }: { delta: DeltaSummary | null | undefined; large?: boolean }) {
  if (!delta || delta.is_baseline || delta.risk_delta == null) {
    return (
      <span className={`inline-flex items-center gap-1.5 rounded-full border border-black/5 bg-sand-light px-3 py-1 font-medium text-ink-muted ${large ? 'text-sm' : 'text-xs'}`}>
        <Minus className="h-3.5 w-3.5" /> Baseline visit — nothing earlier to compare
      </span>
    );
  }
  const d = delta.risk_delta;
  const worse = d >= 0.05, better = d <= -0.05;
  const cls = worse ? 'bg-rose-light/70 text-rose-dark border-rose-dark/20' : better ? 'bg-sage-light text-sage-dark border-sage-dark/20' : 'bg-sand-light text-ink-muted border-black/5';
  const Icon = worse ? ArrowUpRight : better ? ArrowDownRight : Minus;
  return (
    <span className={`inline-flex flex-wrap items-center gap-1.5 rounded-full border px-3 py-1 font-semibold ${cls} ${large ? 'text-sm' : 'text-xs'}`}>
      <Icon className="h-4 w-4" />
      {worse ? '▲' : better ? '▼' : '•'} {d > 0 ? '+' : ''}{d.toFixed(1)}% {worse ? 'worsened' : better ? 'improved' : 'stable'} since {delta.previous_visit_date}
      <span className="font-normal opacity-80">({fmt(delta.previous_risk_score)}% → {fmt(delta.current_risk_score)}%)</span>
    </span>
  );
}

export function VisitSelector() {
  const { visits, activeVisit, selectVisit } = useVitals();
  if (!visits.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {[...visits].reverse().map(v => {
        const active = v.id === activeVisit?.id;
        return (
          <button
            key={v.id}
            onClick={() => selectVisit(v.id)}
            className={`rounded-lg border px-2.5 py-1 text-[11px] font-medium transition ${active ? 'border-ink-main bg-ink-main text-paper' : 'border-black/5 bg-paper text-ink-muted hover:text-ink-main'}`}
          >
            Visit #{visitNumber(visits, v.id)} · {v.visit_date}
          </button>
        );
      })}
    </div>
  );
}

export function EmptyVisits() {
  const navigate = useNavigate();
  const { activeProfile, loadingProfiles, loadingVisits } = useVitals();
  if (loadingProfiles || loadingVisits) {
    return <div className="rounded-2xl border border-black/5 bg-paper p-10 text-center text-sm text-ink-muted">Loading patient history…</div>;
  }
  return (
    <div className="rounded-2xl border border-black/5 bg-paper p-10 text-center">
      <p className="text-sm text-ink-muted">
        {activeProfile ? `${activeProfile.name} has no recorded hospital visits yet.` : 'Select a patient profile to begin.'}
      </p>
      <button onClick={() => navigate('/vitals')} className="mt-4 rounded-xl bg-ink-main px-5 py-2.5 text-sm font-medium text-paper hover:bg-ink-main/90">
        + Record a hospital visit
      </button>
    </div>
  );
}

export function Card({ title, icon, children, className = '', right }: { title?: ReactNode; icon?: ReactNode; children: ReactNode; className?: string; right?: ReactNode }) {
  return (
    <section className={`rounded-2xl border border-black/5 bg-paper p-5 shadow-soft ${className}`}>
      {(title || right) && (
        <div className="mb-4 flex items-center gap-2">
          {icon}
          {title && <h2 className="font-mono text-[11px] uppercase tracking-widest text-ink-muted">{title}</h2>}
          {right && <div className="ml-auto">{right}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export default function LongitudinalShell({ title, subtitle, children, actions }: { title: string; subtitle?: ReactNode; children: ReactNode; actions?: ReactNode }) {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const { isStaffMode } = useVitals();
  return (
    <motion.div variants={pageVariants} initial="initial" animate="enter" exit="exit" className="min-h-screen w-full bg-ivory font-sans text-ink-main antialiased">
      <header className={`sticky top-0 z-40 border-b bg-cream/90 backdrop-blur-md ${isStaffMode ? 'border-lavender-dark/30' : 'border-black/5'}`}>
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-4 px-6 py-3">
          <button onClick={() => navigate('/dashboard')} className="flex items-center gap-2">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
            <span className="font-serif text-lg font-medium tracking-tight">CardiacSense</span>
          </button>
          <div className="hidden flex-1 lg:block"><JourneyStepper /></div>
          <div className="flex items-center gap-1">
            {EXTRA_NAV.map(n => (
              <button key={n.path} onClick={() => navigate(n.path)}
                className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition ${pathname === n.path ? 'bg-paper text-ink-main shadow-sm' : 'text-ink-muted hover:text-ink-main'}`}>
                {n.label}
              </button>
            ))}
          </div>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <VerifiedBadge />
            <PatientProfileSwitcher align="right" />
            <RoleModeSwitch compact />
          </div>
        </div>
        {isStaffMode && (
          <div className="bg-lavender-dark py-1 text-center font-mono text-[10px] uppercase tracking-widest text-white">
            🏥 Hospital Staff / Dietitian View — clinical detail & plan approval enabled
          </div>
        )}
      </header>
      <main className="mx-auto flex max-w-[1400px] flex-col gap-6 px-6 py-6">
        <div className="flex flex-wrap items-end gap-4">
          <div className="mr-auto">
            <h1 className="font-serif text-3xl tracking-tight text-ink-main">{title}</h1>
            {subtitle && <div className="mt-1 text-sm text-ink-muted">{subtitle}</div>}
          </div>
          {actions}
        </div>
        {usingLongitudinalFallback && (
          <div className="rounded-xl border border-amber-dark/20 bg-amber-light/60 px-4 py-2 text-xs text-amber-dark">
            Backend unreachable — showing the offline demo copy of the seeded patients. New visits and approvals need the backend on port 8000.
          </div>
        )}
        {children}
        <JourneyNext />
      </main>
    </motion.div>
  );
}
