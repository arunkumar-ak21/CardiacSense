/* eslint-disable react-refresh/only-export-components */
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';

export const JOURNEY = [
  { path: '/patients', label: 'Choose patient', next: 'Choose a patient' },
  { path: '/vitals', label: 'New hospital report', next: 'Upload the new hospital report' },
  { path: '/predictions', label: 'Severity & ECG', next: 'See severity & ECG explanation' },
  { path: '/history', label: 'Before vs. now', next: 'See what changed since last visit' },
  { path: '/insights', label: 'Diet & exercise plan', next: 'See the updated diet & exercise plan' },
  { path: '/chat', label: 'Ask AI companion', next: 'Ask the AI companion' },
] as const;

/** 1 → 6 step bar: Choose patient → Report → Severity → Before vs now → Plan → Chat. */
export default function JourneyStepper({ compact = false }: { compact?: boolean }) {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const current = JOURNEY.findIndex(s => s.path === pathname);
  return (
    <ol className="flex flex-wrap items-center gap-1">
      {JOURNEY.map((s, i) => {
        const active = i === current;
        const done = current > i;
        return (
          <li key={s.path} className="flex items-center gap-1">
            <button
              onClick={() => navigate(s.path)}
              className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium transition ${
                active ? 'bg-ink-main text-paper shadow-sm' : done ? 'text-ink-main hover:bg-paper' : 'text-ink-muted hover:bg-paper hover:text-ink-main'}`}
            >
              <span className={`flex h-4 w-4 items-center justify-center rounded-full text-[9px] font-bold ${
                active ? 'bg-paper text-ink-main' : done ? 'bg-sage-dark text-white' : 'bg-sand-light text-ink-muted'}`}>
                {done ? '✓' : i + 1}
              </span>
              {(!compact || active) && s.label}
            </button>
            {i < JOURNEY.length - 1 && <span className="text-ink-soft/50">›</span>}
          </li>
        );
      })}
    </ol>
  );
}

/** Big "Next step" button for the current page (hidden on the last step). */
export function JourneyNext() {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const i = JOURNEY.findIndex(s => s.path === pathname);
  if (i < 0 || i >= JOURNEY.length - 1) return null;
  const next = JOURNEY[i + 1];
  return (
    <div className="flex justify-end">
      <button
        onClick={() => navigate(next.path)}
        className="flex items-center gap-2 rounded-xl bg-ink-main px-5 py-3 text-sm font-medium text-paper shadow-[0_4px_14px_rgba(44,41,38,0.2)] transition hover:bg-ink-main/90"
      >
        Next: {next.next} <ArrowRight className="h-4 w-4" />
      </button>
    </div>
  );
}
