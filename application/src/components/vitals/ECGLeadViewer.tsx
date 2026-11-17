import { useMemo, useRef, useState, type MouseEvent } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Activity, Flame, Grid3x3, Rows3, Ruler, X } from 'lucide-react';
import { LEAD_NAMES, type DspMeasurements } from '../../lib/api';

interface ECGLeadViewerProps {
  /** Lead name -> samples (any length) covering `durationSec`. */
  leadsPreview?: Record<string, number[]> | null;
  /** 12 x N saliency matrix (0..1) aligned with the same duration. */
  heatmap?: number[][] | null;
  dsp?: DspMeasurements | null;
  isStaffMode?: boolean;
  durationSec?: number;
  title?: string;
  subtitle?: string;
}

// Classic ECG paper colours
const GRID_MINOR = '#f6d5d5';
const GRID_MAJOR = '#eaa3a3';
const TRACE = '#2C2926';

// Standard 12-lead print layout: rows x columns, each column = consecutive 2.5 s window.
const GRID_LAYOUT: string[][] = [
  ['DI', 'AVR', 'V1', 'V4'],
  ['DII', 'AVL', 'V2', 'V5'],
  ['DIII', 'AVF', 'V3', 'V6'],
];

function heatColor(v: number): string {
  // amber (#f59e0b) -> red (#dc2626)
  const t = Math.max(0, Math.min(1, (v - 0.35) / 0.65));
  const r = Math.round(245 + (220 - 245) * t);
  const g = Math.round(158 + (38 - 158) * t);
  const b = Math.round(11 + (38 - 11) * t);
  return `rgb(${r},${g},${b})`;
}

function buildPath(samples: number[], i0: number, i1: number, w: number, h: number, mvPerPx: number, baseline: number) {
  const n = Math.max(1, i1 - i0 - 1);
  let d = '';
  for (let i = i0; i < i1; i++) {
    const x = ((i - i0) / n) * w;
    const y = h / 2 - (samples[i] - baseline) / mvPerPx;
    d += `${i === i0 ? 'M' : 'L'}${x.toFixed(1)},${Math.max(-h, Math.min(2 * h, y)).toFixed(1)}`;
  }
  return d;
}

function median(values: number[]): number {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

/** Heatmap rectangles for one lead in the window [t0, t1] seconds. */
function HeatOverlay({ row, t0, t1, duration, w, h }: { row: number[]; t0: number; t1: number; duration: number; w: number; h: number }) {
  const binDur = duration / row.length;
  return (
    <g style={{ mixBlendMode: 'multiply' }}>
      {row.map((v, b) => {
        const bs = b * binDur, be = bs + binDur;
        if (v < 0.35 || be <= t0 || bs >= t1) return null;
        const x0 = ((Math.max(bs, t0) - t0) / (t1 - t0)) * w;
        const x1 = ((Math.min(be, t1) - t0) / (t1 - t0)) * w;
        return <rect key={b} x={x0} y={0} width={Math.max(1, x1 - x0)} height={h} fill={heatColor(v)} opacity={0.18 + 0.5 * v} />;
      })}
    </g>
  );
}

function PaperGrid({ id, w, h, pxPerMm }: { id: string; w: number; h: number; pxPerMm: number }) {
  return (
    <>
      <defs>
        <pattern id={`${id}-minor`} width={pxPerMm} height={pxPerMm} patternUnits="userSpaceOnUse">
          <path d={`M ${pxPerMm} 0 L 0 0 0 ${pxPerMm}`} fill="none" stroke={GRID_MINOR} strokeWidth={0.5} />
        </pattern>
        <pattern id={`${id}-major`} width={pxPerMm * 5} height={pxPerMm * 5} patternUnits="userSpaceOnUse">
          <rect width={pxPerMm * 5} height={pxPerMm * 5} fill={`url(#${id}-minor)`} />
          <path d={`M ${pxPerMm * 5} 0 L 0 0 0 ${pxPerMm * 5}`} fill="none" stroke={GRID_MAJOR} strokeWidth={0.9} />
        </pattern>
      </defs>
      <rect width={w} height={h} fill="#fffaf9" />
      <rect width={w} height={h} fill={`url(#${id}-major)`} />
    </>
  );
}

export default function ECGLeadViewer({
  leadsPreview, heatmap, dsp, isStaffMode = false, durationSec = 10.24,
  title = '12-Lead ECG', subtitle,
}: ECGLeadViewerProps) {
  const [layout, setLayout] = useState<'grid' | 'stacked'>('grid');
  const [showHeat, setShowHeat] = useState(true);
  const [zoomLead, setZoomLead] = useState<string | null>(null);
  const [paperSpeed, setPaperSpeed] = useState<25 | 50>(25);

  const leads = useMemo(() => LEAD_NAMES.filter(l => (leadsPreview?.[l]?.length ?? 0) > 1), [leadsPreview]);
  const baselines = useMemo(() => {
    const out: Record<string, number> = {};
    for (const l of leads) out[l] = median(leadsPreview![l]);
    return out;
  }, [leads, leadsPreview]);
  const heatRow = (lead: string) => {
    const i = LEAD_NAMES.indexOf(lead as (typeof LEAD_NAMES)[number]);
    return heatmap && heatmap[i] ? heatmap[i] : null;
  };

  if (!leadsPreview || leads.length === 0) {
    return (
      <div className="rounded-2xl border border-black/5 bg-paper p-6 text-center text-sm text-ink-muted">
        <Activity className="mx-auto mb-2 h-6 w-6 text-ink-soft" />
        No 12-lead waveform loaded yet. Upload an ECG or pick a CODE-15% preset.
      </div>
    );
  }

  const st = dsp?.st_status ?? null;
  const ribbon = [
    { label: 'Heart Rate', value: dsp?.heart_rate != null ? `${Math.round(dsp.heart_rate)} bpm` : '—' },
    { label: 'QRS Duration', value: dsp?.qrs_ms != null ? `${Math.round(dsp.qrs_ms)} ms` : '—', alert: (dsp?.qrs_ms ?? 0) >= 120 },
    { label: 'PR Interval', value: dsp?.pr_ms != null ? `${Math.round(dsp.pr_ms)} ms` : 'No P waves', alert: dsp?.pr_ms == null || (dsp?.pr_ms ?? 0) > 200 },
    { label: 'ST Segment', value: st ? `${st}${dsp?.st_amp != null ? ` (${dsp.st_amp > 0 ? '+' : ''}${dsp.st_amp.toFixed(2)} mV)` : ''}` : '—', alert: !!st && !st.startsWith('Normal') },
    { label: 'RR Irregularity', value: dsp?.rr_irregularity != null ? dsp.rr_irregularity.toFixed(3) : '—', alert: (dsp?.rr_irregularity ?? 0) > 0.15 },
  ];

  return (
    <div className="rounded-2xl border border-black/5 bg-paper shadow-soft">
      {/* Header + controls */}
      <div className="flex flex-wrap items-center gap-3 border-b border-black/5 px-4 py-3">
        <div className="mr-auto">
          <div className="flex items-center gap-2 text-sm font-semibold text-ink-main">
            <Activity className="h-4 w-4 text-rose-dark" /> {title}
          </div>
          {subtitle && <div className="text-[11px] text-ink-soft">{subtitle}</div>}
        </div>
        <div className="flex rounded-lg border border-black/5 bg-cream p-0.5 text-[11px] font-medium">
          <button onClick={() => setLayout('grid')} className={`flex items-center gap-1 rounded-md px-2 py-1 ${layout === 'grid' ? 'bg-paper text-ink-main shadow-sm' : 'text-ink-muted'}`}>
            <Grid3x3 className="h-3.5 w-3.5" /> 4×3
          </button>
          <button onClick={() => setLayout('stacked')} className={`flex items-center gap-1 rounded-md px-2 py-1 ${layout === 'stacked' ? 'bg-paper text-ink-main shadow-sm' : 'text-ink-muted'}`}>
            <Rows3 className="h-3.5 w-3.5" /> Stacked
          </button>
        </div>
        {heatmap && (
          <button
            onClick={() => setShowHeat(s => !s)}
            className={`flex items-center gap-1 rounded-lg border px-2.5 py-1 text-[11px] font-medium transition ${showHeat ? 'border-amber-dark/30 bg-amber-light text-amber-dark' : 'border-black/5 bg-cream text-ink-muted'}`}
          >
            <Flame className="h-3.5 w-3.5" /> Saliency heatmap {showHeat ? 'on' : 'off'}
          </button>
        )}
        {isStaffMode && (
          <div className="flex rounded-lg border border-black/5 bg-cream p-0.5 text-[11px] font-medium">
            {([25, 50] as const).map(s => (
              <button key={s} onClick={() => setPaperSpeed(s)} className={`rounded-md px-2 py-1 ${paperSpeed === s ? 'bg-paper text-ink-main shadow-sm' : 'text-ink-muted'}`}>
                {s} mm/s
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Staff DSP ribbon */}
      {isStaffMode && (
        <div className="grid grid-cols-2 gap-2 border-b border-black/5 bg-cream/50 px-4 py-2 sm:grid-cols-5">
          {ribbon.map(r => (
            <div key={r.label} className={`rounded-lg border px-2 py-1.5 ${r.alert ? 'border-rose-dark/20 bg-rose-light/40' : 'border-black/5 bg-paper'}`}>
              <div className="font-mono text-[9px] uppercase tracking-widest text-ink-soft">{r.label}</div>
              <div className={`text-xs font-semibold ${r.alert ? 'text-rose-dark' : 'text-ink-main'}`}>{r.value}</div>
            </div>
          ))}
        </div>
      )}

      {/* Lead panels */}
      <div className="p-3">
        {layout === 'grid' ? (
          <div className="grid grid-cols-4 gap-1.5">
            {GRID_LAYOUT.flatMap((row) => row.map((lead, c) => {
              const t0 = c * 2.5, t1 = t0 + 2.5;
              return (
                <LeadCell key={lead} lead={lead} samples={leadsPreview[lead] ?? []} baseline={baselines[lead] ?? 0}
                  t0={t0} t1={t1} duration={durationSec} heat={showHeat ? heatRow(lead) : null}
                  w={240} h={96} onClick={() => setZoomLead(lead)} />
              );
            }))}
            <div className="col-span-4">
              <LeadCell lead="DII" label="DII (rhythm strip)" samples={leadsPreview.DII ?? []} baseline={baselines.DII ?? 0}
                t0={0} t1={Math.min(10, durationSec)} duration={durationSec} heat={showHeat ? heatRow('DII') : null}
                w={984} h={96} onClick={() => setZoomLead('DII')} />
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-1">
            {leads.map(lead => (
              <LeadCell key={lead} lead={lead} samples={leadsPreview[lead]} baseline={baselines[lead] ?? 0}
                t0={0} t1={Math.min(10, durationSec)} duration={durationSec} heat={showHeat ? heatRow(lead) : null}
                w={984} h={64} onClick={() => setZoomLead(lead)} />
            ))}
          </div>
        )}
        <div className="mt-2 flex items-center justify-between text-[10px] text-ink-soft">
          <span>Click any lead to zoom · 10 mm/mV · {durationSec.toFixed(2)} s @ CODE-15% 400 Hz</span>
          {heatmap && showHeat && (
            <span className="flex items-center gap-1">
              Model attention
              <span className="inline-block h-2 w-16 rounded-full" style={{ background: 'linear-gradient(90deg,#f59e0b,#dc2626)' }} />
            </span>
          )}
        </div>
      </div>

      <AnimatePresence>
        {zoomLead && leadsPreview[zoomLead] && (
          <ZoomedLead
            lead={zoomLead}
            samples={leadsPreview[zoomLead]}
            baseline={baselines[zoomLead] ?? 0}
            duration={durationSec}
            heat={showHeat ? heatRow(zoomLead) : null}
            paperSpeed={isStaffMode ? paperSpeed : 25}
            calipers={isStaffMode}
            onClose={() => setZoomLead(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

function LeadCell({ lead, label, samples, baseline, t0, t1, duration, heat, w, h, onClick }: {
  lead: string; label?: string; samples: number[]; baseline: number; t0: number; t1: number; duration: number;
  heat: number[] | null; w: number; h: number; onClick: () => void;
}) {
  const i0 = Math.floor((t0 / duration) * samples.length);
  const i1 = Math.min(samples.length, Math.ceil((t1 / duration) * samples.length));
  // 10 mm/mV, grid at 25 mm/s: w px spans (t1-t0)*25 mm
  const pxPerMm = w / ((t1 - t0) * 25);
  const mvPerPx = 1 / (pxPerMm * 10);
  const id = `g-${lead}-${t0}-${w}`;
  return (
    <button onClick={onClick} className="group relative overflow-hidden rounded-md border border-black/5 text-left transition hover:ring-2 hover:ring-lavender-dark/40">
      <svg viewBox={`0 0 ${w} ${h}`} className="block h-auto w-full" preserveAspectRatio="none">
        <PaperGrid id={id} w={w} h={h} pxPerMm={pxPerMm} />
        {heat && <HeatOverlay row={heat} t0={t0} t1={t1} duration={duration} w={w} h={h} />}
        <path d={buildPath(samples, i0, i1, w, h, mvPerPx, baseline)} fill="none" stroke={TRACE} strokeWidth={1.1} strokeLinejoin="round" />
      </svg>
      <span className="absolute left-1.5 top-1 rounded bg-paper/80 px-1 font-mono text-[10px] font-bold text-ink-main">{label ?? lead}</span>
    </button>
  );
}

function ZoomedLead({ lead, samples, baseline, duration, heat, paperSpeed, calipers, onClose }: {
  lead: string; samples: number[]; baseline: number; duration: number; heat: number[] | null;
  paperSpeed: 25 | 50; calipers: boolean; onClose: () => void;
}) {
  const pxPerMm = 4;
  const w = Math.round(duration * paperSpeed * pxPerMm);
  const h = 40 * pxPerMm; // 40 mm = 4 mV window
  const mvPerPx = 1 / (pxPerMm * 10);
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<{ x: number; t: number; mv: number } | null>(null);
  const [anchor, setAnchor] = useState<{ x: number; t: number; mv: number } | null>(null);

  const pointAt = (e: MouseEvent<SVGSVGElement>) => {
    const rect = svgRef.current!.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * w;
    const t = (x / w) * duration;
    const idx = Math.max(0, Math.min(samples.length - 1, Math.round((t / duration) * (samples.length - 1))));
    return { x, t, mv: samples[idx] - baseline };
  };

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 z-[80] flex items-center justify-center bg-ink-main/40 p-6 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ y: 24, scale: 0.98 }} animate={{ y: 0, scale: 1 }} exit={{ y: 24, scale: 0.98 }}
        className="w-full max-w-6xl rounded-2xl bg-paper p-4 shadow-float"
        onClick={e => e.stopPropagation()}
      >
        <div className="mb-3 flex items-center gap-3">
          <div className="font-mono text-sm font-bold text-ink-main">Lead {lead}</div>
          <div className="text-[11px] text-ink-soft">{paperSpeed} mm/s · 10 mm/mV · 1 small box = {paperSpeed === 25 ? 40 : 20} ms × 0.1 mV</div>
          {calipers && (
            <div className="ml-auto flex items-center gap-2 rounded-lg border border-lavender-dark/20 bg-lavender-light/50 px-2.5 py-1 font-mono text-[11px] text-lavender-dark">
              <Ruler className="h-3.5 w-3.5" />
              {hover ? `t ${(hover.t * 1000).toFixed(0)} ms · ${hover.mv.toFixed(2)} mV` : 'Hover to measure'}
              {anchor && hover && ` · Δt ${Math.abs((hover.t - anchor.t) * 1000).toFixed(0)} ms · ΔV ${(hover.mv - anchor.mv).toFixed(2)} mV`}
              {anchor ? <button className="underline" onClick={() => setAnchor(null)}>clear</button> : <span className="text-ink-soft">(click to set caliper)</span>}
            </div>
          )}
          <button onClick={onClose} className={`${calipers ? '' : 'ml-auto'} rounded-lg p-1 text-ink-soft hover:bg-cream hover:text-ink-main`}>
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="overflow-x-auto rounded-lg border border-black/5">
          <svg
            ref={svgRef}
            width={w} height={h} viewBox={`0 0 ${w} ${h}`}
            className={calipers ? 'cursor-crosshair' : ''}
            onMouseMove={calipers ? e => setHover(pointAt(e)) : undefined}
            onMouseLeave={() => setHover(null)}
            onClick={calipers ? e => setAnchor(pointAt(e)) : undefined}
          >
            <PaperGrid id={`zoom-${lead}`} w={w} h={h} pxPerMm={pxPerMm} />
            {heat && <HeatOverlay row={heat} t0={0} t1={duration} duration={duration} w={w} h={h} />}
            <path d={buildPath(samples, 0, samples.length, w, h, mvPerPx, baseline)} fill="none" stroke={TRACE} strokeWidth={1.4} strokeLinejoin="round" />
            {calipers && anchor && <line x1={anchor.x} x2={anchor.x} y1={0} y2={h} stroke="#6b5fa8" strokeDasharray="4 3" />}
            {calipers && hover && (
              <>
                <line x1={hover.x} x2={hover.x} y1={0} y2={h} stroke="#6b5fa8" strokeWidth={0.8} />
                <line x1={0} x2={w} y1={h / 2 - hover.mv / mvPerPx} y2={h / 2 - hover.mv / mvPerPx} stroke="#6b5fa8" strokeWidth={0.5} strokeDasharray="2 3" />
              </>
            )}
          </svg>
        </div>
      </motion.div>
    </motion.div>
  );
}
