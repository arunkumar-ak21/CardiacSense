/* eslint-disable react-refresh/only-export-components */
import {
  createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode,
} from 'react';
import {
  approveVisitPlan, createPatient, fetchPatients, fetchPatientVisits, submitCode15Visit,
  type DeltaSummary, type DietExercisePlan, type LongitudinalVisit, type NewPatientInput,
  type PatientProfile, type ShapDriver, type VisitPredictions,
} from '../lib/api';
import { usePredictionStore } from '../store/predictionStore';

export type { PatientProfile, LongitudinalVisit, DietExercisePlan } from '../lib/api';

export interface SubmittedVitals {
  hr: number;
  bp_sys: number;
  bp_dia: number;
  spo2: number;
  temp: number;
  resp: number;
  hrv: number;
  raw: Record<string, string>;
}

interface VitalsContextValue {
  // ── Legacy (unchanged API) ────────────────────────────────────────────────
  /** Manually submitted vitals, or the active visit's vitals when none were submitted. */
  submittedVitals: SubmittedVitals | null;
  submitVitals: (raw: Record<string, string>) => SubmittedVitals;
  setSubmittedVitals: (v: SubmittedVitals | null) => void;

  // ── Longitudinal state ────────────────────────────────────────────────────
  profiles: PatientProfile[];
  activeProfile: PatientProfile | null;
  visits: LongitudinalVisit[];            // newest first
  activeVisit: LongitudinalVisit | null;  // defaults to visits[0]
  previousVisit: LongitudinalVisit | null; // visit immediately before activeVisit
  isStaffMode: boolean;
  loadingProfiles: boolean;
  loadingVisits: boolean;
  longitudinalError: string | null;

  // Convenience views of the active visit
  predictions: VisitPredictions | null;
  shapDrivers: ShapDriver[];
  deltaSummary: DeltaSummary | null;
  dietPlan: DietExercisePlan | null;

  // ── Actions ───────────────────────────────────────────────────────────────
  selectProfile: (patientId: string) => Promise<void>;
  createNewProfile: (profileData: NewPatientInput) => Promise<PatientProfile>;
  selectVisit: (visitId: string) => void;
  toggleStaffMode: () => void;
  setIsStaffMode: (value: boolean) => void;
  recordNewVisit: (formData: FormData) => Promise<LongitudinalVisit>;
  approveActiveVisitPlan: (notes: string, updatedDietPlan?: DietExercisePlan) => Promise<LongitudinalVisit>;
  refreshProfiles: () => Promise<void>;
}

const VitalsContext = createContext<VitalsContextValue | null>(null);

const LS_PROFILE = 'vitalsense.activeProfileId';
const LS_STAFF = 'vitalsense.isStaffMode';

function readLS(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}
function writeLS(key: string, value: string) {
  try { localStorage.setItem(key, value); } catch { /* storage unavailable */ }
}

function sortNewestFirst(visits: LongitudinalVisit[]): LongitudinalVisit[] {
  return [...visits].sort((a, b) =>
    b.visit_date.localeCompare(a.visit_date) || (b.created_at ?? '').localeCompare(a.created_at ?? ''));
}

function vitalsFromVisit(v: LongitudinalVisit | null): SubmittedVitals | null {
  if (!v) return null;
  const n = (x: number | null | undefined) => (typeof x === 'number' ? x : 0);
  const { hr, bp_sys, bp_dia, spo2, temp, rr } = v.vitals;
  const raw: Record<string, string> = {
    hr: String(n(hr)), bp: `${n(bp_sys)}/${n(bp_dia)}`, spo2: String(n(spo2)),
    temp: String(n(temp)), resp: String(n(rr)), hrv: String(n(v.blood.hrv)),
  };
  return {
    hr: n(hr), bp_sys: n(bp_sys), bp_dia: n(bp_dia), spo2: n(spo2),
    temp: n(temp), resp: n(rr), hrv: n(v.blood.hrv), raw,
  };
}

export function VitalsProvider({ children }: { children: ReactNode }) {
  const [manualVitals, setManualVitals] = useState<SubmittedVitals | null>(null);
  const [profiles, setProfiles] = useState<PatientProfile[]>([]);
  const [activeProfileId, setActiveProfileId] = useState<string | null>(null);
  const [visits, setVisits] = useState<LongitudinalVisit[]>([]);
  const [activeVisitId, setActiveVisitId] = useState<string | null>(null);
  const [isStaffMode, setStaffMode] = useState<boolean>(() => {
    // `?mode=staff` / `?mode=patient` in the URL wins (handy for demos), else the remembered choice.
    const mode = typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('mode') : null;
    if (mode === 'staff' || mode === 'patient') {
      writeLS(LS_STAFF, String(mode === 'staff'));
      return mode === 'staff';
    }
    return readLS(LS_STAFF) === 'true';
  });
  const [loadingProfiles, setLoadingProfiles] = useState(true);
  const [loadingVisits, setLoadingVisits] = useState(false);
  const [longitudinalError, setLongitudinalError] = useState<string | null>(null);

  // ── Legacy submitVitals (unchanged behaviour) ─────────────────────────────
  const submitVitals = useCallback((raw: Record<string, string>): SubmittedVitals => {
    const bpParts = (raw.bp ?? '').split('/');
    const vitals: SubmittedVitals = {
      hr:     parseFloat(raw.hr)   || 0,
      bp_sys: parseFloat(bpParts[0]) || 0,
      bp_dia: parseFloat(bpParts[1]) || 0,
      spo2:   parseFloat(raw.spo2) || 0,
      temp:   parseFloat(raw.temp) || 0,
      resp:   parseFloat(raw.resp) || 0,
      hrv:    parseFloat(raw.hrv)  || 0,
      raw,
    };
    setManualVitals(vitals);
    return vitals;
  }, []);

  // ── Loading ───────────────────────────────────────────────────────────────
  const loadVisits = useCallback(async (patientId: string, selectId?: string) => {
    setLoadingVisits(true);
    try {
      const list = sortNewestFirst(await fetchPatientVisits(patientId));
      setVisits(list);
      setActiveVisitId(selectId && list.some(v => v.id === selectId) ? selectId : list[0]?.id ?? null);
    } catch (e) {
      setLongitudinalError(e instanceof Error ? e.message : 'Failed to load visits');
      setVisits([]);
      setActiveVisitId(null);
    } finally {
      setLoadingVisits(false);
    }
  }, []);

  const refreshProfiles = useCallback(async () => {
    try {
      setProfiles(await fetchPatients());
      setLongitudinalError(null);
    } catch (e) {
      setLongitudinalError(e instanceof Error ? e.message : 'Failed to load patients');
    }
  }, []);

  const selectProfile = useCallback(async (patientId: string) => {
    setActiveProfileId(patientId);
    writeLS(LS_PROFILE, patientId);
    setManualVitals(null);
    await loadVisits(patientId);
  }, [loadVisits]);

  // On mount: fetch profiles, select the remembered one or the first (Ramesh Kumar).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoadingProfiles(true);
      try {
        const list = await fetchPatients();
        if (cancelled) return;
        setProfiles(list);
        const remembered = readLS(LS_PROFILE);
        const first = list.find(p => p.id === remembered)
          ?? list.find(p => p.id === 'pat_ramesh_001')
          ?? list[0];
        if (first) {
          setActiveProfileId(first.id);
          await loadVisits(first.id);
        }
      } catch (e) {
        if (!cancelled) setLongitudinalError(e instanceof Error ? e.message : 'Failed to load patients');
      } finally {
        if (!cancelled) setLoadingProfiles(false);
      }
    })();
    return () => { cancelled = true; };
  }, [loadVisits]);

  // ── Derived state ─────────────────────────────────────────────────────────
  const activeProfile = useMemo(
    () => profiles.find(p => p.id === activeProfileId) ?? null, [profiles, activeProfileId]);
  const activeIndex = visits.findIndex(v => v.id === activeVisitId);
  const activeVisit = activeIndex >= 0 ? visits[activeIndex] : null;
  const previousVisit = activeIndex >= 0 ? visits[activeIndex + 1] ?? null : null;
  const submittedVitals = manualVitals ?? vitalsFromVisit(activeVisit);

  // Keep the legacy prediction store's vitals in sync with the selected visit
  // so DashboardSidebar / RightPanel show the active patient's numbers.
  const setLastVitals = usePredictionStore(s => s.setLastVitals);
  useEffect(() => {
    if (!activeVisit || !activeProfile) return;
    const v = activeVisit.vitals;
    setLastVitals({
      hr: v.hr ?? 0, bp_systolic: v.bp_sys ?? 0, bp_diastolic: v.bp_dia ?? 0,
      spo2: v.spo2 ?? 0, rr: v.rr ?? 0, temp: v.temp ?? 0,
      age: activeProfile.age, sex: activeProfile.gender,
    });
  }, [activeVisit, activeProfile, setLastVitals]);

  // ── Actions ───────────────────────────────────────────────────────────────
  const createNewProfile = useCallback(async (data: NewPatientInput) => {
    const created = await createPatient(data);
    setProfiles(prev => [created, ...prev.filter(p => p.id !== created.id)]);
    await selectProfile(created.id);
    return created;
  }, [selectProfile]);

  const selectVisit = useCallback((visitId: string) => {
    setActiveVisitId(visitId);
    setManualVitals(null);
  }, []);

  const setIsStaffMode = useCallback((value: boolean) => {
    setStaffMode(value);
    writeLS(LS_STAFF, String(value));
  }, []);
  const toggleStaffMode = useCallback(() => {
    setStaffMode(prev => { writeLS(LS_STAFF, String(!prev)); return !prev; });
  }, []);

  const recordNewVisit = useCallback(async (formData: FormData) => {
    const visit = await submitCode15Visit(formData);
    if (visit.patient_id !== activeProfileId) {
      setActiveProfileId(visit.patient_id);
      writeLS(LS_PROFILE, visit.patient_id);
    }
    setManualVitals(null);
    await loadVisits(visit.patient_id, visit.id);
    await refreshProfiles();
    return visit;
  }, [activeProfileId, loadVisits, refreshProfiles]);

  const approveActiveVisitPlan = useCallback(async (notes: string, updatedDietPlan?: DietExercisePlan) => {
    if (!activeVisit) throw new Error('No active visit to approve');
    const updated = await approveVisitPlan(activeVisit.id, {
      approved_by_staff: true, staff_notes: notes, ...(updatedDietPlan ? { diet_plan: updatedDietPlan } : {}),
    });
    setVisits(prev => prev.map(v => (v.id === updated.id ? { ...v, ...updated, heatmap: updated.heatmap ?? v.heatmap } : v)));
    void refreshProfiles(); // updates the "Hospital Verified" badge on profile cards
    return updated;
  }, [activeVisit, refreshProfiles]);

  const value: VitalsContextValue = {
    submittedVitals,
    submitVitals,
    setSubmittedVitals: setManualVitals,
    profiles,
    activeProfile,
    visits,
    activeVisit,
    previousVisit,
    isStaffMode,
    loadingProfiles,
    loadingVisits,
    longitudinalError,
    predictions: activeVisit?.predictions ?? null,
    shapDrivers: activeVisit?.shap ?? [],
    deltaSummary: activeVisit?.delta_summary ?? null,
    dietPlan: activeVisit?.diet_plan ?? null,
    selectProfile,
    createNewProfile,
    selectVisit,
    toggleStaffMode,
    setIsStaffMode,
    recordNewVisit,
    approveActiveVisitPlan,
    refreshProfiles,
  };

  return <VitalsContext.Provider value={value}>{children}</VitalsContext.Provider>;
}

export function useVitals(): VitalsContextValue {
  const ctx = useContext(VitalsContext);
  if (!ctx) throw new Error('useVitals must be used within VitalsProvider');
  return ctx;
}
