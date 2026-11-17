"""
Headless smoke test of the VitalSense UI (needs `npm run dev` on :5173 and the backend on :8000).

    python scripts/smoke_render.py                 # all journey pages
    python scripts/smoke_render.py /predictions    # one page

Renders each route in headless Chrome/Edge, then reports console errors and checks that
expected text is on the page.
"""
import html
import os
import re
import subprocess
import sys
import tempfile

BASE = os.environ.get("VS_FRONTEND", "http://localhost:5173")
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]
PAGES = {
    "/patients": ["Ramesh Kumar", "Eleanor Vance", "New Patient Profile"],
    "/dashboard": ["Ramesh Kumar", "MRN-2026-104"],
    "/vitals": ["New Hospital Visit", "Analyze & Save Visit", "Did the patient follow", "AF + LBBB High Risk"],
    "/predictions": ["SCD / cardiac risk", "ECG Heart Age", "Arrhythmia Probabilities", "SHAP", "DII"],
    "/history": ["What went wrong", "Biomarker", "Multi-Visit Trends", "Serum Creatinine"],
    "/insights": ["Why your plan changed", "Breakfast", "Foods to strictly avoid", "max heart rate"],
    "/chat": ["full memory", "What went wrong since my last hospital visit?"],
    "/reports": ["Before vs. Now", "Diet & Exercise Plan", "MRN-2026-104"],
    # Hospital Staff / Dietitian view
    "/predictions?mode=staff": ["Hospital Staff / Dietitian View", "QRS Duration", "PR Interval", "neural cross-attention"],
    "/insights?mode=staff": ["Review, Edit & Approve", "Approve & Publish to Patient Profile", "Staff notes"],
    "/chat?mode=staff": ["Summarize biomarker and 12-lead ECG deltas since last visit"],
    # back to patient view
    "/insights?mode=patient": ["Breakfast", "Why your plan changed"],
}


def render(path: str, browser: str) -> tuple[str, str]:
    profile = tempfile.mkdtemp(prefix="vs_smoke_")
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}",
           "--enable-logging=stderr", "--v=0", "--virtual-time-budget=12000", "--dump-dom", BASE + path]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    return p.stdout, p.stderr


def main() -> int:
    browser = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not browser:
        print("No Chrome/Edge found")
        return 2
    pages = {p: PAGES.get(p, []) for p in sys.argv[1:]} or PAGES
    failures = 0
    for path, expected in pages.items():
        dom, log = render(path, browser)
        body = dom.split("<body", 1)[-1]
        body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body, flags=re.S)
        text = html.unescape(re.sub(r"<[^>]+>", " ", body))
        benign = ("[vite] connecting", "[vite] connected", "React DevTools", "Download the React DevTools")
        errors = [l for l in log.splitlines() if "CONSOLE" in l and not any(b in l for b in benign)]
        missing = [e for e in expected if e.lower() not in text.lower()]
        ok = not errors and not missing and len(text.strip()) > 200
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {path}  ({len(text.split())} words)")
        for e in errors[:5]:
            print("    console:", e[-300:])
        if missing:
            print("    missing text:", missing)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
