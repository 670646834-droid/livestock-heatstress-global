"""Re-render ONLY Fig4 for the v3 figure set and audit it.

Panel (d) is the corrected US NASS two-segment fit figure (intercept +
state/month/year fixed effects, breakpoint THI 63.1).

The v3 set is E:\\1014deepseek\\review_1019\\09_Figures_v3.  Only Fig4 is
re-rendered (US NASS panel-d correction); the other eight v3 figures are not
touched.  Writes the figure into 09_Figures_v3/main/ and prints the audit
record as JSON for the QA report update.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import figstyle as S
import fig_numbers as F
import make_fig4 as M

ROOT = os.path.dirname(HERE)
V3 = os.path.join(ROOT, "09_Figures_v3")
V3_MAIN = os.path.join(V3, "main")


def main():
    S.setup_style()
    S.QA_RECORDS.clear()
    M.OUT = V3_MAIN
    M.main()
    rec = S.QA_RECORDS[-1]
    rec["panel_letters"] = list(rec["panel_letters"])
    print("\nAUDIT " + json.dumps(rec, indent=1, default=str))
    svg = os.path.join(V3_MAIN, "Fig4_dose_response_validation.svg")
    png = os.path.join(V3_MAIN, "Fig4_dose_response_validation.png")
    print("SVG bytes", os.path.getsize(svg))
    print("PNG bytes", os.path.getsize(png))
    dr = F.dose_response()
    print("NASS stats:", dr["nass"]["stats"])
    print("checks:", len(F.checks()))


if __name__ == "__main__":
    main()
