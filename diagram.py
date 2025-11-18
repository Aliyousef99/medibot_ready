"""
Generate an HTML file that renders the MediBot data-flow diagram using Mermaid.

Usage:
  python diagram.py            # writes diagram.html next to this file
  python diagram.py --open     # also opens in the default browser
  python diagram.py --out out.html
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap
import webbrowser


MERMAID = r"""
flowchart TD
  %% Level 0: High-level context
  subgraph L0[Level 0: System Context]
    U[Users]
    AI[AI Services<br/>(Gemini, Hugging Face)]
    MB[MediBot System]
    U -->|Uploads / Chat| MB
    MB -->|Calls| AI
    MB -->|Responses| U
  end

  %% Level 1: Detailed data flow
  subgraph L1[Level 1: Core Data Flows]
    FE[Frontend (React)]

    subgraph API[FastAPI Routes]
      R1[POST /api/extract_text]
      R2[POST /api/parse_lab]
      R3[POST /api/explain]
      R4[POST /api/chat]
    end

    subgraph SVC[Services]
      OCR[OCR Service]
      PARSE[Parsing Service]
      GEM[Gemini Client]
      RECS[Recommendations (recs.py + YAML rules)]
    end

    %% Subgraph titles must use [label] syntax without node-shape brackets
    subgraph DB[Database - SQLAlchemy]
      Ttmp[(Temp Extracted Text)]
      T1[(LabReport)]
      T2[(RecommendationSet)]
      T3[(Conversations)]
      T4[(Messages)]
    end

    U2[Users]
    AI2[AI Services<br/>(Gemini, Hugging Face)]

    %% User Upload Flow
    U2 -->|Upload file| FE --> R1 --> OCR --> Ttmp

    %% Lab Parsing Flow
    Ttmp --> R2 --> PARSE --> T1

    %% Explanation Flow
    T1 --> R3 --> GEM --> T1

    %% Recommendation Flow
    T1 --> RECS --> T2
    RECS -. optional rephrase .-> GEM

    %% Chat Flow
    U2 -->|Send message| FE --> R4 --> GEM
    R4 --> T3
    GEM --> R4 --> T4

    %% External AI link (for clarity)
    GEM --- AI2
  end

  %% Summary path removed (avoid linking to subgraphs directly)
"""


HTML_TEMPLATE = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>MediBot Data Flow Diagram</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
      .wrap { max-width: 1200px; margin: 0 auto; }
      .summary { margin-top: 16px; color: #333; }
      .mermaid { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; }
    </style>
    <script type=\"module\">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({ startOnLoad: true, theme: 'default' });
    </script>
  </head>
  <body>
    <div class=\"wrap\">
      <h1>MediBot Data Flow</h1>
      <div class=\"mermaid\">{MERMAID}</div>
      <p class=\"summary\"><strong>Summary:</strong> Frontend (React) → FastAPI Routes (Auth, OCR, Parse, Explain, Chat) → Services (OCR, Parser, Gemini, Recommendations) → Database (SQLAlchemy).</p>
    </div>
  </body>
</html>
"""


def generate_html(out_path: pathlib.Path, open_after: bool = False) -> pathlib.Path:
    html = HTML_TEMPLATE.replace("{MERMAID}", MERMAID)
    out_path.write_text(html, encoding="utf-8")
    if open_after:
        try:
            webbrowser.open(out_path.as_uri())
        except Exception:
            pass
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Mermaid diagram HTML for MediBot data flow")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("diagram.html"), help="Output HTML file path")
    parser.add_argument("--open", action="store_true", help="Open the generated HTML in a browser")
    args = parser.parse_args(argv)

    out = args.out if args.out.suffix else args.out.with_suffix(".html")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    generate_html(out, open_after=args.open)
    print(f"Wrote diagram to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
