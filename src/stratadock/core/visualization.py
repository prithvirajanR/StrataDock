from __future__ import annotations

import html
import json
from pathlib import Path


def write_3dmol_viewer_html(
    complex_pdb: Path,
    output_html: Path,
    *,
    title: str = "StrataDock Complex Viewer",
    receptor_style: str = "cartoon",
    ligand_style: str = "stick",
    surface_opacity: float = 0.18,
    show_interactions: bool = False,
    interactions: list[dict[str, object]] | None = None,
    title_metadata: dict[str, object] | None = None,
) -> Path:
    pdb_text = complex_pdb.read_text(encoding="utf-8")
    receptor_style_js = _style_js(
        receptor_style,
        {
            "cartoon": {"cartoon": {"color": "spectrum"}},
            "surface": {"line": {"colorscheme": "whiteCarbon", "opacity": 0.25}},
            "stick": {"stick": {"colorscheme": "whiteCarbon"}},
            "line": {"line": {"colorscheme": "whiteCarbon"}},
        },
        "receptor_style",
    )
    ligand_style_js = _style_js(
        ligand_style,
        {
            "stick": {"stick": {"colorscheme": "greenCarbon"}},
            "sphere": {"sphere": {"colorscheme": "greenCarbon", "scale": 0.32}},
            "line": {"line": {"colorscheme": "greenCarbon"}},
        },
        "ligand_style",
    )
    surface_opacity = min(1.0, max(0.0, float(surface_opacity)))
    interaction_rows = interactions or []
    metadata_rows = _metadata_html(title_metadata or {})
    interaction_panel = _interactions_html(interaction_rows, show_interactions=show_interactions)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.5.3/3Dmol-min.js"></script>
  <style>
    html, body {{ width: 100%; height: 100%; min-height: 660px; margin: 0; background: #1f1f1f; overflow: hidden; }}
    #viewer {{ position: fixed; inset: 0; width: 100%; height: 100%; min-height: 660px; }}
    #viewer canvas {{ position: absolute !important; inset: 0 !important; width: 100% !important; height: 100% !important; display: block !important; }}
    .label {{ position: fixed; left: 12px; top: 12px; color: #f2f2f2; font: 14px Arial, sans-serif; z-index: 2; max-width: 420px; }}
    .meta {{ color: #cfcfcf; margin-top: 4px; font-size: 12px; }}
    .meta div {{ margin-top: 2px; }}
    .interactions {{ margin-top: 10px; max-width: 360px; color: #e8e8e8; font-size: 12px; background: rgba(0,0,0,0.42); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 8px 10px; }}
    .interactions summary {{ cursor: pointer; }}
    .interactions ul {{ margin: 6px 0 0; padding-left: 18px; max-height: 150px; overflow-y: auto; }}
    .viewer-error {{ position: fixed; inset: 0; display: none; place-items: center; color: #f2f2f2; font: 14px Arial, sans-serif; background: #1f1f1f; text-align: center; padding: 24px; }}
  </style>
</head>
<body>
  <div class="label">{html.escape(title)}{metadata_rows}{interaction_panel}</div>
  <div id="viewer"></div>
  <div id="viewer-error" class="viewer-error"></div>
  <script>
    const pdbData = {json.dumps(pdb_text)};
    const interactions = {json.dumps(interaction_rows)};
    let viewerAttempts = 0;

    function showViewerError(message) {{
      const errorBox = document.getElementById("viewer-error");
      if (!errorBox) return;
      errorBox.textContent = "3D viewer could not initialize. " + message;
      errorBox.style.display = "grid";
    }}

    function hasWebGL() {{
      const canvas = document.createElement("canvas");
      return Boolean(
        window.WebGLRenderingContext &&
        (canvas.getContext("webgl") || canvas.getContext("experimental-webgl"))
      );
    }}

    function initializeViewer() {{
      try {{
        const container = document.getElementById("viewer");
        if (!container) {{
          throw new Error("Viewer container was not found.");
        }}
        const width = Math.max(window.innerWidth || document.documentElement.clientWidth || 0, 640);
        const height = Math.max(window.innerHeight || document.documentElement.clientHeight || 0, 660);
        container.style.width = width + "px";
        container.style.height = height + "px";
        const rect = container.getBoundingClientRect();
        if (rect.width < 10 || rect.height < 10) {{
          viewerAttempts += 1;
          window.setTimeout(initializeViewer, Math.min(1000, 120 + viewerAttempts * 30));
          return;
        }}
        if (typeof $3Dmol === "undefined") {{
          throw new Error("3Dmol.js did not load.");
        }}
        if (!hasWebGL()) {{
          throw new Error("WebGL is disabled or unavailable in this browser.");
        }}
        const viewer = $3Dmol.createViewer(container, {{ backgroundColor: "#1f1f1f" }});
        if (!viewer) {{
          throw new Error("3Dmol returned an empty viewer.");
        }}
        viewer.addModel(pdbData, "pdb");
        viewer.setStyle({{hetflag: false}}, {receptor_style_js});
        viewer.setStyle({{hetflag: true}}, {ligand_style_js});
        if ({surface_opacity} > 0) {{
          try {{
            viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: {surface_opacity}, color: "white"}}, {{hetflag: false}});
          }} catch (surfaceError) {{
            console.warn("3Dmol surface rendering failed", surfaceError);
          }}
        }}
        viewer.resize();
        viewer.zoomTo();
        viewer.center();
        viewer.render();
        window.setTimeout(() => {{
          try {{
            viewer.resize();
            viewer.zoomTo();
            viewer.center();
            viewer.render();
          }} catch (resizeError) {{
            console.warn("3Dmol resize failed", resizeError);
          }}
        }}, 120);
        window.setTimeout(() => {{
          try {{
            viewer.resize();
            viewer.zoomTo();
            viewer.center();
            viewer.render();
          }} catch (resizeError) {{
            console.warn("3Dmol delayed resize failed", resizeError);
          }}
        }}, 450);
      }} catch (error) {{
        console.error("3D viewer initialization failed", error);
        showViewerError(String(error.message || error));
      }}
    }}

    if (document.readyState === "loading") {{
      document.addEventListener("DOMContentLoaded", () => window.requestAnimationFrame(initializeViewer));
    }} else {{
      window.requestAnimationFrame(initializeViewer);
    }}
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return output_html


def _style_js(value: str, allowed: dict[str, dict[str, object]], field_name: str) -> str:
    key = value.lower()
    if key not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {choices}")
    return json.dumps(allowed[key])


def _metadata_html(metadata: dict[str, object]) -> str:
    if not metadata:
        return ""
    rows = "".join(
        f"<div>{html.escape(str(key))}: {html.escape(str(value))}</div>"
        for key, value in metadata.items()
    )
    return f'<div class="meta">{rows}</div>'


def _interactions_html(interactions: list[dict[str, object]], *, show_interactions: bool) -> str:
    if not show_interactions:
        return ""
    if not interactions:
        return '<details class="interactions"><summary>Interactions</summary><div>No interaction data available.</div></details>'
    rows = "".join(f"<li>{html.escape(_interaction_label(row))}</li>" for row in interactions[:12])
    remaining = len(interactions) - 12
    if remaining > 0:
        rows += f"<li>{html.escape(f'and {remaining} more contacts...')}</li>"
    summary = f"Interactions ({len(interactions)})"
    return f'<details class="interactions"><summary>{html.escape(summary)}</summary><ul>{rows}</ul></details>'


def _interaction_label(row: dict[str, object]) -> str:
    residue = row.get("receptor_residue")
    if not residue:
        residue_parts = [
            str(row.get("residue_name", "") or ""),
            str(row.get("chain_id", "") or ""),
            str(row.get("residue_seq", "") or ""),
        ]
        residue = " ".join(part for part in residue_parts if part).strip()
    ligand = row.get("ligand_atom")
    if not ligand:
        ligand_index = row.get("ligand_atom_index")
        ligand_element = row.get("ligand_element")
        if ligand_index not in (None, "") and ligand_element not in (None, ""):
            ligand = f"{ligand_element}{ligand_index}"
        elif ligand_index not in (None, ""):
            ligand = f"ligand atom {ligand_index}"
    distance = row.get("distance_angstrom")
    parts = [
        str(row.get("interaction_type") or row.get("type") or "contact").replace("_", " "),
        str(ligand or ""),
        str(residue or ""),
        f"{float(distance):.2f} A" if distance not in (None, "") else "",
    ]
    return " | ".join(part for part in parts if part)
