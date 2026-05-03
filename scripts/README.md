# StrataDock Scripts

Most users only need the root commands:

```bash
bash setup.sh
bash run_stratadock.sh
```

The scripts are grouped so the project is easier to scan:

- `install/`: installer helpers for dependencies, validation data, Vina, GNINA, and environment checks.
- `user/`: command-line tools a user may run directly for docking, pocket suggestion, receptor inspection, box creation, PDB download, or session export.
- `validation/`: bundled redocking checks used to confirm the pipeline is scientifically sane after changes.
- `dev/`: developer stress-test utilities that are not part of normal app usage.

Keep validation and test scripts in the repo. They are not part of the daily UI workflow, but they catch regressions when protein prep, ligand prep, fpocket, Vina, GNINA, reports, or session loading changes.
