# GCP Sweep Runner Quickstart

Use this guide to burst parameter sweeps onto a temporary Google Cloud VM. The workflow pulls the repo, runs `scripts/run_param_sweeps.py` in parallel, commits the resulting CSVs under `sweeps/`, and pushes back to whatever branch you specify.

## 1. Provision a VM

1. In the Google Cloud console (or `gcloud`), create a Compute Engine VM (Debian/Ubuntu image recommended).
2. Select machine type based on the parallelism you need (e.g. `n2-standard-16` for 16 vCPUs).
3. Allow HTTPS egress so the VM can reach GitHub and (optionally) download grid overrides.
4. Add a shutdown schedule or “delete VM when terminated” if you only need it for one run.

You can also template this with an instance template and set `--metadata-from-file startup-script` to call the runner automatically.

## 2. Copy the runner script

From your local repo root:

```bash
gcloud compute scp cloud/gcp_sweep_runner.sh USER@INSTANCE:~/gcp_sweep_runner.sh
```

Alternatively, paste the script in the Cloud Console editor.

## 3. Prepare secrets & env vars

SSH into the VM and export the required environment variables:

```bash
export GIT_AUTH_TOKEN="<github_personal_access_token>"
export GIT_REPO_URL="https://github.com/your-org/AI_trading_Bot.git"
export GIT_BRANCH="main"
export GIT_USER_NAME="Cloud Sweep"
export GIT_USER_EMAIL="cloud-sweep@example.com"
# Optional tuning:
export SWEEP_FILTER="reversion_band"        # only configs matching this substring
export INCLUDE_STRATS="maker_reversion_band maker_keltner_ride"
export GRID_OVERRIDE_URL="https://storage.googleapis.com/your-bucket/override.yaml"
export MAX_WORKERS=16                       # defaults to nproc if omitted
```

> **Tip:** store the token in Secret Manager and fetch it at runtime instead of pasting it into the shell history.

## 4. Run the sweeps

```bash
chmod +x ~/gcp_sweep_runner.sh
./gcp_sweep_runner.sh
```

The script will:

1. Install apt dependencies (`git`, `python3`, `venv`, `pip`, `curl`).
2. Clone the repo into `~/ai_trading_Bot` using the token.
3. Create/refresh `.venv`, install the project (`pip install -e .`).
4. Download an optional grid override file.
5. Execute `python scripts/run_param_sweeps.py --max-workers $MAX_WORKERS` with your filters.
6. Commit and push any new CSVs under `sweeps/`.

Check progress via `tail -f /var/log/syslog` or watch the console output. Sweep results land in `sweeps/maker_*_<timestamp>.csv` plus `sweep_summary.json` (so each VM run is auditable).

## 5. Tear down

After the script completes:

- Copy any additional artifacts you need (`gsutil cp sweeps/*.csv gs://your-bucket/`).
- Delete the VM to stop billing: `gcloud compute instances delete INSTANCE --zone=...`

## Cloud options & scheduling

- **Spot VMs**: launch the VM as a spot/preemptible instance to reduce cost—combine with a startup script that exports the env vars and runs `gcp_sweep_runner.sh` automatically.
- **Cloud Scheduler**: set a Scheduler job to call a Cloud Function that creates the VM, waits for it to finish, and deletes it when done.
- **Batch**: for many configs, consider wrapping `gcp_sweep_runner.sh` in a Cloud Batch task; each task can set a different `SWEEP_FILTER` or `INCLUDE_STRATS` value.

## Troubleshooting

- **Auth errors**: ensure `GIT_AUTH_TOKEN` has `repo` scope and the token is included in `GIT_REPO_URL` by the script.
- **No changes committed**: sweeps only commit when new/updated CSVs appear in `sweeps/`. Delete or archive old CSVs if you want a fresh diff per run.
- **Long runtimes**: increase `MAX_WORKERS` (up to vCPU count). Each worker processes one config at a time, so eight configs + `--max-workers 8` keeps all cores busy.

You can reuse the same VM by re-running the script; it’ll refresh the repo, rerun sweeps, and push again.
