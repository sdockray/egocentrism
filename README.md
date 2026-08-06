# egocentrism

Video segmentation + audio (MFCC) feature extraction + 2D dimensionality
reduction over the [Ego4D](https://ego4d-data.org/docs/start-here) dataset,
running on the ARDC Nectar Research Cloud.

## Architecture (why it's shaped this way)

- **This container** does download + segment + extract + reduce. It runs on
  a small always-on CPU instance for now.
- **Nectar object storage (Swift)** holds the actual bytes: downloaded
  video, extracted audio segments, `.npy` feature arrays. See `src/storage.py`.
  It's durable and outlives any single instance — treat instances as disposable.
- **Postgres** (Dockerized, alongside the app, for now) holds metadata and
  small values you want to query/filter on: segment boundaries, feature
  summary stats, 2D projection coordinates, and a `runs` table linking
  every row back to the git commit + config that produced it. See `schema.sql`.
- We are **not** using Nectar's managed database service (Trove) yet, and
  **not** using Terraform/Ansible yet — deliberately. The schema will churn
  during prototyping; add the managed DB and IaC once the pipeline shape
  is proven, not before.
- GPU: not provisioned yet. `umap-learn`/scikit-learn on CPU is fine for
  prototyping on subsets. When reducing over the full corpus later,
  spin up a GPU instance running RAPIDS `cuML` for that step only, then
  terminate it — no need to keep a GPU instance running continuously.

## One-time setup

### 1. Ego4D access
1. Review and sign the license agreement: https://ego4d.dev/request/ego4d
   (~48hr approval, can be done as an individual).
2. You'll be emailed AWS credentials — **these expire in 14 days**, so
   don't set this up until you're ready to actually download. Put them
   in `.env` as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.
3. Size check: your Nectar allocation has ~1TB volume + 1.5TB object
   storage (~2.5TB total). Ego4D's full primary dataset is ~7.1TB. Start
   with a narrow subset (see `src/pipeline/download.py`) — don't run
   `--datasets full_scale` unscoped.
4. Download the Ego4D metadata JSON before running the Sonic export flow:
   ```bash
   mkdir -p ./2026
   docker compose run --rm app ego4d --aws_profile_name=ego4d --metadata -o ./2026/
   ```
   This should create `2026/ego4d.json`, which `src/sonic/dataset.py`
   expects when it builds the fake shop video list.
   The AWS profile is mounted automatically from `${HOME}/.aws` into
   `/root/.aws` for both `app` and `notebook` via `docker-compose.yml`.
   The `2026/` directory is also bind-mounted into the containers, so the
   downloaded metadata stays available for later runs of the app.
   The `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` values in `.env`
   are still available inside the container for AWS SDK code that uses
   environment-based credentials, but the Ego4D CLI profile lookup uses
   the mounted `~/.aws` files. Make sure the instance has a named profile
   called `ego4d` in `/home/ubuntu/.aws/credentials` and
   `/home/ubuntu/.aws/config` before running this.
   Do **not** use `sudo aws configure` here: it writes to `/root/.aws`,
   which is not what the container mounts. If you hit a permissions error,
   fix the host directory first and then run the command as `ubuntu`:
   ```bash
   sudo chown -R ubuntu:ubuntu ~/.aws
   chmod 700 ~/.aws
   chmod 600 ~/.aws/credentials ~/.aws/config
   aws configure --profile ego4d
   ```

### 2. Nectar object storage credentials
1. Nectar Dashboard → Identity → Application Credentials → Create.
   Scope it to the `egocentrism` project.
2. Copy the ID and Secret into `.env` immediately (the secret is only
   shown once) as `SWIFT_APPLICATION_CREDENTIAL_ID` / `_SECRET`.
3. This does not expire the way the Ego4D credentials do, but rotate it
   if it ever leaks — that's the point of using an application credential
   instead of your personal login here.

### 3. Local setup
```bash
cp .env.example .env   # then fill in the values from steps 1-2
docker compose build
docker compose up -d db
docker compose exec -T db psql -U egocentrism -d egocentrism < schema.sql
```

### 4. Smoke test
```bash
docker compose build --no-cache app
docker compose run --rm app python -m src.sonic.export_map_data --all
```

If you have only changed files under `src/`, the image rebuild is optional because those files are live-mounted into the container. If you change the Dockerfile or `requirements.txt`, rebuild first so the image picks up the new dependencies.

## Day to day

- Your existing pipeline logic goes into `src/pipeline/` as sibling
  modules to `download.py` (segment.py, features.py, reduce.py), wired
  up from `src/pipeline/__main__.py`.
- Every run should call `db.start_run(config)` once at the start and
  stamp every row it writes with the returned `run_id` — that's what
  makes results traceable back to the exact code/params that made them.
- Read/write large artifacts (audio, feature arrays) via `src/storage.py`,
  never local disk beyond `$SCRATCH_DIR` (mounted from the Nectar volume,
  see `docker-compose.yml` — update that mount path to match whatever
  volume you attach to the instance).

## Nectar instance setup (outside this repo)

1. Dashboard → Compute → Key Pairs → Create Key Pair, save the `.pem`
   somewhere like `~/.ssh/nectar.pem` and `chmod 600` it.
2. Dashboard → Compute → Instances → Launch. Start with something like
   `m3.medium` for the dev box — you don't need GPU/huge-RAM flavors for
   this instance. Select the key pair from step 1. Under
   "Configuration", paste the contents of `cloud-init.yaml` into the
   Customisation Script / User Data field — this installs Docker +
   Compose automatically at boot, so the instance is ready to use as
   soon as it's up (check with `docker --version` over SSH).
3. Dashboard → Network → Floating IPs → Allocate, then Associate it with
   the instance. Dashboard → Network → Security Groups → make sure port
   22 (SSH) is allowed inbound. Do **not** open port 8888 — Jupyter is
   reached via SSH tunnel (see below), not directly over the internet.
4. Dashboard → Network → Security Groups, add a Rule to the default security group
   to allow SSH incoming connections.
5. Dashboard → Volumes → Create Volume, attach it to the instance, then
   on the instance, format the volume and mount it to a directory:
   ```bash
   sudo mkfs.ext4 /dev/vdb
   sudo mkdir -p /mnt/data-volume
   sudo mount /dev/vdb /mnt/data-volume
   sudo chown -R ubuntu:ubuntu /mnt/data-volume
   ```
   This directory (`/mnt/data-volume`) is what `docker-compose.yml`'s volume mounts expect.
6. `git clone` this repo onto the instance, follow "One-time setup" above.
7. Remember to stop instances you're not using — the Service Unit budget
   is consumed by wall-clock runtime, not just by having resources
   allocated.

## Notebooks (without rebuilding anything)

The `notebook` service in `docker-compose.yml` runs JupyterLab in the
*same* image as `app` — same deps, same `src/` code (live-mounted, so
edits in Jupyter or your editor show up in both), same `.env` and
database access. Nothing needs rebuilding to switch between "run the
pipeline" and "poke at it in a notebook."

On the instance:
```bash
docker compose up -d notebook
docker compose logs notebook   # copy the token from the printed URL
```

On your laptop, open an SSH tunnel (leave this running):
```bash
ssh -i ~/.ssh/nectar.pem -L 8888:localhost:8888 ubuntu@<floating-ip>
```

Then open `http://localhost:8888` in your browser and paste the token.
Notebooks you save land in `./notebooks` on the instance (and are
git-trackable — commit the ones worth keeping, the rest is scratch).

Import your own modules directly, since `src/` is on the path inside the
container:
```python
from src import storage, db
```