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
docker compose run --rm app
# should print "Started run <uuid>. Wire up download/segment/features/reduce steps here."
```

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

1. Dashboard → Compute → Instances → Launch. Start with something like
   `m3.medium` for the dev box; you don't need GPU/huge-RAM flavors for
   this instance.
2. Dashboard → Volumes → Create Volume, attach it, then on the instance:
   `mkfs.ext4` + mount it at (e.g.) `/mnt/data-volume` — this is what
   `docker-compose.yml`'s `app` volume mount expects.
3. Install Docker + Docker Compose on the instance, `git clone` this repo,
   follow "One-time setup" above.
4. Remember to stop instances you're not using — the Service Unit budget
   is consumed by wall-clock runtime, not just by having resources
   allocated.
