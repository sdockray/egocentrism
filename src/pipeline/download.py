"""
Example wrapper around the ego4d CLI.

Your Nectar allocation has ~1TB volume + 1.5TB object storage = ~2.5TB
total. Ego4D's full primary dataset alone is ~7.1TB, and the entire
dataset is 30+TB. Don't run `--datasets full_scale` unscoped — pick a
benchmark subset (e.g. `--benchmarks FHO` or a specific scenario) or a
video-uid list first, and treat "download everything" as a much later
decision once the pipeline is proven and you've thought about which
subset actually answers your research question.

Credentials: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (Ego4D's own,
issued after license approval, expire in 14 days — see .env.example).
"""
import subprocess


def download_subset(output_directory: str, datasets: list[str], benchmarks: list[str] | None = None):
    cmd = [
        "ego4d",
        "--output_directory", output_directory,
        "--datasets", *datasets,
        "--yes",  # skip interactive confirmation; you've already sized the request above
    ]
    if benchmarks:
        cmd += ["--benchmarks", *benchmarks]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # Small, cheap starting point: narrations + annotations only (~350MB),
    # enough to validate the pipeline shape before pulling any full-scale video.
    download_subset(
        output_directory="/data/scratch/ego4d",
        datasets=["annotations"],
    )
