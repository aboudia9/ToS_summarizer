"""Deploy the app to the Raspberry Pi (served publicly via Tailscale Funnel).

Copies only the app files + deploy/pi/ to ~/clause-and-effect on the Pi, then builds the
image there (natively on arm64, pulling the models from the HF model repo) and starts it.

    python scripts/deploy_pi.py               # deploy / redeploy
    python scripts/deploy_pi.py --host pii    # SSH host alias (default: pii)

The Pi needs ~/clause-and-effect/.env containing TS_AUTHKEY=... for the first start.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.staging import ROOT, stage_app

REMOTE_DIR = "clause-and-effect"


def run(cmd):
    print("$", " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="pii")
    parser.add_argument("--app-only", action="store_true",
                        help="Build/start only the app container (e.g. before the Tailscale key exists)")
    args = parser.parse_args()
    services = " app" if args.app_only else ""

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stage_app(tmp / "app")
        for f in ["docker-compose.yml", "serve.json"]:
            shutil.copy(ROOT / "deploy" / "pi" / f, tmp / f)

        run(["ssh", args.host, f"mkdir -p ~/{REMOTE_DIR}"])
        # --delete keeps the remote app/ in sync; .env (the auth key) is never touched
        run(["rsync", "-a", "--delete", "--exclude=.env", f"{tmp}/", f"{args.host}:{REMOTE_DIR}/"])

    run(["ssh", args.host, f"cd ~/{REMOTE_DIR} && docker compose up -d --build{services} && docker image prune -f"])


if __name__ == "__main__":
    main()
