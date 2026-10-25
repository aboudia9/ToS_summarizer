"""Deploy the app to Google Cloud Run, with a hard $1/month spending cap.

Needs the gcloud CLI, `gcloud auth login`, and a project with billing linked.

    python scripts/deploy_cloudrun.py --project my-project --setup   # first time
    python scripts/deploy_cloudrun.py --project my-project           # redeploy after changes

--setup (safe to re-run) enables the APIs, creates a $1 budget wired to a Pub/Sub topic,
and deploys the billing kill-switch function that unlinks billing when the budget is hit.

The image is built on Cloud Build (x86, so it doesn't matter that the Mac is ARM) and pulls
the models from the HF model repo, so nothing large is uploaded from this machine.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.staging import ROOT, stage_app

SERVICE = "clause-and-effect"
# us-central1 is one of the regions covered by Cloud Run's free tier
REGION = "us-central1"
TOPIC = "billing-killswitch"
BUDGET_NAME = "clause-and-effect-cap"
APIS = [
    "run.googleapis.com", "cloudbuild.googleapis.com", "artifactregistry.googleapis.com",
    "cloudfunctions.googleapis.com", "eventarc.googleapis.com", "pubsub.googleapis.com",
    "cloudbilling.googleapis.com", "billingbudgets.googleapis.com",
]


def gcloud(*args, capture=False, check=True):
    cmd = ["gcloud", *args]
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=capture, check=False)
    if check and result.returncode != 0:
        if capture:
            print(result.stderr, file=sys.stderr)
        sys.exit(f"Command failed: {' '.join(cmd)}")
    return result.stdout.strip() if capture else result.returncode


def setup(project, budget_usd):
    gcloud("services", "enable", *APIS, "--project", project)

    billing_account = gcloud("billing", "projects", "describe", project,
                             "--format=value(billingAccountName)", capture=True).split("/")[-1]
    if not billing_account:
        sys.exit("No billing account linked to this project — link one in the console first")

    project_number = gcloud("projects", "describe", project, "--format=value(projectNumber)", capture=True)
    function_sa = f"{project_number}-compute@developer.gserviceaccount.com"

    # Pub/Sub topic the budget publishes to
    if gcloud("pubsub", "topics", "describe", TOPIC, "--project", project, capture=True, check=False) == "":
        gcloud("pubsub", "topics", "create", TOPIC, "--project", project)

    # $1 budget scoped to this project, wired to the topic (emails at 50%/90%/100% too)
    existing = gcloud("billing", "budgets", "list", f"--billing-account={billing_account}",
                      f"--filter=displayName={BUDGET_NAME}", "--format=value(name)", capture=True)
    if not existing:
        gcloud("billing", "budgets", "create", f"--billing-account={billing_account}",
               f"--display-name={BUDGET_NAME}", f"--budget-amount={budget_usd}USD",
               f"--filter-projects=projects/{project}",
               "--threshold-rule=percent=0.5", "--threshold-rule=percent=0.9", "--threshold-rule=percent=1.0",
               f"--notifications-rule-pubsub-topic=projects/{project}/topics/{TOPIC}")

    # Let the function's service account unlink billing from this project (and nothing else)
    gcloud("projects", "add-iam-policy-binding", project, f"--member=serviceAccount:{function_sa}",
           "--role=roles/billing.projectManager", "--condition=None", "--quiet", capture=True)

    gcloud("functions", "deploy", TOPIC, "--gen2", "--runtime=python312", f"--region={REGION}",
           f"--source={ROOT / 'deploy' / 'cloudrun' / 'billing_killswitch'}", "--entry-point=stop_billing",
           f"--trigger-topic={TOPIC}", f"--set-env-vars=TARGET_PROJECT={project}",
           "--max-instances=1", "--memory=256Mi", f"--project={project}", "--quiet")


def keep_only_recent_images(project):
    # Each deploy pushes a ~2GB image; storage past 0.5GB is billed, so keep just the latest two
    policy = [
        {"name": "keep-latest", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 2}},
        {"name": "delete-old", "action": {"type": "Delete"}, "condition": {"tagState": "any"}},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(policy, f)
    gcloud("artifacts", "repositories", "set-cleanup-policies", "cloud-run-source-deploy",
           f"--policy={f.name}", f"--location={REGION}", f"--project={project}", "--no-dry-run", "--quiet",
           check=False)


def deploy(project):
    with tempfile.TemporaryDirectory() as tmp:
        stage_app(tmp)
        gcloud("run", "deploy", SERVICE, f"--source={tmp}", f"--region={REGION}", f"--project={project}",
               "--port=8501",
               # 6Gi: models sit at ~3.2GB and BART generation peaks near 3.8GB
               "--memory=6Gi", "--cpu=2",
               # Scale to zero when idle, and never more than one instance (caps the worst case)
               "--min-instances=0", "--max-instances=1", "--concurrency=20",
               # Streamlit is one long websocket per tab; session affinity keeps a tab on its instance
               "--timeout=3600", "--session-affinity", "--cpu-boost",
               "--set-env-vars=IDLE_TIMEOUT_MINUTES=15",
               "--allow-unauthenticated", "--quiet")
    keep_only_recent_images(project)
    url = gcloud("run", "services", "describe", SERVICE, f"--region={REGION}", f"--project={project}",
                 "--format=value(status.url)", capture=True)
    print(f"\nLive at: {url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--setup", action="store_true", help="APIs, $1 budget, and billing kill switch")
    parser.add_argument("--budget", type=int, default=1, help="Monthly cap in USD (with --setup)")
    args = parser.parse_args()

    # Model download + torch install can outlast Cloud Build's default 10 minute timeout
    gcloud("config", "set", "builds/timeout", "1800", capture=True)

    if args.setup:
        setup(args.project, args.budget)
    deploy(args.project)


if __name__ == "__main__":
    main()
