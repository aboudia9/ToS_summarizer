"""Hard spending cap. A budget alert only *emails* you; this Cloud Function is subscribed
to the budget's Pub/Sub topic and unlinks billing from the project once the month's cost
reaches the budget — which shuts down everything billable (the app goes offline).

Re-enable in the console (Billing → Account management → link the project) when ready.
Based on https://cloud.google.com/billing/docs/how-to/disable-billing-with-notifications
"""
import base64
import json
import os

import functions_framework
from google.cloud import billing_v1

PROJECT_ID = os.environ["TARGET_PROJECT"]


@functions_framework.cloud_event
def stop_billing(cloud_event):
    data = json.loads(base64.b64decode(cloud_event.data["message"]["data"]).decode())
    cost, budget = data["costAmount"], data["budgetAmount"]
    print(f"Budget notification: cost={cost} budget={budget}")

    if cost < budget:
        return

    client = billing_v1.CloudBillingClient()
    name = f"projects/{PROJECT_ID}"
    if not client.get_project_billing_info(name=name).billing_enabled:
        print("Billing already disabled")
        return

    client.update_project_billing_info(
        name=name, project_billing_info=billing_v1.ProjectBillingInfo(billing_account_name="")
    )
    print(f"Cost {cost} reached budget {budget} — billing disabled for {PROJECT_ID}")
