import base64, json
from google.auth import default
from googleapiclient.discovery import build

PROJECT_ID = "weathercloud-460719"

def kill_switch(event, context):
    """Detach the project from its billing account
    when a Cloud-Billing Budget alert hits 100 %."""
    # 1️⃣ Decode Pub/Sub message
    if "data" not in event:          # defensive: some test publishes have only attributes
        return
    msg = json.loads(base64.b64decode(event["data"]).decode())

    # 2️⃣ Fire ONLY when cost ≥ budget
    if msg.get("costAmount", 0) < msg.get("budgetAmount", 0):
        return                       # still under budget

    # 3️⃣ Detach billing (idempotent – safe to repeat)
    creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-billing"])
    svc = build("cloudbilling", "v1", credentials=creds, cache_discovery=False)
    name = f"projects/{PROJECT_ID}"
    svc.projects().updateBillingInfo(
        name=name,
        body={"billingAccountName": ""}
    ).execute()
    print(f"Billing disabled for {PROJECT_ID}")