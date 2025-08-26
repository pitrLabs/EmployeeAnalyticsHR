import json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import CommandJob
from azure.identity import DefaultAzureCredential

with open("config.json", "r") as f:
    config = json.load(f)

SUBSCRIPTION = config["subscription_id"]
RESOURCE_GROUP = config["resource_group"]
WS_NAME = config["workspace_name"]

credential = DefaultAzureCredential()
ml_client = MLClient(credential=credential,
                     subscription_id=SUBSCRIPTION,
                     resource_group_name=RESOURCE_GROUP,
                     workspace_name=WS_NAME)

job = CommandJob(code=".",
                 command="python src/train.py",
                 environment="python-sdk-v2:35",
                 compute="cpu-cluster",
                 display_name="hr-analytics-train-job",
                 experiment_name="hr-analytics-exp")

returned_job = ml_client.jobs.create_or_update(job)
print(f"Submitted job: {returned_job.name}")
print(f"Job status: {returned_job.status}")
