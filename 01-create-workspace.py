# tutorial/01-create-workspace.py
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace

interactive_auth = InteractiveLoginAuthentication(tenant_id="99e1e721-7184-498e-8aff-b2ad4e53c1c2")
ws = Workspace.create(name='azure-ml',
            subscription_id='59e1d56a-8a2d-48a7-9cd3-a52c1a268c55',
            resource_group='cloud-ml',
            create_resource_group=True,
            location='eastus2',
            auth=interactive_auth
            )
            
# write out the workspace details to a configuration file: .azureml/config.json
ws.write_config(path='.azureml')    