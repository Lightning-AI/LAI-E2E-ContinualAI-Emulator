import json

import boto3
import numpy as np

im_ = np.random.rand(1, 1, 28, 28)
image = np.array(im_, dtype=np.float32)
EndpointName = "name-mlm4xlarge-06-07-2022-19-36-07"
runtime = boto3.Session().client("sagemaker-runtime")
payload = json.dumps(image.tolist())
# Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
response = runtime.invoke_endpoint(EndpointName=EndpointName, ContentType="application/json", Body=payload)
result = json.loads(response["Body"].read().decode())
print(result)
