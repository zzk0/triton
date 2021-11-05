import numpy as np
import tritonclient.http as httpclient
import torch
from PIL import Image


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='172.17.0.2:8000')

    image = Image.open('cat.jpg')
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    inputs = []
    inputs.append(httpclient.InferInput('DALI_INPUT_0', image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('DALI_OUTPUT_0', binary_data=False))

    results = triton_client.infer('resnet50_dali', inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('DALI_OUTPUT_0')
    print(output_data0.shape)
    print(output_data0)
