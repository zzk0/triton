import numpy as np
import tritonclient.http as httpclient


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

    inputs = []
    inputs.append(httpclient.InferInput('INPUT0', [1, 10], "FP32"))
    input_data0 = np.random.randn(1, 10).astype(np.float32)
    inputs[0].set_data_from_numpy(input_data0, binary_data=True)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('OUTPUT0', binary_data=True))

    results = triton_client.infer('identity', inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('OUTPUT0')

    print(input_data0)
    print(output_data0)
