import time
import numpy as np
import tritonclient.http as httpclient


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

    inputs = []
    inputs.append(httpclient.InferInput('INPUT0', [4], "FP32"))
    inputs.append(httpclient.InferInput('INPUT1', [4], "FP32"))
    input_data0 = np.random.randn(4).astype(np.float32)
    input_data1 = np.random.randn(4).astype(np.float32)
    inputs[0].set_data_from_numpy(input_data0, binary_data=False)
    inputs[1].set_data_from_numpy(input_data1, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('OUTPUT0', binary_data=False))
    outputs.append(httpclient.InferRequestedOutput('OUTPUT1', binary_data=False))

    t0 = time.time()
    results = triton_client.infer('example_python', inputs=inputs, outputs=outputs)
    t1 = time.time()
    print('inference time cost: {} ms'.format(1000 * (t1 - t0)))
    output_data0 = results.as_numpy('OUTPUT0')
    output_data1 = results.as_numpy('OUTPUT1')

    print(input_data0)
    print(input_data1)
    print(output_data0)
    print(output_data1)
