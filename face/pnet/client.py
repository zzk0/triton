import numpy as np
import cv2
import tritonclient.http as httpclient
import time


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

    img = cv2.imread('cat.jpg').astype(np.float32)
    img = np.expand_dims(img, axis=0)

    inputs = []
    inputs.append(httpclient.InferInput('input_1', [*img.shape], "FP32"))
    inputs[0].set_data_from_numpy(img, binary_data=True)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('conv4-2', binary_data=True))
    outputs.append(httpclient.InferRequestedOutput('conv4-1', binary_data=True))

    t1 = time.time()
    results = triton_client.infer('pnet', inputs=inputs, outputs=outputs)
    t2 = time.time()
    print('inference time is: {}ms'.format(1000 * (t2 - t1)))
    output_data0 = results.as_numpy('conv4-2')

    print(img.shape)
    print(output_data0.shape)

