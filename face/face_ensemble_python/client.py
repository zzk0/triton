import numpy as np
import cv2
import tritonclient.http as httpclient
import time


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

    img = cv2.imread('batman.jpg').astype(np.float32)

    inputs = []
    inputs.append(httpclient.InferInput('INPUT0', [*img.shape], "FP32"))
    # binary_data 默认是 True, 表示传输的时候使用二进制格式, 否则使用 JSON 文本(大小不一样)
    inputs[0].set_data_from_numpy(img, binary_data=True)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('OUTPUT0', binary_data=False))

    t1 = time.time()
    results = triton_client.infer('face_ensemble_python', inputs=inputs, outputs=outputs)
    t2 = time.time()
    print('inference time is: {}ms'.format(1000 * (t2 - t1)))
    output_data0 = results.as_numpy('OUTPUT0')

    print(img.shape)
    print(output_data0.shape)
    cv2.imwrite('out.jpg', output_data0.astype(np.uint8))
