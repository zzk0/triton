import numpy as np
import triton_python_backend_utils as pb_utils
import utils


class facenet(object):
    def __init__(self):
        self.Facenet_inputs =  ['input_1']
        self.Facenet_outputs =  ['Bottleneck_BatchNorm']

    def calc_128_vec(self, img):
        face_img = utils.pre_process(img)
        inference_request = pb_utils.InferenceRequest(
            model_name='facenet',
            requested_output_names=[self.Facenet_outputs[0]],
            inputs=[pb_utils.Tensor(self.Facenet_inputs[0], face_img.astype(np.float32))]
        )
        inference_response = inference_request.exec()
        pre = utils.pb_tensor_to_numpy(pb_utils.get_output_tensor_by_name(inference_response, self.Facenet_outputs[0]))
        pre = utils.l2_normalize(np.concatenate(pre))
        pre = np.reshape(pre, [128])
        
        return pre

