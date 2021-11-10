import json
import triton_python_backend_utils as pb_utils
import numpy as np
import cv2
import os
import utils
import time
from torch.utils.dlpack import from_dlpack


def pb_tensor_to_numpy(pb_tensor):
    if pb_tensor.is_cpu():
        return pb_tensor.as_numpy()
    else:
        pytorch_tensor = from_dlpack(pb_tensor.to_dlpack())
        return pytorch_tensor.cpu().numpy()


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])
        print('----------------------------------')
        print(output0_config['data_type'])
        print(self.output0_dtype)
        print(os.getcwd())
        print('----------------------------------')

        self.Pnet_inputs = ['input_1']
        self.Pnet_outputs = ['conv4-1', 'conv4-2']
        self.Rnet_inputs = ['input_1']
        self.Rnet_outputs = ['conv5-1', 'conv5-2']
        self.Onet_inputs = ['input_1']
        self.Onet_outputs = ['conv6-1', 'conv6-2', 'conv6-3']

    def execute(self, requests):
        output0_dtype = self.output0_dtype
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, 'INPUT0')
            img = self._detect_face_on_img(in_0.as_numpy())
            out_tensor_0 = pb_utils.Tensor('OUTPUT0', img.astype(output0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Cleaning up...')

    def _detect_face(self, img, threshold):
        #-----------------------------#
        #        归一化
        #-----------------------------#
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape
        # print("orgin image's shape is: ", origin_h, origin_w)
        #-----------------------------#
        #        计算原始输入图像
        #        每一次缩放的比例
        #-----------------------------#
        scales = utils.calculateScales(img)

        out = []

        #-----------------------------#
        #        粗略计算人脸框
        #        pnet部分
        #-----------------------------#
        count = 0
        for scale in scales:
            # print('{}/{}'.format(count, len(scales)))
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = np.expand_dims(scale_img, 0).astype(np.float32)
            inference_request = pb_utils.InferenceRequest(
                model_name='pnet',
                requested_output_names=[self.Pnet_outputs[0], self.Pnet_outputs[1]],
                inputs=[pb_utils.Tensor(self.Pnet_inputs[0], inputs)]
            )
            inference_response = inference_request.exec()
            # print('pnet inference finished ' + str(pb_utils.get_output_tensor_by_name(inference_response, self.Pnet_outputs[0]).is_cpu()))
            # print(pb_utils.get_output_tensor_by_name(inference_response, self.Pnet_outputs[0]).triton_dtype)

            out0 = pb_tensor_to_numpy(pb_utils.get_output_tensor_by_name(inference_response, self.Pnet_outputs[0]))[0]
            out1 = pb_tensor_to_numpy(pb_utils.get_output_tensor_by_name(inference_response, self.Pnet_outputs[1]))[0]

            output = [out0, out1]
            out.append(output)
            # print(out0.shape, out1.shape)
            count += 1

        # print('finish')
        rectangles = []
        #-------------------------------------------------#
        #   在这个地方我们对图像金字塔的预测结果进行循环
        #   取出每张图片的种类预测和回归预测结果
        #-------------------------------------------------#
        for i in range(len(scales)):
            # print(out[i][0][:, :, 1].shape)
            #------------------------------------------------------------------#
            #   为了方便理解，这里和视频上看到的不太一样
            #   因为我们在上面对图像金字塔循环的时候就把batch_size维度给去掉了
            #------------------------------------------------------------------#
            cls_prob = out[i][0][:, :, 1]
            roi = out[i][1]
            #--------------------------------------------#
            #   取出每个缩放后图片的高宽
            #--------------------------------------------#
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            #--------------------------------------------#
            #   解码的过程
            #--------------------------------------------#
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)

        #-----------------------------------------#
        #    进行非极大抑制
        #-----------------------------------------#
        rectangles = np.array(utils.NMS(rectangles, 0.7))
        # print(rectangles)

        if len(rectangles) == 0:
            return rectangles

        #-----------------------------------------#
        #    稍微精确计算人脸框
        #    Rnet部分
        #-----------------------------------------#
        predict_24_batch = []
        for rectangle in rectangles:
            #--------------------------------------------#
            #    利用获取到的粗略坐标，在原图上进行截取
            #--------------------------------------------#
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            #--------------------------------------------#
            #    将截取到的图片进行resize，调整成24x24的大小
            #--------------------------------------------#
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)
        
        # print('rnet start Inference')
        inference_request = pb_utils.InferenceRequest(
            model_name='rnet',
            requested_output_names=[self.Rnet_outputs[0], self.Rnet_outputs[1]],
            inputs=[pb_utils.Tensor(self.Rnet_inputs[0], np.array(predict_24_batch).astype(np.float32))]
        )
        inference_response = inference_request.exec()
        out0 = pb_tensor_to_numpy(pb_utils.get_output_tensor_by_name(inference_response, self.Rnet_outputs[0]))
        out1 = pb_tensor_to_numpy(pb_utils.get_output_tensor_by_name(inference_response, self.Rnet_outputs[1]))
        cls_prob = out0
        roi_prob = out1
        # print('rnet finish Inference')

        #-------------------------------------#
        #   解码的过程
        #-------------------------------------#
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

        if len(rectangles) == 0:
            return rectangles

        # print(rectangles)

        #-----------------------------#
        #   计算人脸框
        #   onet部分
        #-----------------------------#
        predict_batch = []
        for rectangle in rectangles:
            #------------------------------------------#
            #   利用获取到的粗略坐标，在原图上进行截取
            #------------------------------------------#
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            #-----------------------------------------------#
            #   将截取到的图片进行resize，调整成48x48的大小
            #-----------------------------------------------#
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        # print('onet start Inference')
        inference_request = pb_utils.InferenceRequest(
            model_name='onet',
            requested_output_names=[self.Onet_outputs[0], self.Onet_outputs[1], self.Onet_outputs[2]],
            inputs=[pb_utils.Tensor(self.Onet_inputs[0], np.array(predict_batch).astype(np.float32))]
        )
        inference_response = inference_request.exec()
        out0 = pb_tensor_to_numpy(pb_utils.get_output_tensor_by_name(inference_response, self.Onet_outputs[0]))
        out1 = pb_tensor_to_numpy(pb_utils.get_output_tensor_by_name(inference_response, self.Onet_outputs[1]))
        out2 = pb_tensor_to_numpy(pb_utils.get_output_tensor_by_name(inference_response, self.Onet_outputs[2]))
        cls_prob = out0
        roi_prob = out1
        pts_prob = out2
        # print('onet finish Inference')

        #-------------------------------------#
        #   解码的过程
        #-------------------------------------#
        # print('cls_prob:')
        # print(cls_prob)
        # print('roi_prob:')
        # print(roi_prob)
        # print('pts_prob:')
        # print(pts_prob)
        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])

        return rectangles

    def _detect_face_on_img(self, img):
        threshold = [0.5, 0.6, 0.7]
        temp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t1 = time.time()
        rectangles = self._detect_face(temp_img, threshold)
        draw = img.copy()

        for rectangle in rectangles:
            cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])),
                        (int(rectangle[2]), int(rectangle[3])), (0, 0, 255), 2)

            for i in range(5, 15, 2):
                cv2.circle(draw, (int(rectangle[i+0]), int(rectangle[i+1])), 1, (255, 0, 0), 4)
        
        t2 = time.time()
        print('inference time is: {}ms'.format(1000 * (t2 - t1)))
        # cv2.imwrite('out.jpg', draw)
        return draw
