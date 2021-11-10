import json
from time import time
import triton_python_backend_utils as pb_utils
import numpy as np
import cv2
import os
import utils
from mtcnn import mtcnn
from facenet import facenet


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])

        self.is_init = False

    def _build_face_dataset(self):
        self.mtcnn_model = mtcnn()
        self.threshold = [0.5, 0.6, 0.8]
        self.facenet_model = facenet()
        face_list = os.listdir('face_dataset')
        self.known_face_encodings = []
        self.known_face_names = []
        for face in face_list:
            name = face.split('.')[0]
            img = cv2.imread('./face_dataset/' + face)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #---------------------#
            #   检测人脸
            #---------------------#
            rectangles = self.mtcnn_model.detect_face(img, self.threshold)
            if len(rectangles) <= 0:
                continue
            #---------------------#
            #   转化成正方形
            #---------------------#
            rectangles = utils.rect2square(np.array(rectangles))
            #-----------------------------------------------#
            #   facenet要传入一个160x160的图片
            #   利用landmark对人脸进行矫正
            #-----------------------------------------------#
            rectangle = rectangles[0]
            landmark = np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])
            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            print(crop_img.shape)
            crop_img, _ = utils.Alignment_1(crop_img, landmark)
            # cv2.imwrite(name+'.jpg', crop_img)
            # print(crop_img.shape)
            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)
            #--------------------------------------------------------------------#
            #   将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
            #--------------------------------------------------------------------#
            face_encoding = self.facenet_model.calc_128_vec(crop_img)

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)

    def execute(self, requests):
        if not self.is_init:
            self._build_face_dataset()
            self.is_init = True
        output0_dtype = self.output0_dtype
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, 'INPUT0')
            t1 = time()
            img = self._recognize(in_0.as_numpy())
            t2 = time()
            print('inference time cost: {} ms'.format(1000 * (t2 - t1)))
            out_tensor_0 = pb_utils.Tensor('OUTPUT0', img.astype(output0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Cleaning up...')

    def _recognize(self, draw):
        #-----------------------------------------------#
        #   人脸识别
        #   先定位，再进行数据库匹配
        #-----------------------------------------------#
        height, width, _ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        #--------------------------------#
        #   检测人脸
        #--------------------------------#
        rectangles = self.mtcnn_model.detect_face(draw_rgb, self.threshold)

        if len(rectangles)==0:
            return

        # 转化成正方形
        rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
        rectangles[:, [0,2]] = np.clip(rectangles[:, [0,2]], 0, width)
        rectangles[:, [1,3]] = np.clip(rectangles[:, [1,3]], 0, height)

        #-----------------------------------------------#
        #   对检测到的人脸进行编码
        #-----------------------------------------------#
        face_encodings = []
        for rectangle in rectangles:
            #---------------#
            #   截取图像
            #---------------#
            landmark = np.reshape(rectangle[5:15], (5,2)) - np.array([int(rectangle[0]), int(rectangle[1])])
            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            #-----------------------------------------------#
            #   利用人脸关键点进行人脸对齐
            #-----------------------------------------------#
            crop_img, _ = utils.Alignment_1(crop_img, landmark)
            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)

            face_encoding = self.facenet_model.calc_128_vec(crop_img)
            face_encodings.append(face_encoding)

        face_names = []
        for face_encoding in face_encodings:
            #-------------------------------------------------------#
            #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
            #-------------------------------------------------------#
            matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance = 0.5)
            name = "Unknown"
            #-------------------------------------------------------#
            #   找出距离最近的人脸
            #-------------------------------------------------------#
            face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
            #-------------------------------------------------------#
            #   取出这个最近人脸的评分
            #-------------------------------------------------------#
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        rectangles = rectangles[:, 0:4]
        #-----------------------------------------------#
        #   画框~!~
        #-----------------------------------------------#
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left , bottom - 15), font, 0.75, (255, 255, 255), 2)
        return draw

