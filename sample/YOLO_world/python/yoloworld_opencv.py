#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import json
import time
import cv2
import argparse
import numpy as np
import clip as clip
import sophon.sail as sail
import logging
import torch

from postprocess_numpy import PostProcess
from utils import COCO_CLASSES, COLORS
logging.basicConfig(level=logging.INFO)

class TextEmbedder:
    def __init__(self,args):
        # self.device = select_device(device)
        self.clip_model, _ = clip.load(args.clip_bmodel, args.dev_id)

    def __call__(self, text):
        return self.embed_text(text)

    def embed_text(self, text):
        if not isinstance(text, list):
            text = [text]

        text_token = clip.tokenize(text).to("cpu")
        tokens = text_token.split(1)
        txt_feats = []
        for token in tokens:
            encoded_token = self.clip_model.encode_text(token)  # 编码
            #print("encoded_token:",encoded_token)
            detached_token =torch.Tensor(encoded_token).detach()
            #detached_token = encoded_token.detach()              # 分离
            txt_feats.append(detached_token)                     # 添加到列表中

        txt_feats = torch.cat(txt_feats, dim=0)
        txt_feats /= txt_feats.norm(dim=1, keepdim=True)
        txt_feats = txt_feats.unsqueeze(0)

        return txt_feats

class YOLOworld:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_name_txt = self.net.get_input_names(self.graph_name)[1]
        # self.output_names = self.net.get_output_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shape_txt = self.net.get_input_shape(self.graph_name, self.input_name_txt)
        
        self.clip = args.clip_bmodel
        self.dev_id = args.dev_id

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        self.num_classes = self.input_shape_txt[1]

        # get output
        self.output_names = self.net.get_output_names(self.graph_name)
        # self.output_names = ['output_Concat_f32']
        # self.output_names = [self.output_names]
        # print("output_names:",self.output_names)
   
        for output_name in self.output_names:
            output_shape = self.net.get_output_shape(self.graph_name, output_name)
            if(output_shape[1]>output_shape[2]):
                raise ValueError('Python programs do not support the OPT model')
         
            
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.agnostic = False
        self.multi_label = False
        self.max_det = 300

        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self, args):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

        self.text_embedder = TextEmbedder(args)
        num_nan_to_add = self.num_classes - len(args.class_names)
        num_classes_extended = args.class_names + ['nan'] * num_nan_to_add

        self.class_embeddings = self.text_embedder(num_classes_extended)

        class_embeddings = self.prepare_embeddings(self.class_embeddings)
        return class_embeddings

    def preprocess(self, ori_img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (3,h,w) numpy.ndarray after pre-processing

        """
        letterbox_img, ratio, (tx1, ty1) = self.letterbox(
            ori_img,
            new_shape=(self.net_h, self.net_w),
            color=(114, 114, 114),
            auto=False,
            scaleFill=False,
            scaleup=True,
            stride=32
        )

        img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img.astype(np.float32)
        # input_data = np.expand_dims(input_data, 0)
        img = np.ascontiguousarray(img / 255.0)
        return img, ratio, (tx1, ty1) 

    def prepare_embeddings(self, class_embeddings):
        if class_embeddings.shape[1] != self.num_classes:
            class_embeddings = torch.nn.functional.pad(class_embeddings, (0, 0, 0, self.num_classes - class_embeddings.shape[1]), mode='constant', value=0)
        
        return class_embeddings.cpu().numpy().astype(np.float32)

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return im, ratio, (dw, dh)

    def predict(self, input_img, img_num ,class_embeddings):
        input_data = {self.input_name: input_img, self.input_name_txt: class_embeddings}
        outputs = self.net.process(self.graph_name, input_data)
        out_keys = list(outputs.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n == k:
                    ord.append(i)
                    break
        out = [outputs[out_keys[i]][:img_num] for i in ord]
        return out
   

    def __call__(self, img_list,class_embeddings):
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()
            preprocessed_img, ratio, (tx1, ty1) = self.preprocess(ori_img)
            self.preprocess_time += time.time() - start_time
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])
        
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
        
        start_time = time.time()
        outputs = self.predict(input_img, img_num,class_embeddings)
        # print("outputs:",outputs)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return results

def draw_numpy(image, boxes, masks=None, classes_ids=None, conf_scores=None , classes=None):
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) + 1]
        else:
            color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            # print("classes_ids[idx]:",classes_ids[idx])
            cv2.putText(image, classes[classes_ids[idx]] + ':' + str(round(conf_scores[idx], 2)),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)
        # print("classe ss_ids[idx]:",classes_ids[idx])
        if masks is not None:
            mask = masks[:, :, idx]
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
        # print("COCO_CLASSES.index(classes[classes_ids[idx]]:",COCO_CLASSES.index(classes[classes_ids[idx]]))
        logging.debug("class id={}, score={}, (x1={},y1={},x2={},y2={})".format(COCO_CLASSES.index(classes[classes_ids[idx]]),conf_scores[idx], x1, y1, x2, y2))
    return image

def main(args): 
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    
    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir) 

    yoloworld = YOLOworld(args)
    batch_size = yoloworld.batch_size
    
    if args.class_names == ['all']:
            args.class_names= list(COCO_CLASSES)
    classes = args.class_names
    
    # warm up 
    # for i in range(10):
    #     results = yoloworld([np.zeros((640, 640, 3))])
    class_embeddings = yoloworld.init(args)

    decode_time = 0.0
    # test images
    if os.path.isdir(args.input): 
        img_list = []
        filename_list = []
        results_list = []
        cn = 0
        for root, dirs, filenames in os.walk(args.input):
            filenames.sort()
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                    continue
                img_file = os.path.join(root, filename)
                cn += 1
                logging.info("{}, img_file: {}".format(cn, img_file))
                # decode
                start_time = time.time()
                src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
                if src_img is None:
                    logging.error("{} imdecode is None.".format(img_file))
                    continue
                if len(src_img.shape) != 3:
                    src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
                decode_time += time.time() - start_time
                
                img_list.append(src_img)
                filename_list.append(filename)
                if len(img_list) == batch_size:
                    # predict
                    results = yoloworld(img_list, class_embeddings)
                    
                    for i, filename in enumerate(filename_list):
                        det = results[i]
                        # save image
                        det_draw = det[det[:, -2] > 0.25]
                        res_img = draw_numpy(img_list[i], det_draw[:,:4], masks=None, classes_ids=det_draw[:, -1], conf_scores=det_draw[:, -2], classes=classes)
                        cv2.imwrite(os.path.join(output_img_dir, filename), res_img)
                        
                        # save result
                        res_dict = dict()
                        res_dict['image_name'] = filename
                        res_dict['bboxes'] = []
                        for idx in range(det.shape[0]):
                            bbox_dict = dict()
                            x1, y1, x2, y2, score, category_id = det[idx]
                            bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
                            bbox_dict['category_id'] = COCO_CLASSES.index(classes[int(category_id)]) - 1
                            bbox_dict['score'] = float(round(score,5))
                            res_dict['bboxes'].append(bbox_dict)
                        results_list.append(res_dict)
                        
                    img_list.clear()
                    filename_list.clear()

        if len(img_list):
            results = yoloworld(img_list, class_embeddings)
            for i, filename in enumerate(filename_list):
                det = results[i]
                det_draw = det[det[:, -2] > 0.25]
                res_img = draw_numpy(img_list[i], det_draw[:,:4], masks=None, classes_ids=det_draw[:, -1], conf_scores=det_draw[:, -2], classes=classes)
                cv2.imwrite(os.path.join(output_img_dir, filename), res_img)
                res_dict = dict()
                res_dict['image_name'] = filename
                res_dict['bboxes'] = []
                for idx in range(det.shape[0]):
                    bbox_dict = dict()
                    x1, y1, x2, y2, score, category_id = det[idx]
                    bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
                    bbox_dict['category_id'] = COCO_CLASSES.index(classes[int(category_id)]) - 1
                    bbox_dict['score'] = float(round(score,5))
                    res_dict['bboxes'].append(bbox_dict)
                results_list.append(res_dict)
            img_list.clear()
            filename_list.clear()   

        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_opencv" + "_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            # json.dump(results_list, jf)
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

    # test video
    else:
        cap = cv2.VideoCapture()
        if not cap.open(args.input):
            raise Exception("can not open the video")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # print(fps, size)
        save_video = os.path.join(output_dir, os.path.splitext(os.path.split(args.input)[1])[0] + '.avi')
        out = cv2.VideoWriter(save_video, fourcc, fps, size)
        cn = 0
        frame_list = []
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            decode_time += time.time() - start_time
            if not ret or frame is None:
                break
            frame_list.append(frame)
            if len(frame_list) == batch_size:
                results = yoloworld(frame_list, class_embeddings)
                # print("results:",results)
                for i, frame in enumerate(frame_list):
                    det = results[i]
                    cn += 1
                    logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                    det_draw = det[det[:, -2] > 0.25]
                    res_frame = draw_numpy(frame_list[i], det_draw[:,:4], masks=None, classes_ids=det_draw[:, -1], conf_scores=det_draw[:, -2], classes=classes)
                    out.write(res_frame)
                frame_list.clear()
        if len(frame_list):
            results = yoloworld(frame_list, class_embeddings)
            for i, frame in enumerate(frame_list):
                det = results[i]
                cn += 1
                logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                det_draw = det[det[:, -2] > 0.25]
                res_frame = draw_numpy(frame_list[i], det_draw[:,:4], masks=None, classes_ids=det_draw[:, -1], conf_scores=det_draw[:, -2], classes=classes)
                out.write(res_frame)
        cap.release()
        out.release()
        logging.info("result saved in {}".format(save_video))

    
    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = yoloworld.preprocess_time / cn
    inference_time = yoloworld.inference_time / cn
    postprocess_time = yoloworld.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='yolov8s_text_world.bmodel', help='path of bmodel')
    parser.add_argument('--clip_bmodel', type=str, default='clip_text_vitb32_bm1684x_f16_1b.bmodel', help='path of clip')
    parser.add_argument('--class_names', nargs='+', default=["person", "car", "dog", "cat"], help='dev id')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='nms threshold')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
