import cv2
import numpy as np
import torch2trt
import torch
import yaml
import os

from loguru import logger
from time import time

from utils import NMS, fastNMS, scale_coords, plot_one_box, plot_one_box_PIL, colors, letterbox


class YOLOTRT:

    """
    INPUT: batch_size x 3 x 640 x 640 image
    OUTPUT: 1 x N x (5 + NumClass) results
    """

    input_name = ["input_0"]
    output_name = ["output_0"]
    img_size = 640
    batch_size = 1
    names = [f"{i+1}" for i in range(80)]

    conf_thres = 0.25
    nms_thres = 0.5

    def __init__(self,
                 engine_file=None,
                 pt_file=None,
                 img_size=None,
                 batch_size=None,
                 input_names=None,
                 output_names=None,
                 yaml_file=None):

        assert not (engine_file is None and pt_file is None)

        logger.info("generating model")
        self.model = torch2trt.TRTModule()
        if img_size is not None:
            self.img_size = img_size
        if batch_size is not None:
            self.batch_size = batch_size

        if yaml_file is not None:
            logger.info("loading params from yaml file")
            self._load_params(yaml_file)

        if input_names is not None:
            self.input_name = input_names
        if output_names is not None:
            self.output_name = output_names

        logger.info("loading model weights")
        if engine_file is not None:
            self.ckpt = {
                "engine": bytearray(open(engine_file, "rb").read()),
                "input_names": self.input_name,
                "output_names": self.output_name
            }
        else:
            ckpt = torch.load(pt_file, map_location="cpu")
            self.ckpt = ckpt["model"]
            self.names = ckpt["names"]
            self.img_size = ckpt["img_size"]
            self.batch_size = ckpt["batch_size"]

        self.model.load_state_dict(self.ckpt, strict=False)
        self.model.eval()
        self.model.cuda()

        # warmup
        x = torch.ones([self.batch_size, 3, self.img_size, self.img_size]).cuda()
        logger.info(f"tensorRT output shape: {x.shape}")
        logger.info(f"tensorRT output shape: {[i.shape for i in self.inference(x, False)]}")
        logger.info("tensorRT model loaded")

    def _load_params(self, yaml_file):
        if os.path.isfile(yaml_file):
            data = yaml.load(open(yaml_file).read(), yaml.Loader)
            for k in data:
                self.__setattr__(k, data[k])
        else:
            logger.info(f"no such file named {yaml_file}")

    def inference(self, x, postprocess=True):
        x = [i[0] for i in self.model(x)]
        return self.postprocess(x) if postprocess else x

    def __call__(self, imgs, end2end=False, count_time=False, draw=True):

        t0 = time()
        x = self.preprocess(imgs)
        t1 = time()
        x = self.inference(x, False)
        t2 = time()
        x = self.postprocess(x)
        t3 = time()

        show_str = f"time cost\n" \
                   f"preprocess: {(t1-t0)*1000:.1f}ms\n" \
                   f"inference: {(t2-t1)*1000:.1f}ms\n" \
                   f"postprocess: {(t3-t2)*1000:.1f}ms"

        if end2end:
            x = self.visualize(x, imgs, draw)
            t4 = time()
            show_str += f"\nvisualize: {(t4-t3)*1000:.1f}ms"

        if count_time:
            logger.info(show_str)
        return x

    def preprocess(self, imgs):
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
        ims = []
        for img in imgs:
            im = letterbox(img, (self.img_size, self.img_size), auto=False)[0]
            im = torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1)))
            if im.ndimension() == 3:
                im = im.unsqueeze(0)
            ims.append(im)
        return torch.cat(ims).cuda().float() / 255

    def postprocess(self, x):
        result = fastNMS(x, self.conf_thres, self.nms_thres)
        return result

    def visualize(self, pred, images, draw=True):
        if not isinstance(images, list) or not isinstance(images, tuple):
            images = [images]
        output_image = []
        output_json = []
        for i, det in enumerate(pred):
            this_image: np.ndarray = images[i]
            this_json = []
            if len(det):
                det[:, :4] = scale_coords([self.img_size] * 2,
                                          det[:, :4],
                                          this_image.shape).round()
                det[:, 4] *= det[:, 5]
                for *xyxy, conf, _, cls in reversed(det):
                    # print(xyxy)
                    # conf = conf0 * conf1
                    this_json.append({
                        "class": colors(int(cls)),
                        "bbox": xyxy,
                        "conf": conf
                    })
                    if draw:
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        # try:
                        plot_one_box(xyxy, this_image, label=label, color=colors(int(cls)), line_thickness=2)
                        # except:
                        #     plot_one_box_PIL(xyxy, this_image, label=label, color=colors(int(cls)), line_thickness=2)

            output_image.append(this_image)
            output_json.append(this_json)
        return output_image, output_json

    def save2pt(self, pt_file):
        ckpt = {
            "model": self.ckpt,
            "names": self.names,
            "img_size": self.img_size,
            "batch_size": self.batch_size
        }
        torch.save(ckpt, pt_file)
        logger.info(f"trt pth file saved to {pt_file}")

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()