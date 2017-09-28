import argparse
import tools.find_mxnet
import mxnet as mx
import os
import importlib
import sys
from detect.detector import Detector

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx,
                 nms_thresh=0.5, force_nms=True):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    force_nms : bool
        force suppress different categories
    """
    sys.path.append(os.path.join(os.getcwd(), 'symbol'))
    if net is not None:
        net = importlib.import_module("symbol_" + net) \
            .get_symbol(len(CLASSES), nms_thresh, force_nms)
    detector = Detector(net, prefix + "_" + str(data_shape), epoch, \
        data_shape, mean_pixels, ctx=ctx)
    return detector



if __name__ == '__main__':
    ctx = mx.gpu(0)
    network = 'vgg16_ssd_300'
    prefix = os.path.join(os.getcwd(), 'model', 'ssd')
    epoch = 0
    data_shape = 300
    mean_r = 123
    mean_g = 117
    mean_b = 104
    nms_thresh = 0.5
    force_nms = True
    _dir = ''
    _extension = ''
    thresh = 0.71
    show_timer = True
    for i in range (2323):
        image_list = ['/home/lizhuyun/video/video/frame%d.jpg'%i]
        print (image_list)

        # parse image list
        # image_list = ["data/demo/dog.jpg"]
        # assert len(image_list) > 0, "No valid image specified to detect"

        detector = get_detector(network, prefix,epoch,
                            data_shape,
                            (mean_r, mean_g,mean_b),
                            ctx, nms_thresh, force_nms)
        # run detection
        detector.detect_and_visualize(i,image_list, _dir, _extension,
                                  CLASSES, thresh, show_timer)

