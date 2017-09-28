# MXnet-ssd webcam

All codes came from https://github.com/zhreshold/mxnet-ssd.  
I just added a progarm to make the detection of vedio.  

## Usage
* Git clone the codes
>`git clone 'https://github.com/bitterhoneyy/mxnet-ssd'

* Download the pretrained model: [`ssd_300_voc_0712.zip`](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.5-beta/vgg16_ssd_300_voc0712_trainval.zip), and extract to `model/` directory.
* Run
>`cd mxnet-ssd`  
`python vidoe.py`  
'python demo2.py --gpu 0'
mxnet-cuda8.0==0.9.5
