from collections import OrderedDict
import numpy as np
import math
import time

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import *
from utils import *


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)

def create_mtcnn_net_list(image_list, min_lp_size, device, p_model_path, o_model_path, prob_thresholds=0.8):
    assert p_model_path is not None and o_model_path is not None, 'this is for MTCNN evlaution only'

    pnet = PNet().to(device)
    onet = ONet().to(device)
    pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
    onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage))
    pnet.eval()
    onet.eval()

    with torch.no_grad():
        # since the 1st pass is slow, we past this one
        # start  = time.time()
        detect_onet(onet, image_list[0], np.array(detect_pnet(pnet, image_list[0], min_lp_size, device)), device)
        # print("image predicted in {:2.3f} fps".format((time.time() - start)))
        start  = time.time()
        bboxes_list = [detect_onet(onet, image, np.array(detect_pnet(pnet, image, min_lp_size, device)), device, prob_thresholds=prob_thresholds) for image in image_list]
        return bboxes_list
        print("image predicted in {:2.3f} fps".format((time.time() - start)/len(bboxes_list)))
    
    


def create_mtcnn_net(image, min_lp_size, device, p_model_path=None, o_model_path=None):

    bboxes = np.array([])

    if p_model_path is not None:
        pnet = PNet().to(device)
        pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
        pnet.eval()

        bboxes = detect_pnet(pnet, image, min_lp_size, device)

    if o_model_path is not None:
        onet = ONet().to(device)
        onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage))
        onet.eval()

        bboxes = detect_onet(onet, image, bboxes, device)

    return bboxes

def detect_pnet(pnet, image, min_lp_size, device):

    thresholds = 0.6 # lp detection thresholds
    nms_thresholds = 0.4

    # BUILD AN IMAGE PYRAMID
    height, width = image.shape
    fator_height = height
    fator_width  = width

    factor = 0.707  # sqrt(0.5)
    # scales for scaling the image
    scales = []
    factor_count = 0
    while fator_height > min_lp_size[1] and fator_width > min_lp_size[0]:
        scales.append(factor ** factor_count)
        fator_height *= factor
        fator_width  *= factor
        factor_count += 1

    # it will be returned
    bounding_boxes = []

    with torch.no_grad():
        # run P-Net on different scales
        for scale in scales:
            sw, sh = math.ceil(width * scale), math.ceil(height * scale)
            img = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_LINEAR)
            img = torch.FloatTensor(preprocess(img)).to(device)
            offset, prob = pnet(img)
            probs = prob.cpu().data.numpy()[0, 1, :, :]  # probs: probability of a face at each sliding window
            offsets = offset.cpu().data.numpy()  # offsets: transformations to true bounding boxes
            # applying P-Net is equivalent, in some sense, to moving 12x12 window with stride 2
            stride, cell_size = (2,5), (12,44)
            # indices of boxes where there is probably a lp
            # returns a tuple with an array of row idx's, and an array of col idx's:
            inds = np.where(probs > thresholds)

            if inds[0].size == 0:
                boxes = None
            else:
                # transformations of bounding boxes
                tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
                offsets = np.array([tx1, ty1, tx2, ty2])
                score = probs[inds[0], inds[1]]
                # P-Net is applied to scaled images
                # so we need to rescale bounding boxes back
                bounding_box = np.vstack([
                    np.round((stride[1] * inds[1] + 1.0) / scale),
                    np.round((stride[0] * inds[0] + 1.0) / scale),
                    np.round((stride[1] * inds[1] + 1.0 + cell_size[1]) / scale),
                    np.round((stride[0] * inds[0] + 1.0 + cell_size[0]) / scale),
                    score, offsets])
                boxes = bounding_box.T
                keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
                boxes[keep]

            bounding_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        
        if bounding_boxes != []:
            bounding_boxes = np.vstack(bounding_boxes)
            keep = nms(bounding_boxes[:, 0:5], nms_thresholds)
            bounding_boxes = bounding_boxes[keep]
        else:
            bounding_boxes = np.zeros((1,9))
        # use offsets predicted by pnet to transform bounding boxes
        bboxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5],  x1, y1, x2, y2, score

        bboxes[:, 0:4] = np.round(bboxes[:, 0:4])

        return bboxes

def detect_onet(onet, image, bboxes, device, prob_thresholds=0.8):

    # start = time.time()

    size            = (94,24)
    thresholds      = prob_thresholds  # face detection thresholds
    nms_thresholds  = 0.4
    height, width   = image.shape
    image           = np.expand_dims(image, axis=-1)

    num_boxes = len(bboxes)
    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bboxes, width, height)

    img_boxes = np.zeros((num_boxes, 1, size[1], size[0]))

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 1))

        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = \
            image[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # resize
        img_box = cv2.resize(img_box, size, interpolation=cv2.INTER_LINEAR)
        img_boxes[i, :, :, :] = preprocess(img_box)

    img_boxes = torch.FloatTensor(img_boxes).to(device)
    offset, prob = onet(img_boxes)
    offsets = offset.cpu().data.numpy()  # shape [n_boxes, 4]
    probs = prob.cpu().data.numpy()  # shape [n_boxes, 2]

    keep  = np.where(probs[:, 1] > thresholds)[0]
    bboxes = bboxes[keep]
    bboxes[:, 4] = probs[keep, 1].reshape((-1,))  # assign score from stage 2
    offsets = offsets[keep]
    
    bboxes = calibrate_box(bboxes, offsets)
    keep = nms(bboxes, nms_thresholds, mode='min')
    bboxes = bboxes[keep]
    bboxes[:, 0:4] = np.round(bboxes[:, 0:4])
    return bboxes



class PNet(nn.Module):

    def __init__(self, is_train=False):

        super(PNet, self).__init__()
        self.is_train = is_train

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d((2,5), ceil_mode=True)),
            ('conv2', nn.Conv2d(10, 16, (3,5), 1)),
            ('prelu2', nn.PReLU(16)),
            ('conv3', nn.Conv2d(16, 32, (3,5), 1)),
            ('prelu3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)

        if self.is_train is False:
            a = F.softmax(a, dim=1)

        return b, a


class ONet(nn.Module):

    def __init__(self, is_train=False):

        super(ONet, self).__init__()
        self.is_train = is_train

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, 1, 1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1280, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)

        if self.is_train is False:
            a = F.softmax(a, dim=1)

        return b, a

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1), # outsize (ch_out, H, W)
        )
    def forward(self, x):
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate, color=False):
        super(LPRNet, self).__init__()
        self.class_num = class_num
        
        if color:
            in_channels = 3
        else:
            in_channels = 1

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(), # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=256+class_num+128+64, out_channels=self.class_num, kernel_size=(1,1), stride=(1,1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            # keep those outputs
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]
                keep_features.append(x)

        # only those keep features are passed to the avgpool and pow/mean/div
        # those features are the global_context
        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                # to resize to the same size
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                # to resize to the same size
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        # cat all the global context
        x = torch.cat(global_context, 1)
        x = self.container(x)
        # print(x.size())
        # torch.Size([1, CHARS length, 4, 1output length8])

        # take the average of the height
        logits = torch.mean(x, dim=2)
        # print(logits.size())
        # torch.Size([batch_size, CHARS length, output length ])

        return logits
    
# CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
#          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
#          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
#          'U', 'V', 'W', 'X', 'Y', 'Z'
#         ]
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'M', 'V', 'H','-'
        ]

class Rotation_model(nn.Module):
    def __init__(self, img_size, rgb=False, eval=False):
        super().__init__()

        self.softmax  = nn.Softmax(dim=-1)
        self.img_size = img_size

        if rgb:
            in_channels = 3
        else:
            in_channels = 1

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1), # (64,54,54)
            nn.AvgPool2d(kernel_size=2), # (64, 27, 27)
            nn.BatchNorm2d(num_features = 64), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2), # (128,12,12)
            nn.AvgPool2d(kernel_size=2), # (128, 6, 6)
            nn.BatchNorm2d(num_features = 128), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1), # (128,4,4)
            nn.AvgPool2d(kernel_size=2), # (128, 2, 2)
            nn.BatchNorm2d(num_features = 128), 
            nn.ReLU()
        )

        # calculate the 1D feature size
        size = self.outSize(self.img_size, 3, 1, 0)
        size = self.outSize(size, 2, 2, 0)
        size = self.outSize(size, 3, 2, 0)
        size = self.outSize(size, 2, 2, 0)
        size = self.outSize(size, 3, 1, 0)
        size = self.outSize(size, 2, 2, 0)
        size = size**2*128

        self.classifier = nn.Sequential(
            nn.Linear(in_features=size, out_features=64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=64, out_features=4),
            nn.ReLU()
        )
        
    def outSize(self, input_size, kernal, stride, padding):
        # recall the formula (W−F+2P)/S+1
        outsize = math.floor((input_size - kernal + 2 * padding)/stride +1)
        return outsize

    def features_warpper(self, x):
        x = self.features(x)
        return x

    def logits(self, features):
        x = features.view(features.size(0), -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        # x will already resize to 56x56
        if not eval:
            x = self.features_warpper(x)
            x = self.logits(x)
            return x
        else:
            x = self.features(x)
            x = self.logits(x)
            x = self.softmax(x)
            return x

    
if __name__ == "__main__":
    from torchsummary import summary
    import netron
    # lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    # print(lprnet)
    # summary(lprnet, (1,24,94), device="cpu")

    # from torch.utils.tensorboard import SummaryWriter
    # img = torch.randn(1,1,24,94)
    # writer = SummaryWriter('runs/test_1')
    # writer.add_graph(lprnet, img)
    # writer.close()

    # output = lprnet(img)


    rotation_model = Rotation_model(img_size=128, rgb=False)
    img = torch.randn(1,1,128,128)
    torch.onnx.export(rotation_model, img, 'Rotation_model.onnx')

    # show structure
    netron.start('Rotation_model.onnx')

    # print(rotation_model.features_warpper(img).shape)