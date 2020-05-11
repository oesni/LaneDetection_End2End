#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

import os
import shutil
import sys
import time
import cv2
from test import inference_model, test_model

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm

from Networks.utils import get_homography
from Dataloader.Load_Data_new import (get_loader, get_testloader,
                                      load_valid_set_file_all)
# from eval_lane import LaneEval
from Loss_crit import define_loss_crit, polynomial
from Networks.LSQ_layer import Net
from Networks.utils import (AverageMeter, Logger, define_args,
                            define_init_weights, define_optim,
                            define_scheduler, first_run, mkdir_if_missing,
                            save_weightmap)

def undistort(img):
    DIM = (640, 403)
    K = np.array([[322.8344353809592, 0.0, 322.208290306497], [0.0, 322.75932679455855, 213.76296628754383], [0.0, 0.0, 1.0]])
    D = np.array([[-0.05623399470296618], [0.02450383081374598], [-0.02068599777427462], [0.007061197363889161]])

    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img

def callback(msg):
    global cur_msg
    global get_new_img_msg

    # Change timestamp to current time
    msg.header.stamp = rospy.get_rostime()
    msg.header.frame_id = '/camera_front'
    np_arr = np.fromstring(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 403))
    cur_msg = (undistort(img), msg.header)
    get_new_img_msg = True

def initialize_network(model_path):
    global args

    parser = define_args()
    args = parser.parse_args()
    if not args.end_to_end:
        assert args.pretrained == False
    if args.clas:
        assert args.nclasses == 4
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size

    # Check GPU availability
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn

    global model
    model = Net(args)
    # define_init_weights(model, args.weight_init)

    global params
    params = Projections(args)

    if not args.no_cuda:
        # Load model on gpu before passing params to optimizer
        model = model.cuda()

    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location='cuda:0')
        print("=> check point : {0}".format(model_path))
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print("Exception is :")
            print(e)
            raise Exception("Failed to load model - {0}", model_path)
    else:
        raise Exception("Model path is not valid!")

def resize_coordinates(array):
    return array*2.5
    
"""
Call initialize_network first!!
img: cv 3ch image

return result image
"""
def test_image(img):
    global model
    global params

    model.eval()

    # img = Image.open(img_path).convert('RGB')
    img = cv2.resize(img, dsize=(1280, 720))
    img_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    # w, h = img.size
    h, w, _ = img.shape
    print(img.shape)

    img_pil = F.crop(img_pil, h-640, 0, 640, w)
    img_pil = F.resize(img_pil, size=(256, 512), interpolation=Image.BILINEAR)
    input = F.to_tensor(img_pil).float()
    # print("input:")
    # print(type(input))
    # print("input size:")
    # print(input.size())

    print("##################")

    input = torch.Tensor([torch.Tensor.numpy(input)])
    # print("input:")
    # print(type(input))
    # print("input size:")
    # print(input.size())

    # Reset coordinates
    x_cal0, x_cal1, x_cal2, x_cal3 = [None]*4

    # Put inputs on gpu if possible
    if not args.no_cuda:
        input = input.cuda(non_blocking=True).float()

    # Run model
    torch.cuda.synchronize()
    a = time.time()
    start1 = time.time()
    beta0, beta1, beta2, beta3, weightmap_zeros, \
                    output_net, outputs_line, outputs_horizon, output_seg = model(input, gt_line=np.array([1,1]), 
                                                                                  end_to_end=args.end_to_end, gt=None)
    
    torch.cuda.synchronize()
    b = time.time()

    # Horizon task & Line classification task
    if args.clas:
        horizon_pred = nn.Sigmoid()(outputs_horizon).sum(dim=1)
        horizon_pred = (torch.round((resize_coordinates(horizon_pred) + 80)/10)*10).int()
        line_pred = torch.round(nn.Sigmoid()(outputs_line))
    else:
        assert False
    
    # Calculate X coordinates
    x_cal0 = params.compute_coordinates(beta0)
    x_cal1 = params.compute_coordinates(beta1)
    x_cal2 = params.compute_coordinates(beta2)
    x_cal3 = params.compute_coordinates(beta3)
    lanes_pred = torch.stack((x_cal0, x_cal1, x_cal2, x_cal3), dim=1)

    print("DL FPS: {0}".format(1.0/(time.time()-start1)))

    # Check line type branch
    line_pred = line_pred[:, [1, 2, 0, 3]]
    lanes_pred[(1 - line_pred[:, :, None]).byte().expand_as(lanes_pred)] = -2

    # Check horizon branch
    bounds = ((horizon_pred - 160) / 10)
    for k, bound in enumerate(bounds):
        lanes_pred[k, :, :bound.item()] = -2

    # TODO check intersections
    lanes_pred[lanes_pred > 1279] = -2
    lanes_pred[lanes_pred < 0] = -2

    lanes_pred = np.int_(np.round(lanes_pred.data.cpu().numpy())).tolist()
    num_el = input.size(0)

    for j in range(num_el):
        lanes_to_write = lanes_pred[j]
        
        if args.draw_testset:
            test = weightmap_zeros[j]
            weight0= test[0]
            weight1= test[1]
            weight2= test[2]
            weight3= test[3]
            
            # img_name = img_path
            h_samples = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
            colormap = [(255,0,0), (0,255,0), (255,255,0), (0,0,255), (0, 128, 128)]

            # with open(img_name, 'rb') as f:
            #     img = np.array(Image.open(f).convert('RGB'))
            for lane_i in range(len(lanes_to_write)):
                x_orig = lanes_to_write[lane_i]
                pt_or = [(xcord, ycord) for (xcord, ycord) in zip(x_orig, h_samples) if xcord!=-2]
                for point in pt_or:
                    img = cv2.circle(img, tuple(np.int32(point)), thickness=-1, color=colormap[lane_i], radius = 3)
            # img = Image.fromarray(np.uint8(img))
            # img.show()
            return img

class Projections():
    '''
    Compute coordintes after backprojection to original perspective
    '''
    def __init__(self, options):
        super(Projections, self).__init__()
        M, M_inv = get_homography(resize=options.resize, no_mapping=False)
        self.M, self.M_inv = torch.from_numpy(M), torch.from_numpy(M_inv)
        start = 160
        delta = 10
        num_heights = (720-start)//delta
        self.y_d = (torch.arange(start,720,delta)-80).double() / 2.5
        self.ones = torch.ones(num_heights).double()
        self.y_prime = (self.M[1,1:2]*self.y_d + self.M[1,2:])/(self.M[2,1:2]*self.y_d+self.M[2,2:])
        self.y_eval = 255 - self.y_prime
        self.Y = torch.stack((self.y_eval**2, self.y_eval, self.ones), 1)

        if options.order == 0:
            self.Y = self.tensor_ones
        elif options.order == 1:
            self.Y = torch.stack((self.y_eval, self.ones), 1)
        elif options.order == 2:
            self.Y = torch.stack((self.y_eval**2, self.y_eval, self.ones), 1)
        elif options.order == 3:
            self.Y = torch.stack((self.y_eval**3, self.y_eval**2, self.y_eval, self.ones), 1)
        else:
            raise NotImplementedError(
                    'Requested order {} for polynomial fit is not implemented'.format(options.order))
        self.Y = self.Y.unsqueeze(0).repeat(options.batch_size, 1, 1)
        self.ones = torch.ones(options.batch_size, num_heights, 1).double()
        self.y_prime = self.y_prime.unsqueeze(0).repeat(options.batch_size, 1).unsqueeze(2)
        self.M_inv = self.M_inv.unsqueeze(0).repeat(options.batch_size, 1, 1)

        # use gpu
        self.M = self.M.cuda()
        self.M_inv = self.M_inv.cuda()
        self.y_prime = self.y_prime.cuda()
        self.Y = self.Y.cuda()
        self.ones = self.ones.cuda()

    def compute_coordinates(self, params):
        # Sample at y_d in the homography space
        bs = params.size(0)
        x_prime = torch.bmm(self.Y[:bs], params)

        # Transform sampled points back
        coordinates = torch.stack((x_prime, self.y_prime[:bs], self.ones[:bs]), 2).squeeze(3).permute((0, 2, 1))
        trans = torch.bmm(self.M_inv[:bs], coordinates)
        x_cal = trans[:,0,:]/trans[:,2,:]
        # y_cal = trans[:,1,:]/trans[:,2,:] # sanity check

        # Rezize
        x_cal = resize_coordinates(x_cal)

        return x_cal

if __name__ == '__main__':
    # initialize_network("/home/inseo/Desktop/lane_project/LaneDetection_End2End/Backprojection_Loss/weights/order1.pth_0421.tar")
    initialize_network("/home/inseo/Desktop/lane_project/LaneDetection_End2End/Backprojection_Loss/weights/transfer_0421.pth.tar")

    try:
        rospy.init_node('camera_node')
        rospy.Subscriber('/gmsl_camera/port_0/cam_0/image_raw/compressed', CompressedImage, callback)

        cur_msg = None
        get_new_img_msg = False

        while not rospy.is_shutdown():
            if get_new_img_msg:
                start = time.time()

                show_frame = cur_msg[0]

                show_frame = test_image(show_frame)
                
                print("imshow")
                cv2.imshow("frame", show_frame)
                print("imshow end")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                print("FPS: {0}".format(1.0/(time.time()-start)))

                get_new_img_msg = False

    except rospy.ROSInterruptException:
        rospy.logfatal("{camera_node} is dead.")
