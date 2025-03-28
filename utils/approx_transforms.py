#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:24:02 2023

@author: ghosh37
"""
import os
import cv2
import numpy as np
# from ..builder import PIPELINES

        
def update_approx_args(pipeline, args):    
    pipeline[1]['img_scale'] = (args['max_img_dim'], args['min_img_dim'])
    transforms = pipeline[1]['transforms']
    for i, tf in enumerate(transforms):
        for key in tf.keys():
            if key == 'refresh_interval':
                tf[key] = args['refresh_interval']
            elif key == 'quality':
                tf[key] = args['quality']
            elif key == 'cloud':
                tf[key] = args['cloud']
            elif key == 'img_size':
                tf[key] = args['max_img_dim']
        transforms[i] = tf
    
    pipeline[1]['transforms'] = transforms
    # print("\nUpdated pipeline")
    return pipeline


class MemoryApprox(object):
    """
    Use already created error mask and  perform bitwise xor
    Args:
        :param refresh_interval: DRAM refresh interval to select specific error mask
        :param mask_dir: path to error mask
        :param img_size: size of image, used to get mask of same size as image
        :return: image injected with DRAM error in PIL format
    """

    def __init__(self, refresh_interval: int, mask_dir: str, img_size):
        self.refresh_interval = {'rgb':int(refresh_interval[0]),
                                     'depth':int(refresh_interval[1])
                                     }
        self.error_mask = {'rgb': None, 'depth': None}
        
        # convert to single value if tuple provided, assumption is that the width and height are the same
        if isinstance(img_size, tuple):
            img_size = img_size[0]
            
        if refresh_interval[0] != 1 or refresh_interval[1] != 1:
            # path to dram error mask which when XOR'ed with image will give error injected image            
            err_mask_root = {}
            err_mask_path = {}
            for key in ['rgb', 'depth']:
                if self.refresh_interval[key] != 1:
                    err_mask_root[key] = os.path.join(mask_dir, 'error_mask/mask_dram1_refint_{}_fm_2'.format(self.refresh_interval[key]))
                    err_mask_path[key] = os.path.join(err_mask_root[key], 'error_mask_ri{}_{}'.format(self.refresh_interval[key], img_size))
                    if not os.path.isfile(err_mask_path[key]):
                        err_mask_path[key] = os.path.join(err_mask_root[key], 'error_mask_ri{}_{}'.format(self.refresh_interval[key], 640))
                    # open error mask in image format and store in memory as this will be used for all images in dataset
                    self.error_mask[key] = cv2.imread(err_mask_path[key])                                           
            
                    assert self.error_mask is not None, "Error mask not found"

    def __call__(self, sample):
        for key in sample['inputs']:
            # print("Key = ", key, ", Image shape = ", sample[key].shape, ", Image size = ", sample[key].size, ", Mask shape = ", self.error_mask[key].shape)
            sample[key] = self.transform_input(sample[key], self.refresh_interval[key], self.error_mask[key])
        return sample
    
    def transform_input(self, image, ri, mask):
        if ri != 1:
            # find maximum side of image and crop the mask to fit image size
            # h, w = image.shape[:2]
            # cropped_mask = mask[:h, :w, :]
            
            ## Arghadip:
            h, w = image.shape[:2]
            if (len(image.shape) == 3): # RGB
                c = image.shape[2]  # Number of channels
            else:                       # Depth
                c = 1               # Single channel
            cropped_mask = mask[:h, :w, :c]     # Crop the mask with proper channel as well
            # print('Cropped mask datatype: ', cropped_mask.dtype)
            # print('Cropped mask shape: ', cropped_mask.shape)
            
            bit_flips = np.count_nonzero(cropped_mask)
            error = bit_flips/(image.size*8)
            # print("Image size = ", image.size, ", Mask size = ", mask.size, ", Cropped mask size = ", cropped_mask.size)
            
            assert image.size == cropped_mask.size, "Image size and Error mask size differs"
            # Perform bitwise xor of original resized image (assume coming from SENSOR directly)
            # TODO: VERIFY IF CROP CAN BE DONE SEPARATELY AFTER DRAM ERROR
            # print(image, cropped_mask)
            # print(type(image), type(cropped_mask))

            ## Arghadip: XOR only supports Boolean or integer. That's why is necessary to pass it as integer
            ## DBG:
            # print('Inside Mem Approx: image data type = ', image.dtype)
            image = np.bitwise_xor(image.astype(image.dtype), np.squeeze(cropped_mask).astype(image.dtype)) # Arghadip: squeeze if there is any singleton dimension (needed for depth images) # Arghadip: the astype for image is not needed, but kept anyway, need to uncomment the following line if any datatype other than int is used
            # image = np.bitwise_xor(image.astype(int), np.squeeze(cropped_mask).astype(int))
            ## DBG:
            # print('Image dtype after memory approx: ', image.dtype)
            # print("bit flips/size is error:  {}/{} = {}".format(bit_flips, image.size*8, error))        
        return image


class CommunicationApprox(object):
    """
    Compress image and store in memory to send to server directly
    Args:
        :param quality: JPEG image compression quality
        :return: compressed image
    """
    def __init__(self, quality: int, cloud: bool):
        # print(quality)
        self.quality = {'rgb':quality[0] if quality[0] != 120 else None, 'depth': quality[1] if quality[1] != 120 else None}
        # print(self.quality)
        self.compress = cloud
        self.compress_size = {'rgb': 0, 'depth': 0, 'total': 0}
        self.encode_param = {'rgb': None, 'depth': None}
        # self.size = 0
        self.count = {'rgb': 0, 'depth': 0}
        
        
        for key in ['rgb', 'depth']:
            if self.quality[key] != None:
                print('Setting encode param')
                self.encode_param[key] = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality[key]]
        # print('encode param', self.encode_param)
        
    def __call__(self, sample):
        # print(sample['depth'].size)
        # Not needed in Approx Compute
        # print(len(sample['rgb'].tobytes()))
        if self.compress:
            for key in sample['inputs']:
                sample[key], size = self.transform_input(sample[key], self.quality[key], self.encode_param[key], key)
                # print("Inside comm approx: sample datatype after transform = ", sample[key].dtype, " | size = ", size)
                
                ## DBG:
                # print("Size = ", size)
                self.count[key] += 1
                # self.size += size
                self.compress_size[key] += size
                self.compress_size['total'] += size
                ## DBG:
                # print("Size = ", self.compress_size)
                if self.count[key] == 654:
                    print(key, size, self.count[key], self.compress_size[key], self.compress_size['total'])
            
                
        return sample
        
    def transform_input(self, image, quality, encode_param, key):
        if quality == None or quality == 120:
            # if key == 'depth':
            #     # if len(img_arr.shape) == 2:  # grayscale                    
            #     # _, img = cv2.imencode('.png', image)    # Arghadip: commented out as conversion is not needed
            #     size = len(img.tobytes())
            #     # image = cv2.imdecode(img, cv2.IMREAD_COLOR)

            #     ## Arghadip: commented the below lines to regain lost quality, shape modification is done in dataset.py
            #     # image = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
            #     # image = np.tile(image, [3, 1, 1]).transpose(1, 2, 0)    # Arghadip: why transpose? Since it's done here, should I comment the same thing in the dataset.py?
            # else:
            #     size = len(image.tobytes())
            # print(key, size)
            size = len(image.tobytes())
            
        else:                        
            #encode param image quality 0 to 100. default:95
            #if you want to shrink data size, choose low image quality.                    
            result, encimg = cv2.imencode('.jpg', image, encode_param)

            ## DBG:
            ## Check dtype here

            if False == result:
                print("could not encode image!")
                quit()
            # print(key, len(encimg.tobytes()))
            size = len(encimg.tobytes())
            # print(key, size)
            # print(self.compress_size)
            #decode from jpeg format
            image = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
            # print('decoded image', key)
        return image, size
    