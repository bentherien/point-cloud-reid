import torch
import pytorch3d.transforms as transforms3d

from pytorch3d.structures.pointclouds import Pointclouds
from mmdet3d.core import DepthInstance3DBoxes


def get_affine_torch(rotation,translation,device,angle_type='axis_angle',inverse=True):
    assert len(rotation.shape) == 2, f'rotation must have batch dimension, got {rotation.shape}'
    assert len(translation.shape) == 2, f'translation must have batch dimension, got {translation.shape}'
    b = rotation.size(0)
    out = torch.zeros((b,4,4,), dtype=torch.float32, device=device)
    if angle_type == 'axis_angle':
        rot = transforms3d.axis_angle_to_matrix(rotation)
    elif angle_type == 'quaternion':
        rot = transforms3d.quaternion_to_matrix(rotation)
    else:
        raise NotImplemented(f'not implemented for angle_type: {angle_type}')
        
    out[:,:3,:3] = rot
    out[:,[0,1,2],3] = translation
    out[:,3,3] = 1.0 # homogeneous coords
    
    if inverse == False:
        return out
    else:
        return torch.linalg.inv(out)



def interpolate_per_frame(bboxes,pts,device):
    """Interpolates bounding boxes over sweeps and ls"""
    assert bboxes.dtype == torch.float32
    assert pts.dtype == torch.float32
    assert bboxes.size(1) == 7
    s_num = 1
    box_num = bboxes.size(0)
    b = s_num * box_num
    
    # interp_bboxes = bboxes[None,:,:7].repeat(s_num,1,1)
    # assert interp_bboxes.shape == torch.Size((s_num,box_num,7,))

    #get points in boxes
    pc_batched = Pointclouds([pts[:,:3].to(device)])
    pc_padded = pc_batched.points_padded()
    lboxes = DepthInstance3DBoxes(bboxes,origin=(0.5,0.5,0.5))
    pib_idx = lboxes.points_in_boxes(pc_padded).bool().unsqueeze(0)
    
    points_in_boxes,lengths = [],[]
    for i in range(box_num):
        #for one box, get points from each sweep
        pib_temp = [pc_padded[s,pib_idx[s,:,i],:] for s in range(s_num)]
        points_in_boxes.extend(pib_temp)
        lengths.extend([x.size(0) for x in pib_temp])
        

    # print([x.shape for x in points_in_boxes])
    #create large batch
    points_batch = Pointclouds(points_in_boxes).points_padded()
    
    #create affine matrices to center points
    rot = torch.zeros((b,3),device=device)
    rot[:,2] = bboxes[:,6] 
    interp_affines = get_affine_torch(translation=bboxes[:,:3],
                                      rotation=-rot,
                                      device=device,
                                      angle_type='axis_angle',
                                      inverse=True)
    
    #pad points to homogeneous coordinates
    points_batch = torch.cat(
        [points_batch,torch.ones(points_batch.shape[:2]+(1,),device=device)],dim=2
    )
    centered = torch.bmm(interp_affines,points_batch.permute(0,2,1)).permute(0,2,1)
    centered = centered[:,:,:3]
    return centered.reshape(s_num,box_num,-1,3), torch.tensor(lengths,device=device).reshape(s_num,box_num)
    


def get_input_batch(centered,lengths,subsample_number,device):
    new_size = list(centered.shape)
    new_size[2] = subsample_number
    
    temp = [torch.randint(high=x, size=(1,subsample_number,))  if x != 0 else torch.zeros((1,subsample_number)) \
            for x in lengths.reshape(-1)]
    temp = torch.cat(temp).reshape(new_size[:2]+[-1]).long()
    
    where_pts = torch.where(lengths > 0)
    out = torch.zeros(new_size,dtype=torch.float32,device=device)
    out[where_pts[0],where_pts[1],:,:] = centered[
                                                      where_pts[0],
                                                      where_pts[1],
                                                      temp[where_pts[0],where_pts[1],:].t(),
                                                      :
                                                  ].permute(1,0,2)
    return out















import torch
import pytorch3d

import pytorch3d.transforms as tf3d
import numpy as np
import torch.nn.functional as F

from torchvision.transforms import ToTensor
from nuscenes.utils.geometry_utils import BoxVisibility

from functools import reduce





#image funcs



def boxes_in_image(corners2d,corners3d,imsize,vis_level):
    visible =  reduce(torch.logical_and,[
       0 <= corners2d[...,0],  corners2d[...,0] <= imsize[0],
       0 <= corners2d[...,1],  corners2d[...,1] <= imsize[1],
    ])
    
    in_front = corners3d[...,2] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if vis_level == BoxVisibility.ALL:
        torch.logical_and( visible.all(1), in_front.all(1))
    elif vis_level == BoxVisibility.ANY:
        return torch.logical_and(visible.any(1), in_front.any(1))
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError("vis_level: {} not valid".format(vis_level))

def extract_bboxes(img, bboxes, device, output_size=(224, 224)):
    # Create a grid of size (N, H_out, W_out, 2)
    N, C, H, W = img.shape
    H_out, W_out = output_size
    grid = torch.zeros((N, H_out, W_out, 2), dtype=torch.float32, device=device)

    # Normalize the bounding box coordinates to the range [-1, 1]
    bboxes[:, :2] = 2 * bboxes[:, :2] / torch.tensor([W, H], dtype=torch.float32, device=device) - 1
    bboxes[:, 2:] = 2 * bboxes[:, 2:] / torch.tensor([W, H], dtype=torch.float32, device=device) - 1

    # Create the grid of sampling locations
    for i in range(N):
        x1, y1, x2, y2 = bboxes[i]
        grid[i, :, :, 0] = torch.linspace(x1, x2, W_out, device=device).view(1, 1, -1)
        grid[i, :, :, 1] = torch.linspace(y1, y2, H_out, device=device).view(1, -1, 1)
    
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros')

def get_corners_torch(box,rot, device):
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    w, l, h = box[:,3:4], box[:,4:5], box[:,5:6]

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * torch.tensor([1,  1,  1,  1, -1, -1, -1, -1], device=device).unsqueeze(0)
    y_corners = w / 2 * torch.tensor([1, -1, -1,  1,  1, -1, -1,  1], device=device).unsqueeze(0)
    z_corners = h / 2 * torch.tensor([1,  1, -1, -1,  1,  1, -1, -1], device=device).unsqueeze(0)
    
    corners = torch.cat((x_corners.unsqueeze(1), y_corners.unsqueeze(1), z_corners.unsqueeze(1)),dim=1)
    
    rotation_matrix = pytorch3d.transforms.quaternion_to_matrix(rot)
    corners = torch.bmm(rotation_matrix,corners)
    corners = corners + box[:,:3].unsqueeze(2) # (b, 3, 8) = (b, 3, 8) + (b, 3, 1)
    
    return corners

def get_image_crops_batch(img,
                          l2c,
                          ci,
                          boxes_3d,
                          device,
                          imsize=(1600,900),
                          output_size=(225, 225),
                          visibility=BoxVisibility.ANY):
    num_boxes = boxes_3d.size(0)
    axis_angle = torch.cat([torch.zeros(num_boxes,2).to(device),-boxes_3d[:,6:7] - torch.pi/2],dim=1)
    yaw_quat = pytorch3d.transforms.axis_angle_to_quaternion(axis_angle)
    
    l2c_quat = pytorch3d.transforms.matrix_to_quaternion(l2c[:3,:3])
    
    
    box_temp = boxes_3d.clone()
    box_temp[:,:3] = (torch.cat([box_temp[:,:3],torch.ones([num_boxes,1]).to(device)],dim=1) @ l2c.T)[:,:3]
    mul_quat = pytorch3d.transforms.quaternion_multiply(l2c_quat,yaw_quat)

    corners = get_corners_torch(box_temp,mul_quat,device=device).permute(0,2,1)
    
    
    corners_im = ci @ torch.cat([corners.reshape(-1,3),torch.ones([8*num_boxes,1]).to(device)],dim=1).T
    corners_im = corners_im.T[:,:3] / corners_im.T[:,2:3]
    corners_im = corners_im.reshape(num_boxes,8,3)[:,:,:2]
    
    inim = boxes_in_image(corners2d=corners_im,
                          corners3d=corners,
                          imsize=imsize,
                          vis_level=visibility)
    
    idx = torch.where(inim)[0]
    max_ = corners_im[inim].max(1).values
    min_ = corners_im[inim].min(1).values
    
    
    max_[imsize[0] <= max_[:,0] , 0] = imsize[0]
    max_[imsize[1] <= max_[:,1] , 1] = imsize[1]
    min_[min_[:,0] <= 0 , 0] = 0
    min_[min_[:,1] <= 0 , 1] = 0
    
    box2d = torch.cat([min_,max_],dim=1)
    crops = extract_bboxes(img.unsqueeze(0).repeat(max_.size(0),1,1,1), 
                           box2d,
                           device=device,
                           output_size=output_size)
    
    return box2d, idx, crops


def get_crops_per_image(images,
                        ci_list,
                        l2c_list,
                        boxes_3d,
                        device,
                        imsize=(1600,900,),
                        output_size=(224,224,),
                        visibility=BoxVisibility.ANY):
    
    all_idx = []
    all_crops = []
    all_box2d = []
    for i in range(images.size(0)):
        l2c = torch.tensor(l2c_list[i]).to(device)
        ci = torch.tensor(ci_list[i]).to(device)
        box2d, idx, crops = get_image_crops_batch(img=images[i,...],
                                                  l2c=l2c,
                                                  ci=ci,
                                                  boxes_3d=boxes_3d,
                                                  device=device,
                                                  imsize=imsize,
                                                  output_size=output_size,
                                                  visibility=visibility)
        all_idx.append(idx)
        all_crops.append(crops)
        all_box2d.append(box2d)
        
    all_crops, all_idx, all_box2d = torch.cat(all_crops), torch.cat(all_idx), torch.cat(all_box2d)
    unique, counts = torch.unique(all_idx,return_counts=True)

    #assert there is one bbox per crop
    if unique.size(0) < boxes_3d.size(0):
        print("[Waterning in get_crops_per_image] There were fewer unique crops ({}) than boxes ({}). This is probably because of the visibility parameter.".format(unique.size(0),boxes_3d.size(0)))


    before_shape = all_idx.shape# print(,counts)
    duplicates = torch.where(counts > 1)[0]
    mask = torch.zeros_like(all_idx)
    keep_list = []
    # idx_keep_list = []
    for i in duplicates:
        idx = unique[i]
        u_pos = torch.where(all_idx == idx)[0]
        mask[u_pos] = 1
        u_box = all_box2d[u_pos]
        areas = (u_box[:,3]-u_box[:,1]) * (u_box[:,2]-u_box[:,0])
        keep_idx = torch.argmax(areas,0)
        if keep_idx.nelement() > 1:
            print(areas)
            print(keep_idx)
            raise ValueError('keep_idx should be a single value')
        keep_list.append(all_crops[u_pos[keep_idx],...].unsqueeze(0))
        # idx_keep_list.append(all_idx[u_pos[keep_idx]].unsqueeze(0))
        
    if len(keep_list) > 0:
        all_crops = torch.cat([torch.cat(keep_list),all_crops[mask == 0] ])
        all_idx = torch.cat([unique[duplicates],all_idx[mask == 0]])
    else:
        all_crops = all_crops[mask == 0]
        all_idx = all_idx[mask == 0]
    
    assert before_shape[0] == all_idx.shape[0] + counts[duplicates].sum() - duplicates.size(0)

    assert all_idx.size(0) <= boxes_3d.size(0)
    return all_crops, all_idx