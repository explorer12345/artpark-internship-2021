import os
import sys
import yaml
import g2o
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/two_heads'))
from utils import *
from infer import *

def load_lidar_poses(pose_file, calib_file):
    '''Load and return calibrated lidar poses relative to initial lidar frame
    Input: pose_file = file containing camera frame poses relative to initial camera frame
           calib_file = file containing matrix data to transform camera frame to lidar frame '''
    # load poses from either the dataset or SLAM or odometry methods
    poses = load_poses(pose_file)
    inv_pose_0 = np.linalg.inv(poses[0])

    # load calibration parameter from the dataset
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # convert the poses from camera coordinate system to lidar coordinate system
    calib_poses = []
    for pose in poses:
        calib_poses.append(T_velo_cam.dot(inv_pose_0).dot(pose).dot(T_cam_velo))
    return np.asarray(calib_poses)

def data_stream(pose_file, calib_file, covariance_file):
    ''' yields index, trajectory from initial position to position corresponding to the index, covariance matrix 
        corresponding to the index
        '''
    ldr_poses = load_lidar_poses(pose_file, calib_file)
    covs = open(covariance_file)
    covs = [overlap.replace('\n', '').split() for overlap in covs.readlines()]
    covs = np.asarray(covs, dtype=float)

    for i in range(ldr_poses.shape[0]):
        # traj = ldr_poses[:i+1,:2,3]
        if i != 0:
            cov = covs[i-1].reshape((6,6))
            yield i, ldr_poses[i], cov
        else:
            yield i, ldr_poses[i], ldr_poses.shape[0]

def get_cov_ellipse(cov, center, nstd=3, **kwargs):
    '''Returns a matplotlib Ellipse patch representing the covariance matrix
       cov centred at centre and scaled by the factor nstd (n times the standard deviation).
    '''
    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
    theta = np.arctan2(vy, vx)
    
    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals[:2])
    return Ellipse(xy=center, width=width, height=height, angle=np.degrees(theta), **kwargs)

def predict_overlap_yaw(idx, traj, traj_length, ellipse, overlap_inference_class, inactive_time_thres = 100, inactive_dist_thres = 50,
                        overlap_thres = 0.47):
    """Calculate overlap and yaw prediction of the vertex which is within the pose uncertainty and for which 
    current vertex(idx) has a maximum overlap with it
    """
    # add corresponding feature volume
    overlap_inference_class.infer_multiple(idx, [])

    # search only in previous frames and inactive map
    if idx < inactive_time_thres:
        # overlap_inference_class.infer_multiple(idx, [])
        return None, None, None
    
    indices = np.arange(idx - inactive_time_thres)
    
    dist_delta = traj_length[idx] - np.array(traj_length)[indices]
    indices = indices[dist_delta > inactive_dist_thres]
    
    if len(indices) < 0:
        # overlap_inference_class.infer_multiple(idx, [])
        return None, None, None
    
    # check whether the prediction is in the covarinace matrix defined search space or not
    angle = ellipse.angle
    width = ellipse.width
    height = ellipse.height
    
    cos_angle = np.cos(np.radians(180. - angle))
    sin_angle = np.sin(np.radians(180. - angle))
    
    # print('indices before:')
    # print(indices)
    xc = np.asarray(traj)[idx, 0] - np.asarray(traj)[indices, 0]
    yc = np.asarray(traj)[idx, 1] - np.asarray(traj)[indices, 1]
    
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle
    rad_cc = (xct ** 2 / (width / 2.) ** 2) + (yct ** 2 / (height / 2.) ** 2)
    
    reference_idx = indices[rad_cc < 1]

    print('reference idx:')
    print(reference_idx)
    print(len(reference_idx))
    
    if len(reference_idx) > 0:
        overlaps, yaws = overlap_inference_class.infer_multiple(idx, reference_idx)
        # overlaps, yaws = overlap_inference_class.infer_multiple(idx, [99])
        if np.max(overlaps) > overlap_thres:
            arg_max_overlap = np.argmax(overlaps)
            return int(reference_idx[arg_max_overlap]), overlaps[arg_max_overlap], yaws[arg_max_overlap]
        else:
            # overlap_inference_class.infer_multiple(idx, [])
            return None, None, None
    else:
        # overlap_inference_class.infer_multiple(idx, [])
        return None, None, None   

if __name__ == '__main__':
    config_filename = 'config/demo.yml'
  
    if len(sys.argv) > 1:
        config_filename = sys.argv[1]
    
    # load the configuration file
    config = yaml.load(open(config_filename))
    
    # set the file related parameters
    covariance_file = config['Graphslam']['covariance_file']
    poses_file = config['Graphslam']['poses_file']
    calib_file = config['Graphslam']['calib_file']
    # scan_folder = config['Graphslam']['scan_folder']
    network_file = config['Graphslam']['network_config'] 
    config_overlap = yaml.load(open(network_file))

    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # some other parameters
    inactive_time_thres = 100
    inactive_dist_thres = 50
    overlap_thres = 0.3
    overlap_inference_class = Infer(config_overlap)

    # set up data stream
    data_stream_gen = data_stream(poses_file, calib_file, covariance_file)
    (idx, calib_pose_0, last_idx) = next(data_stream_gen)
    # print('last_idx =', last_idx)

    # initilize graph with the initial pose as a fixed vertex
    optimizer = PoseGraphOptimization()
    rk = g2o.RobustKernelDCS()
    optimizer.add_vertex(0, g2o.Isometry3d(calib_pose_0), True)
    traj, traj_length = [[0,0]], [0]
    idx += 1
    prev_pose = np.eye(4)
    xy_coords = [[0,0]]
    max_overlap = 0.0
    
    while idx != last_idx:
        print('started connecting vertex %d ...' %idx)
        (idx, calib_pose_idx, cov_idx) = next(data_stream_gen)
        xy_coords.append(calib_pose_idx[:2,3])

        # updating trajectory
        traj.append(calib_pose_idx[:2,3])
        delta_dist = np.linalg.norm(traj[idx]-traj[idx-1])
        traj_length.append(traj_length[-1] + delta_dist)

        # updating the graph
        optimizer.add_vertex(idx, g2o.Isometry3d(calib_pose_idx))
        info_matrix = np.linalg.inv(cov_idx)
        inv_prev_pose = np.linalg.inv(prev_pose)
        pose_diff = T_velo_cam.dot(inv_prev_pose).dot(calib_pose_idx).dot(T_cam_velo)
        # print('posediff=', pose_diff, type(pose_diff))
        # print('g2o.isometry3d(posediff)=', g2o.Isometry3d(pose_diff), type(g2o.Isometry3d(pose_diff)) )
        # print('infomatrix=', info_matrix, type(info_matrix))
        # print('[idx-1, idx]',[idx-1, idx] , type([idx-1, idx]))
        optimizer.add_edge([idx-1, idx], g2o.Isometry3d(pose_diff), info_matrix, robust_kernel=rk)

        #  using overlapnet predictions to check for loop closure
        ellipse = get_cov_ellipse(cov_idx, traj[idx])
        lc_idx, overlap_idx, yaw_idx = predict_overlap_yaw(idx, traj, traj_length, ellipse, overlap_inference_class, inactive_time_thres = 100, inactive_dist_thres = 50, overlap_thres = 0.3)

        # print('traj:', traj, end='\n')
        # adding loop closure edges to the graph
        if lc_idx is not None:
            yaw = yaw_idx*np.pi/180.0
            print('YAW=', yaw, type(yaw))
            print('-------------')
            lc_diff = -(np.array(traj[lc_idx]) - np.array(traj[idx]))
            print('lc_diff=',lc_diff, type(lc_diff))
            transform = np.asarray([[np.cos(yaw), np.sin(yaw), 0, lc_diff[0]],
                                    [-1*np.sin(yaw), np.cos(yaw), 0, lc_diff[1]],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
            print('TRANSFORM', transform, type(transform))
            print('g2oisom33d(transform)', g2o.Isometry3d(transform), type(g2o.Isometry3d(transform)))
            print('-------------')
            info_matrix = np.linalg.inv(overlap_idx*np.eye(6))
            print('infomatrix=', info_matrix, type(info_matrix))
            print('[idx, lc_idx]', [idx, lc_idx], type([idx, lc_idx]))
            optimizer.add_edge([idx, lc_idx], g2o.Isometry3d(transform), info_matrix, robust_kernel=rk)

            if overlap_idx > max_overlap:
                max_overlap = overlap_idx
                max_overlap_pair = [idx, lc_idx]

        prev_pose = calib_pose_idx.copy()
        idx += 1

    optimizer.optimize()
    poses_optim = np.asarray([optimizer.get_pose(i).translation().T for i in range(last_idx)])

    xy_coords = np.asarray(xy_coords)
    xy_coords_optim = poses_optim[:,:2]
    
    # calculating mean squared error
    del_xy = np.sum(np.sum((xy_coords-xy_coords_optim)**2, axis=1)**0.5)/xy_coords.shape[0]
    print('del_xy=', del_xy)
    print('max overlap=', max_overlap)
    print('max overlap pair:', max_overlap_pair)

    # compare by plotting trajectory obtained purely from odometry data and one obtained after optimizing the graph
    plt.plot(xy_coords[:,0], xy_coords[:,1], 'r-')
    plt.plot(xy_coords_optim[:,0], xy_coords_optim[:,1], 'b-')
    plt.show()