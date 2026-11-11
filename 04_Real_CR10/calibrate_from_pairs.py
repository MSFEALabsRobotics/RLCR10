#!/usr/bin/env python3
"""Build a candidate optical-to-base rigid transform from stationary-camera 3D correspondences."""
import argparse,json
from pathlib import Path
import numpy as np

def rigid_fit(src,dst):
    src=np.asarray(src,float);dst=np.asarray(dst,float)
    if src.shape!=dst.shape or src.ndim!=2 or src.shape[1]!=3 or len(src)<4:
        raise ValueError('four or more matching optical/base 3D point pairs required')
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        raise ValueError('point coordinates must be finite')
    if np.linalg.matrix_rank(src-src.mean(0))<2:
        raise ValueError('calibration points are collinear; arrange widely spread markers')
    a=src-src.mean(0);b=dst-dst.mean(0)
    u,_,vt=np.linalg.svd(a.T@b)
    rot=vt.T@np.diag([1,1,np.linalg.det(vt.T@u.T)])@u.T
    trans=dst.mean(0)-rot@src.mean(0)
    T=np.eye(4);T[:3,:3]=rot;T[:3,3]=trans
    rmse=float(np.sqrt(np.mean(np.sum(((rot@src.T).T+trans-dst)**2,axis=1))))
    return T,rmse

def main():
    a=argparse.ArgumentParser()
    a.add_argument('pairs',help='JSON: {"camera_optical_m":[[x,y,z],...], "robot_base_m":[[x,y,z],...]}')
    a.add_argument('--baseline-pose-mm-deg',type=float,nargs=6,required=True)
    a.add_argument('--config',default='config/deployment.json')
    args=a.parse_args()
    d=json.loads(Path(args.pairs).read_text())
    T,rmse=rigid_fit(d['camera_optical_m'],d['robot_base_m'])
    cfgpath=Path(args.config);cfg=json.loads(cfgpath.read_text())
    cfg['calibration']['baseline_tcp_pose_mm_deg']=args.baseline_pose_mm_deg
    cfg['calibration']['T_base_camera_at_baseline']=T.tolist()
    cfg['calibration']['approved']=False
    cfgpath.write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps({'candidate_camera_to_robot_transform':T.tolist(),'RMSE_m':rmse,
                      'calibration_approved':False,'note':'Independent physical hold-out validation is mandatory'},indent=2))
    if rmse>0.002:
        print('WARNING: >2 mm RMS; inadequate for a 2 mm-step pushing controller')
if __name__=='__main__':main()
