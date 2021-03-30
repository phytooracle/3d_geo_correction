import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import utm
import json
import csv
import time
import scipy
import sys
import math
import multiprocessing
import argparse
import os
import copy
import glob
from scipy.optimize import lsq_linear
from datetime import datetime

from detecto.core import Model
from detecto import core, utils, visualize

VOXEL_SIZE = 1e-2

def get_plant_location_in_image(images,model_path):
    
    model = core.Model.load(model_path,['plant'])

    top_predictions = model.predict(images)
    
    return top_predictions	

# --------------------------------------------------
def get_plant_locations_no_alignment(pcd,mins,maxs,pass_path,model_path):

    locations=[]
        
    meta_path = glob.glob(pass_path+'/*metadata.json')
    if len(meta_path) == 0:
        print(":: Error: Metadata file missing or cannot open.")
        return locations
    meta_path = meta_path[0]

    width_pcd = maxs[0]-mins[0]
    height_pcd = maxs[1]-mins[1]

    merged_image_path = glob.glob(pass_path+'/*merged_east_west.png')
    if len(merged_image_path) == 0:
        print(":: Error: Merged image file missing or cannot open.")
        return locations
    merged_image_path = merged_image_path[0]

    merged_image = cv2.imread(merged_image_path)
    
    if merged_image is None:
        print(":: Error: Merged image file missing or cannot open.")
        return locations
    
    merged_image = cv2.normalize(merged_image, None, 255,0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    lon_size = merged_image.shape[0]
    lat_size = merged_image.shape[1]
    
    predicted_plants = get_plant_location_in_image([merged_image],model_path)

    for _,cr,sc in predicted_plants:

        coords = cr
        score = sc

        if score is None or score.shape[0] == 0:
            
            print(':: Error: Cannot detect plants. No plants were detected.')
            continue

        for j,s in enumerate(score):
            coord = coords[j]

            if s<0.5:
                # print(':: Error: Detection is discarded due to low score ({0}).'.format(s))
                continue

            x1 = coord[1]
            y1 = coord[0]
            x2 = coord[3]
            y2 = coord[2]

            x = int((x1+x2)/2)
            y = int((y1+y2)/2)

            w = x2-x1
            h = y2-y1

            corner_ratio_lon = (x)/lon_size
            corner_ratio_lat = (y)/lat_size
            
            location_plant = [mins[0]+width_pcd*corner_ratio_lon,mins[1]+height_pcd*corner_ratio_lat]

            locations.append(location_plant)

    print(":: Detected {0} plants.".format(len(locations)))

    return locations

def read_plant_detection_csv(path,scan_date):
    
    dict_plants = {}

    with open(path, mode='r',encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            
            if rows[0] != "date" and rows[0] != "EMPTY":

                p = [float(rows[5]),float(rows[4])]
                p = utm.from_latlon(p[0],p[1])

                if rows[0] in dict_plants:
                    dict_plants[rows[0]].append([p[0],p[1]])
                else:
                    dict_plants[rows[0]] = [[p[0],p[1]]]

    d1 = datetime.strptime(scan_date,"%Y-%m-%d")

    min_diff = sys.maxsize
    min_date = None

    for d in dict_plants:
        d2 = datetime.strptime(d,"%Y-%m-%d")

        diff = abs((d2-d1).days)

        if diff<min_diff:
            min_diff = diff
            min_date = d

    print(":: The closes date from RGB to this scan date is {0}".format(min_date))

    plants = np.array(dict_plants[min_date])

    return plants


def generate_randome_similarity(d,rotation_point):
    
    scale_x_mean = 0.999999469
    scale_x_std = 0.00000005

    theta_mean = 0
    theta_std = 0.05

    t_x_mean = d
    t_x_std = 0.05

    scale_x = np.random.normal(scale_x_mean,scale_x_std)
    scale_y = 1
    theta = np.random.normal(theta_mean,theta_std)
    t_x = np.random.normal(t_x_mean,t_x_std)
    t_y = 0

    T_tr = np.eye(4)
    T_tr[0,3] = t_x
    T_tr[1,3] = t_y

    T_rot = np.eye(4)
    T_rot[0,0] = math.cos(math.radians(theta))
    T_rot[0,1] = -math.sin(math.radians(theta))
    T_rot[1,0] = math.sin(math.radians(theta))
    T_rot[1,1] = math.cos(math.radians(theta))

    T_to_org = np.eye(4)
    T_to_org[0,3] = -rotation_point[0]
    T_to_org[1,3] = -rotation_point[1]

    T_from_org = np.eye(4)
    T_from_org[0,3] = rotation_point[0]
    T_from_org[1,3] = rotation_point[1]

    T_rot = np.matmul(T_from_org,np.matmul(T_rot,T_to_org))
   
    T_sc = np.eye(4)
    T_sc[0,0] = scale_x
    T_sc[1,1] = scale_y

    T = np.matmul(T_tr,np.matmul(T_sc,T_rot))

    return T

def find_closest_plant(l,inside_plants):

    closest_d = sys.maxsize
    closest_i = -1

    for i,p in enumerate(inside_plants):

        distance = math.sqrt((l[0]-p[0])**2+(l[1]-p[1])**2)

        if distance<closest_d:
            closest_d = distance
            closest_i = i
    
    return closest_d,closest_i

def transform_detections_and_measure_error(args):
    
    d = args[0]
    rotation_point = args[1]
    locations = args[2]
    inside_plants = args[3]
    max_z = args[4]

    T = generate_randome_similarity(d,rotation_point)

    col_z = np.array([[max_z]]*locations.shape[0])
    col_h = np.array([[1]]*locations.shape[0])

    locations = np.append(locations, col_z, 1)
    locations = np.append(locations, col_h, 1)

    new_locations = np.matmul(T,locations.T).T

    all_errors = []
    closely_matched = 0

    for l in new_locations:
        d,i = find_closest_plant(l,inside_plants)
        
        if d<0.25:
            closely_matched+=1

        all_errors.append(d)

    error = -closely_matched

    return T,error



def get_best_similarity_for_plant_transformation(d,rotation_point,locations,inside_plants,max_z,cores):
    
    args = []
    for i in range(0,500):
        args.append((d,rotation_point,locations,inside_plants,max_z))

    if cores is None:
        cores = multiprocessing.cpu_count()

    processes = multiprocessing.Pool(cores)
    results = processes.map(transform_detections_and_measure_error,args)
    processes.close()

    min_error = sys.maxsize
    min_T = None

    for T,e in results:

        if e< min_error:
            min_error = e
            min_T = T
    
    print(":: Maximum number of matched plants is: {0}".format(-min_error))
    print(":: The transformation is:")
    print(min_T)

    return min_T
    

def plant_based_transform_no_alignment(args):

    scan_timestamp = args.path.split('/')[-1]    

    pcd_path = glob.glob(args.path+"/*icp_merge_registered.ply")
    
    if pcd_path is None or len(pcd_path) == 0:
        print(":: Could not find icp_merge_registered.ply for {0}".format(scan_timestamp))
    pcd_path = pcd_path[0]

    plants = read_plant_detection_csv(args.plants,args.scandate)
    
    pcd = o3d.io.read_point_cloud(pcd_path,format="ply")

    mins = np.min(np.array(pcd.points),axis=0)
    maxs = np.max(np.array(pcd.points),axis=0)
    
    colors = np.array(pcd.points)
    colors[:,:2] = 0
    
    colors[:,2] = (colors[:,2]-mins[2])/(maxs[2]-mins[2])
    colors[:,1] = colors[:,2]

    np.asarray(pcd.colors)[:, :] = colors

    locations = get_plant_locations_no_alignment(pcd,mins,maxs,args.path,args.model)

    if len(locations) == 0:
        print(":: Error occured while detecting the plants locations in the PNG file.")
        return 
    
    locations = np.array(locations)
    
    inside_plants = np.array([plt for plt in plants if plt[1]>mins[1] and plt[1]<maxs[1]])

    min_detection = np.min(locations,axis=0)
    min_original = np.min(inside_plants,axis=0)

    d = min_original[0]-min_detection[0]
    
    rotation_point = [(mins[0]+maxs[0])/2,(mins[1]+maxs[1])/2]

    cores = None
    if args.core is not None:
        cores = int(args.core)
    
    T = get_best_similarity_for_plant_transformation(d,rotation_point,locations,inside_plants,maxs[2],cores)

    pcd = pcd.transform(T)

    tree = o3d.geometry.KDTreeFlann(pcd)

    for plt in plants:
        if plt[0]>mins[0] and plt[0]<maxs[0] and plt[1]>mins[1] and plt[1]<maxs[1]:
            
            [k, idx, _] = tree.search_knn_vector_3d([plt[0],plt[1],maxs[2]], 10000)
            np.asarray(pcd.colors)[idx[1:], :] = [1, 0, 1]

    output_path = pcd_path.replace(args.path,args.output).replace('_icp_merge_registered.ply','_corrected.ply')
    o3d.io.write_point_cloud(output_path, pcd)

    pcd_down = copy.deepcopy(pcd).voxel_down_sample(1e-2)
    o3d.io.write_point_cloud(output_path.replace('_corrected.ply','_down_sampled.ply'), pcd_down)


def get_args():
    
    parser = argparse.ArgumentParser(
        description='Geo-correction of point cloud data using plant locations.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p',
                        '--path',
                        help='The path to the registered and rotated point cloud file.',
                        metavar='path',
                        required=True)

    parser.add_argument('-m',
                        '--model',
                        help='Path to the detecto model pth file. ',
                        metavar='model',
                        required=True)

    parser.add_argument('-c',
                        '--core',
                        help='Number of cores to use',
                        metavar='core',
                        required=False)

    parser.add_argument('-d',
                        '--plants',
                        help='The path to the full plant detection file.',
                        metavar='plants',
                        required=True)

    parser.add_argument('-s',
                        '--scandate',
                        help='The date of scan, used for picking the closest RGB detection.',
                        metavar='scandate',
                        required=True)

    parser.add_argument('-o',
                        '--output',
                        help='The output directory.',
                        metavar='output',
                        required=True)

    return parser.parse_args()

def main():
    
    args = get_args()

    plant_based_transform_no_alignment(args)


if __name__ == "__main__":
    main()