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

def get_IoU(args):

    img_p = args[0]
    img_h = args[1]
    x = args[2]
    y = args[3]

    img_p = cv2.resize(img_p,(int(0.1*img_p.shape[1]),int(0.1*img_p.shape[0])))
    img_h = cv2.resize(img_h,(img_p.shape[1],img_p.shape[0]))

    intersection = np.sum(cv2.bitwise_and(img_p,img_h))
    union = np.sum(cv2.bitwise_or(img_p,img_h))

    return intersection/union,x,y

class single_pass_cloud:

    def __init__(self, path_to_ply, meta_path, plants_path):
        self.pcd = o3d.io.read_point_cloud(path_to_ply,format="ply")
        self.full_res_pcd = self.pcd
        self.down_sample()

        # save down_sampled for visualization. Comment this later
        o3d.io.write_point_cloud(path_to_ply.replace('merge_registered','merge_registered_down_sampled'), self.pcd)

        self.get_plants(plants_path)

    def down_sample(self):
        print(self.pcd)
        downpcd = self.pcd.voxel_down_sample(voxel_size=1e-2)
        print(downpcd)
        new_coords = np.asarray(downpcd.points)
        self.pcd = o3d.geometry.PointCloud() 
        self.pcd.points = o3d.utility.Vector3dVector(new_coords)

    def get_plants(self,plants_path):

        plants = []
        with open(plants_path, mode='r') as infile:
            reader = csv.reader(infile)
            for rows in reader:
                if rows[0] != "Latitude" and rows[0] != "EMPTY":
                    
                    p = [float(rows[0]),float(rows[1])]
                    p = utm.from_latlon(p[0],p[1])
                    plants.append([p[0],p[1]])
                    
        plants = np.array(plants)
        self.plants = plants

    def generate_plant_mask(self,width,height,scan_min_x,scan_max_x,scan_min_y,scan_max_y,st_x=0,st_y=0):

        plants = []

        for p in self.plants:
            if p[0]>= scan_min_x and p[0]<=scan_max_x and p[1] >= scan_min_y and p[1] <= scan_max_y:
                plants.append([p[0],p[1]])
                  
        plants = np.array(plants)

        # print("Nearby plants: ")
        # print(plants)

        image = np.zeros((width,height,3))

        for p in plants:
            x = int(width*(p[1]-scan_min_y)/(scan_max_y-scan_min_y))
            y = int(height*(p[0]-scan_min_x)/(scan_max_x-scan_min_x))
            cv2.circle(image,(y+st_y,x+st_x),100,(0,255,0),-1)

        image = image.astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image,20,255,cv2.THRESH_BINARY)
        
        return image

    def generate_heatmap_and_mask(self,name="old"):
        
        points = np.asarray(self.pcd.points)
    
        scan_min_x = np.min(points[:,0])
        scan_max_x = np.max(points[:,0])
        scan_min_y = np.min(points[:,1])
        scan_max_y = np.max(points[:,1])
        min_z = np.min(points[:,2])
        max_z = np.max(points[:,2])

        print("Min Max x in pc: ",scan_min_x,scan_max_x)
        print("Min Max y in pc: ",scan_min_y,scan_max_y)
        
        scale = 1000

        width = int((np.max(points[:,1])-np.min(points[:,1])+0.01)*scale)
        height = int((np.max(points[:,0])-np.min(points[:,0])+0.01)*scale)
        
        print("Width and Heihght of the image: ",width,height)
        
        image = np.zeros((width,height,3))

        window_size = 10

        for pt in points:
            
            loc_y = int(height*(pt[0]-scan_min_x)/(scan_max_x-scan_min_x))
            loc_x = int(width*(pt[1]-scan_min_y)/(scan_max_y-scan_min_y))
            
            # image[loc_x-window_size:loc_x+window_size,loc_y-window_size:loc_y+window_size,2] += int(255*(pt[2]-min_z)/(max_z-min_z))
            image[loc_x-window_size:loc_x+window_size,loc_y-window_size:loc_y+window_size,1] += int(127*(pt[2]-min_z)/(max_z-min_z))
            image[loc_x-window_size:loc_x+window_size,loc_y-window_size:loc_y+window_size,0] += int(127*(pt[2]-min_z)/(max_z-min_z))
            image[loc_x-window_size:loc_x+window_size,loc_y-window_size:loc_y+window_size,2] += int(255*(pt[2]-min_z)/(max_z-min_z))

        image = cv2.normalize(image, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        image = image.astype('uint8')

        # cv2.imwrite('{0}_heatmap.png'.format(name),image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # cv2.imwrite('{0}_otsu.png'.format(name),image)

        kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # cv2.imwrite('{0}_mask.png'.format(name),image)
        self.mask_heatmap = image.copy()

        self.mask_plants = self.generate_plant_mask(width,height,scan_min_x,scan_max_x,scan_min_y,scan_max_y)
        # cv2.imwrite('{0}_plants_mask.png'.format('full'),self.mask_plants)

    def geo_correct(self):

        mask_p = self.mask_plants
        mask_h = self.mask_heatmap

        w = mask_p.shape[0]
        h = mask_h.shape[1]
        
        center_x = int(h/2)
        center_y = int(w/2)

        arglist = []
        results = []

        for k in range(int(w/2-200),int(w/2+200),10):
            for c in range(int(h/2-2500),int(h/2+2500),100):
                img_p = mask_p[max(center_y-k,0):w+min(0,center_y-k),max(center_x-c,0):h+min(0,center_x-c)]
                img_h = mask_h[abs(min(0,center_y-k)):min(center_y+k,w),abs(min(0,center_x-c)):min(center_x+c,h)]

                # arglist.append((img_p.copy(),img_h.copy(),c,k))
                results.append(get_IoU((img_p,img_h,c,k)))
                

        # print('Beggining the geocorrection...')

        # processes = multiprocessing.Pool(45)
        # results = processes.map(get_IoU,arglist)
        # processes.close()

        max_iou = 0
        max_c = None
        max_k = None

        for IoU, c, k in results:
            
            if IoU>max_iou:
                max_iou = IoU
                max_c = c
                max_k = k
        
        print("Max c and k and the IoU: ",max_c,max_k,max_iou)

        points = np.asarray(self.pcd.points)
        scan_min_x = np.min(points[:,0])
        scan_max_x = np.max(points[:,0])
        scan_min_y = np.min(points[:,1])
        scan_max_y = np.max(points[:,1])

        img_p = mask_p[max(center_y-max_k,0):w+min(0,center_y-max_k),max(center_x-max_c,0):h+min(0,center_x-max_c)]
        img_h = mask_h[abs(min(0,center_y-max_k)):min(center_y+max_k,w),abs(min(0,center_x-max_c)):min(center_x+max_c,h)]

        # cv2.imwrite('res_p.png',img_p)
        # cv2.imwrite('res_h.png',img_h)

        
        ratio = ((scan_max_x-scan_min_x)/(h),(scan_max_y-scan_min_y)/(w))
        translation_vec = np.array([(center_x-max_c)*ratio[0],(center_y-max_k)*ratio[1],0]).astype('float64')

        print("Translation vector: ",translation_vec)

        self.pcd = self.pcd.translate(translation_vec)
        self.full_res_pcd = self.full_res_pcd.translate(translation_vec)

        
    def save_new_ply_file(self,path,full_res_path):

        o3d.io.write_point_cloud(path, self.pcd)
        o3d.io.write_point_cloud(full_res_path, self.full_res_pcd)

        print('geo-corrected point cloud successfully saved into a new ply file. ')

    def gantry_point_to_latlon(gantry_x, gantry_y):
        
        # -------------- Gantry to UTM --------------

        # Linear transformation coefficients
        ay = 3659974.971; by = 1.0002; cy = 0.0078;
        ax = 409012.2032; bx = 0.009; cx = - 0.9986;

        utm_x = ax + (bx * gantry_x) + (cx * gantry_y)
        utm_y = ay + (by * gantry_x) + (cy * gantry_y)

        # -------------- UTM to Latlon --------------

        # Get UTM information from southeast corner of field
        SE_utm = utm.from_latlon(33.07451869, -111.97477775)
        utm_zone = SE_utm[2]
        utm_num  = SE_utm[3]

        return utm.to_latlon(utm_x, utm_y, utm_zone, utm_num)


def get_args():
    
    parser = argparse.ArgumentParser(
        description='Geo-correction of point cloud data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('pcd',
                        metavar='pcd',
                        help='Merged and registered point cloud')

    parser.add_argument('-o',
                        '--outdir',
                        help='Output directory',
                        metavar='outdir',
                        type=str,
                        default='geocorrect_out')

    parser.add_argument('-m',
                        '--meta_path',
                        help='Metadata path',
                        metavar='meta_path',
                        required=True)

    parser.add_argument('-l',
                        '--plant_loc',
                        help='CSV file containing known locations of plants',
                        metavar='plant_loc',
                        required=True)

    return parser.parse_args()


def main():
    
    args = get_args()

    f_name = os.path.splitext(os.path.basename(args.pcd))[-2] + '_geocorrected.ply'
    out_path = os.path.join(args.outdir, f_name)
    
    full_res_f_name = os.path.splitext(os.path.basename(args.pcd))[-2] + '_geocorrected_full.ply'
    full_res_out_path = os.path.join(args.outdir, full_res_f_name)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    pc = single_pass_cloud(\
    args.pcd,\
    args.meta_path,\
    args.plant_loc)
    
    pc.generate_heatmap_and_mask()

    pc.geo_correct()

    pc.generate_heatmap_and_mask('new')

    pc.save_new_ply_file(out_path,full_res_out_path)
 


if __name__ == "__main__":
    main()

