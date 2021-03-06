# 3D Point Cloud Geo-Correction
This is a repository that includes all the codes to geo_correct 3d point clouds generated by the gantry 3d laser scanner. Before running this repository one should merge east and west point clouds using [https://github.com/phytooracle/3d_icp_point_cloud_registration] repository and then properly rotate and register the merged ply file using [https://github.com/phytooracle/3d_geo_registration] repo. The docker file can be used to create a container image (or a singularity image) and use it to run (exec) the geo_correct_point_cloud.py script. 

## Inputs

* The merged and registered point cloud as a ply file.
* The metadata json file.
* The csv file that contains the known locations of the plants.

## Outputs

* The down sampled point cloud before geo-correction
* The full resolution geo-corrected point cloud 
* The down sampled geo-corrected point cloud

## Arguments

* Positional Arguments:
    * point cloud file: pcd
* Required Arguments:
    * path to the meta data json file: meta_path (-m --meta_path)
    * path to the csv file containing known locations of the plants: plant_loc (-l --plant_loc)
* Optional Arguments:
    * output directory: outdir (-o --outdir)
    * is detected csv file: use_detected_plants (-d --use_detected_plants) which determines whether or not the csv file contains the coordinates of the detected plants using Faster-RCNN model. 
    * Pixel value for the prior in the x direction when the scan direction is 0. Default is 13398 for early season 10. (-x --x_prior_zero_dir)
    * Pixel value for the prior in the y direction when the scan direction is 0. Default is 459 for early season 10. (-y --y_prior_zero_dir)
    * Pixel value for the prior in the x direction when the scan direction is 1. Default is 10491 for early season 10. (-k --x_prior_one_dir)
    * Pixel value for the prior in the y direction when the scan direction is 1. Default is 529 for early season 10. (-n --y_prior_one_dir)


## Running the Script

* Docker:
    * Docker run \[-v /mount_this:/to_this\] image_name python3 geo_correct_point_cloud.py \[params\]
* Singularity:
    * Singularity exec \[-B /mount_this:/to_this\] image_name.simg python3 geo_correct_point_cloud.py \[params\]
