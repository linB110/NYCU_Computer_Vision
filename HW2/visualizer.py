import open3d as o3d
import numpy as np

class Visualizer:
    def clean_pointclouds(self, points_3d, colors=None):
        """
        statistical outlier removal to clean the pointclouds
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # statistical outlier removal 
        pcd, index = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        
        # select inlier points
        inlier_pointclouds = pcd.select_by_index(index)
        
        return inlier_pointclouds
        
    def show_pointclouds(self, cloud):
        if isinstance(cloud, np.ndarray):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)
        else:
            pcd = cloud
                    
        # visualization and render settings
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='SfM Result', width=1024, height=768)
        
        # add coordinate axes for reference
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(pcd)
        vis.add_geometry(axes)
        
        opt = vis.get_render_option()
        opt.point_size = 5.0  
        opt.background_color = np.asarray([0.1, 0.1, 0.1]) 
        
        vis.run()
        vis.destroy_window()
    
    def save_pointclouds(self, cloud, filename = "SfM.ply"):
        if isinstance(cloud, np.ndarray):            
            pcd = o3d.geometry.PointCloud()            
            pcd.points = o3d.utility.Vector3dVector(cloud)        
        else:            
            pcd = cloud        
        o3d.io.write_point_cloud(filename, pcd)
