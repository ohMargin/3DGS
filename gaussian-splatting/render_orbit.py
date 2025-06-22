#
# 环绕轨迹渲染脚本 - 基于3D Gaussian Splatting
#

import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
import math
import subprocess
from PIL import Image
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def generate_orbit_trajectory(center, radius, height, num_frames=120):
    """生成环绕轨迹的相机参数"""
    cameras = []
    
    # 获取更好的初始相机位置 - 使用现有相机的平均位置和方向
    for i in range(num_frames):
        # 计算相机位置 - 沿圆周运动
        angle = (i / num_frames) * 2 * math.pi
        x = center[0] + radius * math.cos(angle)
        z = center[2] + radius * math.sin(angle)
        y = center[1] + height  # 稍微提高相机位置，以便更好地俯视场景
        
        # 相机始终朝向中心点
        position = np.array([x, y, z])
        look_at = np.array(center)
        up = np.array([0, 1, 0])  # 保持相机的上方向为Y轴
        
        # 计算相机旋转矩阵
        forward = look_at - position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # 构建旋转矩阵 (世界坐标到相机坐标)
        rotation = np.stack([right, up, -forward], axis=1)
        
        cameras.append((position, rotation))
    
    return cameras

def render_orbit(model_path, iteration, gaussians, pipeline, background, height=0.0, radius_factor=1.5, num_frames=120, separate_sh=False):
    """渲染环绕轨迹"""
    render_path = os.path.join(model_path, "orbit", "ours_{}".format(iteration), "renders")
    video_path = os.path.join(model_path, "orbit", "ours_{}".format(iteration))
    
    makedirs(render_path, exist_ok=True)
    makedirs(video_path, exist_ok=True)
    
    # 获取场景中心和半径
    center = torch.mean(gaussians.get_xyz, dim=0).cpu().numpy()
    points = gaussians.get_xyz.cpu().numpy()
    
    # 计算更精确的场景中心 - 使用点云的边界框中心
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    center = (min_coords + max_coords) / 2
    
    # 计算合适的半径
    max_dist = np.max(np.linalg.norm(points - center, axis=1))
    radius = max_dist * radius_factor
    
    print(f"场景中心: {center}")
    print(f"轨迹半径: {radius}")
    
    # 生成轨迹
    trajectory = generate_orbit_trajectory(center, radius, height, num_frames)
    
    # 创建一个空白图像
    resolution = (800, 800)
    dummy_img = Image.new('RGB', resolution, color=(0, 0, 0))
    
    # 渲染每一帧
    for idx, (position, rotation) in enumerate(tqdm(trajectory, desc="Rendering orbit")):
        # 创建相机
        camera = Camera(
            resolution=resolution,
            colmap_id=idx,
            R=rotation,
            T=position,
            FoVx=0.7,  # 调整视场角，使其更合适
            FoVy=0.7,
            depth_params=None,
            image=dummy_img,
            invdepthmap=None,
            image_name=f"orbit_{idx:05d}",
            uid=idx,
            data_device="cuda"
        )
        
        # 渲染
        rendering = render(camera, gaussians, pipeline, background, separate_sh=separate_sh)["render"]
        
        # 保存图像
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    
    # 使用FFmpeg将图像序列转换为视频
    video_file = os.path.join(video_path, "orbit_video.mp4")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-pattern_type", "glob",
        "-i", os.path.join(render_path, "*.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        video_file
    ]
    
    print(f"生成视频: {video_file}")
    subprocess.run(ffmpeg_cmd)
    
    return video_file

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="环绕轨迹渲染参数")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--height", default=0.1, type=float, help="相机高度偏移")
    parser.add_argument("--radius_factor", default=1.5, type=float, help="轨迹半径因子")
    parser.add_argument("--num_frames", default=120, type=int, help="帧数")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("渲染环绕轨迹: " + args.model_path)

    # 初始化系统状态
    safe_state(args.quiet)

    with torch.no_grad():
        gaussians = GaussianModel(model.extract(args).sh_degree)
        scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
        
        bg_color = [1,1,1] if model.extract(args).white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        video_file = render_orbit(
            args.model_path, 
            scene.loaded_iter, 
            gaussians, 
            pipeline.extract(args), 
            background, 
            args.height, 
            args.radius_factor, 
            args.num_frames, 
            SPARSE_ADAM_AVAILABLE
        )
        
        print(f"环绕视频已生成: {video_file}") 