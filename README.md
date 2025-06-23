# 3D-Recon: 3D Reconstruction Methods Repository

This repository contains implementations of three popular 3D reconstruction and novel view synthesis approaches:

## Repository Structure

- **[NeRF/](./NeRF/)**: Neural Radiance Fields implementation in PyTorch
- **[Plenoxel/](./Plenoxel/)**: Plenoxels: Radiance Fields without Neural Networks
- **[3DGS/](./3DGS/)**: 3D Gaussian Splatting for Real-Time Radiance Field Rendering

## NeRF: Neural Radiance Fields

NeRF represents scenes as continuous volumetric functions using MLPs to map from 5D coordinates (3D position + 2D viewing direction) to color and density. This implementation is based on the original paper "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (ECCV 2020).

For more details, see the [NeRF README](./NeRF/README.md).

## Plenoxel: Radiance Fields without Neural Networks

Plenoxel represents radiance fields using a sparse voxel grid with optimizable features, achieving significantly faster training than NeRF-based approaches by avoiding neural networks entirely.

For more details, see the [Plenoxel README](./Plenoxel/README.md).

## 3DGS: 3D Gaussian Splatting

3D Gaussian Splatting represents scenes using a set of 3D Gaussians with optimizable properties (position, opacity, color, covariance), enabling real-time high-quality rendering while maintaining competitive training times.


For more details, see the [3DGS README](./3DGS/README-official.md).

## Dataset Preparation

The repository supports various dataset formats:
- COLMAP sparse/dense reconstructions
- NeRF synthetic datasets
- LLFF format datasets
- Custom image sequences (with COLMAP preprocessing)
