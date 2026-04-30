"""
Gradio Demo: Upload ảnh + nhập text → tạo mô hình 3D.
"""

import sys
from pathlib import Path

# Thêm project root vào sys.path để import 'src' package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os

import gradio as gr
import numpy as np
import torch
import trimesh
import yaml

from src.models.multimodal3d import build_model


def point_cloud_to_mesh(point_cloud):
    """
    Chuyển point cloud → mesh bằng Delaunay/convex hull (trimesh).
    Không dùng Poisson reconstruction (cần open3d không hỗ trợ Python 3.13).

    Args:
        point_cloud: numpy [N, 6] (XYZ + RGB)
    Returns:
        trimesh.Trimesh hoặc None nếu thất bại
    """
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:]

    try:
        # Thử tạo convex hull trước
        mesh = trimesh.convex.convex_hull(xyz)
        # Gán màu cho vertices
        if hasattr(mesh, 'visual'):
            mesh.visual.vertex_colors = np.column_stack([
                (rgb * 255).astype(np.uint8),
                np.full(len(rgb), 255, dtype=np.uint8)
            ])
        return mesh
    except Exception:
        pass

    try:
        # Fallback: tạo point cloud scene
        pcd = trimesh.PointCloud(xyz, colors=rgb)
        return pcd
    except Exception:
        return None


def save_point_cloud_ply(point_cloud, output_path):
    """Lưu point cloud ra .ply file dùng trimesh."""
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:]
    pcd = trimesh.PointCloud(xyz, colors=rgb)
    pcd.export(output_path)


def save_mesh_obj(mesh, output_path):
    """Lưu mesh ra .obj file."""
    if mesh is not None:
        mesh.export(output_path)


class DemoApp:
    def __init__(self, config_path, checkpoint_path, device=None):
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = build_model(self.config).to(self.device)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"WARNING: No checkpoint at {checkpoint_path}, using random weights")

        self.model.eval()

        # Output dir
        project_root = Path(__file__).resolve().parent.parent
        self.output_dir = os.path.join(project_root, "results", "demo")
        os.makedirs(self.output_dir, exist_ok=True)

    def _densify_point_cloud(self, pc, num_neighbors=3):
        """
        Tăng mật độ point cloud bằng cách thêm điểm trung gian
        giữa mỗi điểm và nearest neighbors.
        """
        from scipy.spatial import cKDTree

        xyz = pc[:, :3]
        rgb = pc[:, 3:]

        tree = cKDTree(xyz)
        dists, indices = tree.query(xyz, k=num_neighbors + 1)

        new_points = []
        for i in range(len(xyz)):
            for j in range(1, num_neighbors + 1):
                nn_idx = indices[i, j]
                mid_xyz = (xyz[i] + xyz[nn_idx]) / 2.0
                mid_rgb = (rgb[i] + rgb[nn_idx]) / 2.0
                new_points.append(np.concatenate([mid_xyz, mid_rgb]))

        if new_points:
            new_pc = np.array(new_points, dtype=np.float32)
            return np.concatenate([pc, new_pc], axis=0)
        return pc

    def generate_3d(self, image, text):
        """
        Generate 3D point cloud từ ảnh + text.

        Args:
            image: PIL Image (từ Gradio upload)
            text: str (mô tả)
        Returns:
            plotly figure, .ply path, .obj path
        """
        import plotly.graph_objects as go
        from PIL import Image
        from torchvision import transforms

        if image is None and (text is None or text.strip() == ""):
            return None, None, None, "Vui lòng cung cấp ảnh hoặc mô tả văn bản!"

        # Chuẩn bị input
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

        if image is not None:
            img = Image.fromarray(image).convert("RGB") if not isinstance(image, Image.Image) else image.convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(self.device)
        else:
            # Tạo ảnh đen nếu không có ảnh
            img_tensor = torch.zeros(1, 3, 224, 224, device=self.device)

        if text is None or text.strip() == "":
            text = "a generic object"

        # Tokenize
        text_tokens = self.model.tokenize([text]).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor, text_tokens)

        # Lấy point cloud
        pc = outputs["point_cloud"][0].cpu().numpy()  # [N, 6]

        # Densify nhẹ: thêm midpoints giữa nearest neighbors
        pc = self._densify_point_cloud(pc, num_neighbors=3)

        # Tạo Plotly figure
        xyz = pc[:, :3]
        rgb = np.clip(pc[:, 3:], 0, 1)

        fig = go.Figure(data=[go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=[f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in rgb],
                opacity=0.9,
            )
        )])
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',
                bgcolor='rgb(30,30,30)',
            ),
            paper_bgcolor='rgb(30,30,30)',
            margin=dict(l=0, r=0, b=0, t=0),
            height=500,
        )

        # Lưu files
        ply_path = os.path.join(self.output_dir, "output.ply")
        obj_path = os.path.join(self.output_dir, "output.obj")
        try:
            mesh = point_cloud_to_mesh(pc)
            if mesh is not None:
                save_mesh_obj(mesh, obj_path)
                save_point_cloud_ply(pc, ply_path)
            else:
                save_point_cloud_ply(pc, ply_path)
                obj_path = None
        except Exception as e:
            print(f"Mesh reconstruction failed: {e}")
            save_point_cloud_ply(pc, ply_path)
            obj_path = None

        return fig, ply_path, obj_path, "Tạo mô hình 3D thành công!"

    def create_ui(self):
        """Tạo Gradio UI."""
        with gr.Blocks(title="Multimodal 3D Generation") as app:
            gr.Markdown("# Tạo Mô Hình 3D từ Ảnh và Mô Tả Văn Bản")
            gr.Markdown("Upload ảnh 2D và/hoặc nhập mô tả văn bản để tạo mô hình 3D.")

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="Upload ảnh 2D", type="pil")
                    text_input = gr.Textbox(label="Mô tả văn bản", placeholder="Ví dụ: a red chair with wooden legs")
                    generate_btn = gr.Button("Generate 3D", variant="primary")

                with gr.Column(scale=1):
                    output_3d = gr.Plot(label="3D Point Cloud")
                    status = gr.Textbox(label="Trạng thái")

            with gr.Row():
                ply_file = gr.File(label="Download .PLY")
                obj_file = gr.File(label="Download .OBJ")

            generate_btn.click(
                fn=self.generate_3d,
                inputs=[image_input, text_input],
                outputs=[output_3d, ply_file, obj_file, status],
            )

        return app

    def launch(self, share=False, port=7860):
        app = self.create_ui()
        app.launch(share=share, server_port=port)


def main():
    parser = argparse.ArgumentParser(description="Demo Gradio cho Multimodal 3D")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--share", action="store_true", help="Tạo public link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = DemoApp(args.config, args.checkpoint)
    demo.launch(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
