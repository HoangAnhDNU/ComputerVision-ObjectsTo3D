"""
Gradio Demo: Upload ảnh + nhập text → tạo mô hình 3D.
Hỗ trợ: Point cloud visualization, mesh reconstruction, xuất .ply/.obj/.glb
"""

import sys
from pathlib import Path

# Thêm project root vào sys.path để import 'src' package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import re
import time
import unicodedata

import gradio as gr
import numpy as np
import torch
import trimesh
import yaml

from src.models.multimodal3d import build_model

# Thử import open3d cho mesh reconstruction chất lượng cao
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[WARN] open3d không có, dùng trimesh fallback (mesh chất lượng thấp hơn)")


def densify_point_cloud(pc, iterations=2, num_neighbors=5, max_dist=0.15):
    """
    Tăng mật độ point cloud bằng adaptive interpolation.
    Chỉ tạo điểm mới giữa các cặp gần nhau (< max_dist) → tránh lấp lỗ sai.

    Args:
        pc: numpy [N, 6] (XYZ + RGB)
        iterations: số lần densify
        num_neighbors: số neighbors để interpolate
        max_dist: khoảng cách tối đa để interpolate (tránh nối qua lỗ)
    Returns:
        numpy [M, 6] với M >> N
    """
    from scipy.spatial import cKDTree

    for it in range(iterations):
        xyz = pc[:, :3]
        rgb = pc[:, 3:]
        tree = cKDTree(xyz)
        dists, indices = tree.query(xyz, k=num_neighbors + 1)

        new_points = []
        for i in range(len(xyz)):
            for j in range(1, num_neighbors + 1):
                nn_idx = indices[i, j]
                d = dists[i, j]
                # Chỉ interpolate giữa cặp gần nhau
                if d < max_dist:
                    for alpha in [0.25, 0.5, 0.75]:
                        mid_xyz = xyz[i] * (1 - alpha) + xyz[nn_idx] * alpha
                        mid_rgb = rgb[i] * (1 - alpha) + rgb[nn_idx] * alpha
                        new_points.append(np.concatenate([mid_xyz, mid_rgb]))

        if new_points:
            new_pc = np.array(new_points, dtype=np.float32)
            pc = np.concatenate([pc, new_pc], axis=0)

            # Voxel downsampling để giữ mật độ đều
            voxel = 0.015 if it == 0 else 0.01
            pc = voxel_downsample(pc, voxel_size=voxel)

    return pc


def voxel_downsample(pc, voxel_size=0.02):
    """Voxel downsampling để giữ mật độ đều."""
    xyz = pc[:, :3]
    voxel_indices = np.floor(xyz / voxel_size).astype(int)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    return pc[unique_idx]


def smooth_point_cloud(pc, num_neighbors=10, iterations=1):
    """Laplacian smoothing cho point cloud."""
    from scipy.spatial import cKDTree

    for _ in range(iterations):
        xyz = pc[:, :3]
        rgb = pc[:, 3:]
        tree = cKDTree(xyz)
        _, indices = tree.query(xyz, k=num_neighbors + 1)

        # Trung bình vị trí với neighbors
        smoothed_xyz = np.mean(xyz[indices], axis=1)
        smoothed_rgb = np.mean(rgb[indices], axis=1)

        # Blend 50% original + 50% smoothed
        pc = np.column_stack([
            0.5 * xyz + 0.5 * smoothed_xyz,
            0.5 * rgb + 0.5 * smoothed_rgb
        ])

    return pc.astype(np.float32)


def remove_statistical_outliers(pc, num_neighbors=16, std_ratio=1.5):
    """Loại bỏ outliers dựa trên khoảng cách trung bình đến k-neighbors."""
    if pc.shape[0] <= num_neighbors + 1:
        return pc

    from scipy.spatial import cKDTree

    xyz = pc[:, :3]
    tree = cKDTree(xyz)
    k = min(num_neighbors + 1, len(pc))
    dists, _ = tree.query(xyz, k=k)
    if dists.ndim == 1:
        return pc
    mean_d = np.mean(dists[:, 1:], axis=1)

    stat_threshold = mean_d.mean() + std_ratio * mean_d.std()
    quantile_threshold = np.quantile(mean_d, 0.99)
    threshold = max(stat_threshold, quantile_threshold)
    keep = mean_d <= threshold

    # Tránh lọc quá mạnh làm mất cấu trúc tổng thể
    min_keep = max(320, int(0.85 * len(pc)))
    if int(np.sum(keep)) < min_keep:
        return pc

    return pc[keep]


def auto_crop_foreground(image, min_area_ratio=0.015, margin_ratio=0.12):
    """Tự động crop foreground để vật thể chiếm khung hình lớn hơn."""
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        return image.convert("RGB"), False

    mask = None
    if "A" in image.getbands():
        alpha = np.array(image.getchannel("A"))
        alpha_mask = alpha > 10
        if float(np.mean(alpha_mask)) >= min_area_ratio:
            mask = alpha_mask

    if mask is None:
        border = np.concatenate(
            [rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0
        ).astype(np.float32)
        bg_color = np.median(border, axis=0)
        diff = np.linalg.norm(rgb.astype(np.float32) - bg_color[None, None, :], axis=2)
        threshold = max(12.0, float(np.percentile(diff, 85) * 0.45))
        mask = diff > threshold

    if float(np.mean(mask)) < min_area_ratio:
        return image.convert("RGB"), False

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    pad_y = max(2, int((y1 - y0) * margin_ratio))
    pad_x = max(2, int((x1 - x0) * margin_ratio))
    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)

    if x0 == 0 and y0 == 0 and x1 == w and y1 == h:
        return image.convert("RGB"), False

    return image.crop((x0, y0, x1, y1)).convert("RGB"), True


def point_cloud_shape_score(pc):
    """Heuristic score để chọn kết quả inference dễ nhìn hơn."""
    if pc is None or len(pc) < 32:
        return -1e9

    xyz = pc[:, :3].astype(np.float32)
    xyz = xyz - np.mean(xyz, axis=0, keepdims=True)

    cov = np.cov(xyz.T)
    if cov.shape != (3, 3):
        return -1e9
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    anisotropy = float(eigvals[0] / (eigvals[2] + 1e-8))

    radius = np.linalg.norm(xyz, axis=1)
    max_r = float(np.max(radius)) + 1e-8
    xyz_n = xyz / max_r

    bins = 10
    voxel_idx = np.floor((xyz_n + 1.0) * 0.5 * bins).astype(np.int32)
    voxel_idx = np.clip(voxel_idx, 0, bins - 1)
    occupancy = np.unique(voxel_idx, axis=0).shape[0] / float(bins ** 3)

    r50 = float(np.quantile(radius, 0.5)) + 1e-8
    r90 = float(np.quantile(radius, 0.9))
    spread = r90 / r50

    return anisotropy + 8.0 * occupancy + 0.2 * spread


VIET_DIACRITIC_CHARS = set("ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợíìỉĩịúùủũụứừửữựýỳỷỹỵ")

OBJECT_KEYWORDS = {
    "chair": ("chair", "armchair", "stool", "seat", "ghế"),
    "table": ("table", "desk", "bàn"),
    "lamp": ("lamp", "light", "đèn"),
    "car": ("car", "automobile", "xe"),
    "plant": ("plant", "potted plant", "tree", "cây"),
    "sofa": ("sofa", "couch"),
}

VI_HINT_TOKENS = {
    "mot", "mau", "go", "ghe", "ban", "chan", "chiec", "co", "voi", "va", "lung", "dem", "hinh"
}


def _strip_accents(text):
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def detect_object_category(text):
    if not text:
        return None

    lowered = text.lower()
    ascii_text = _strip_accents(lowered)

    for category, keywords in OBJECT_KEYWORDS.items():
        for kw in keywords:
            kw_lower = kw.lower()
            if any(ch in VIET_DIACRITIC_CHARS for ch in kw_lower):
                if kw_lower in lowered:
                    return category
            else:
                kw_ascii = _strip_accents(kw_lower)
                if re.search(rf"\b{re.escape(kw_ascii)}\b", ascii_text):
                    return category

    return None


def is_likely_vietnamese_text(text):
    if not text:
        return False

    lowered = text.lower()
    if any(ch in VIET_DIACRITIC_CHARS for ch in lowered):
        return True

    tokens = set(re.findall(r"[a-z]+", _strip_accents(lowered)))
    return len(tokens.intersection(VI_HINT_TOKENS)) >= 2


def build_english_prompt_candidates(text):
    """Tạo danh sách prompt tiếng Anh (ưu tiên EN để demo ổn định hơn)."""
    user_text = (text or "").strip()
    candidates = []
    prompt_source = "default"
    category = detect_object_category(user_text)

    if user_text:
        if is_likely_vietnamese_text(user_text):
            prompt_source = "auto-en-from-vi"
            if category is not None:
                candidates.append(f"a detailed 3d model of a {category}")
                candidates.append(f"a {category}")
            else:
                candidates.append("a detailed 3d object")
        else:
            prompt_source = "user-english"
            candidates.append(user_text)
            if category is not None:
                candidates.append(f"a detailed 3d model of a {category}")
    else:
        candidates.append("a detailed 3d object")

    candidates.append("a 3d object")

    dedup = []
    for prompt in candidates:
        p = re.sub(r"\s+", " ", prompt).strip()
        if p and p not in dedup:
            dedup.append(p)

    return dedup[:4], prompt_source, category


def remove_radius_outliers(pc, keep_ratio=0.985):
    """Loại bỏ các điểm quá xa tâm robust (median center)."""
    if len(pc) < 128:
        return pc

    xyz = pc[:, :3]
    center = np.median(xyz, axis=0)
    radius = np.linalg.norm(xyz - center, axis=1)

    quantile = np.clip(keep_ratio, 0.7, 0.99)
    threshold = np.quantile(radius, quantile)
    keep = radius <= threshold

    min_keep = max(320, int(0.85 * len(pc)))
    if int(np.sum(keep)) < min_keep:
        return pc

    return pc[keep]


def point_cloud_to_mesh_o3d(point_cloud, method="poisson"):
    """
    Chuyển point cloud → mesh chất lượng cao bằng Open3D.

    Args:
        point_cloud: numpy [N, 6] (XYZ + RGB)
        method: "poisson" hoặc "ball_pivoting"
    Returns:
        trimesh.Trimesh hoặc None
    """
    if not HAS_OPEN3D:
        return None

    xyz = point_cloud[:, :3]
    rgb = np.clip(point_cloud[:, 3:], 0, 1)

    # Tạo Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Estimate normals (bắt buộc cho mesh reconstruction)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    try:
        if method == "poisson":
            # Poisson Surface Reconstruction - chất lượng cao nhất
            mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, width=0, scale=1.1, linear_fit=False
            )
            # Loại bỏ vertices có density thấp (noise)
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.05)
            vertices_to_remove = densities < density_threshold
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
        else:
            # Ball Pivoting Algorithm
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist * f for f in [0.5, 1.0, 2.0, 4.0]]
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )

        # Smooth mesh
        mesh_o3d = mesh_o3d.filter_smooth_laplacian(number_of_iterations=3)
        mesh_o3d.compute_vertex_normals()

        # Convert Open3D → trimesh
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        vertex_colors = np.asarray(mesh_o3d.vertex_colors) if mesh_o3d.has_vertex_colors() else None

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if vertex_colors is not None:
            mesh.visual.vertex_colors = np.column_stack([
                (vertex_colors * 255).astype(np.uint8),
                np.full(len(vertex_colors), 255, dtype=np.uint8)
            ])
        return mesh

    except Exception as e:
        print(f"[WARN] Open3D mesh reconstruction failed: {e}")
        return None


def point_cloud_to_mesh_trimesh(point_cloud):
    """
    Fallback: Chuyển point cloud → mesh bằng Alpha Shape (trimesh).

    Args:
        point_cloud: numpy [N, 6] (XYZ + RGB)
    Returns:
        trimesh.Trimesh hoặc None
    """
    xyz = point_cloud[:, :3]
    rgb = np.clip(point_cloud[:, 3:], 0, 1)

    try:
        # Alpha shape - tốt hơn convex hull
        from scipy.spatial import Delaunay
        tri = Delaunay(xyz)
        faces = tri.simplices

        # Lọc faces quá lớn (noise)
        edge_lengths = []
        for face in faces:
            for i in range(3):
                for j in range(i+1, 3):
                    edge_lengths.append(np.linalg.norm(xyz[face[i]] - xyz[face[j]]))
        max_edge = np.percentile(edge_lengths, 95)

        good_faces = []
        for face in faces:
            edges_ok = True
            for i in range(3):
                for j in range(i+1, 3):
                    if np.linalg.norm(xyz[face[i]] - xyz[face[j]]) > max_edge:
                        edges_ok = False
            if edges_ok:
                good_faces.append(face)

        if len(good_faces) == 0:
            good_faces = faces

        mesh = trimesh.Trimesh(vertices=xyz, faces=np.array(good_faces))
        mesh.visual.vertex_colors = np.column_stack([
            (rgb * 255).astype(np.uint8),
            np.full(len(rgb), 255, dtype=np.uint8)
        ])

        # Remove degenerate faces
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()

        return mesh
    except Exception as e:
        print(f"[WARN] Delaunay mesh failed: {e}")

    try:
        # Final fallback: convex hull
        mesh = trimesh.convex.convex_hull(xyz)
        if hasattr(mesh, 'visual'):
            mesh.visual.vertex_colors = np.column_stack([
                (rgb[:len(mesh.vertices)] * 255).astype(np.uint8),
                np.full(len(mesh.vertices), 255, dtype=np.uint8)
            ])
        return mesh
    except Exception:
        return None


def point_cloud_to_mesh(point_cloud, method="auto"):
    """
    Chuyển point cloud → mesh chất lượng cao.
    Tự động chọn phương pháp tốt nhất.
    """
    if method == "auto":
        # Thử Open3D Poisson trước (tốt nhất)
        mesh = point_cloud_to_mesh_o3d(point_cloud, "poisson")
        if mesh is not None and len(mesh.faces) > 10:
            print(f"[Mesh] Poisson reconstruction: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh

        # Thử Open3D Ball Pivoting
        mesh = point_cloud_to_mesh_o3d(point_cloud, "ball_pivoting")
        if mesh is not None and len(mesh.faces) > 10:
            print(f"[Mesh] Ball Pivoting: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh

    # Fallback trimesh
    mesh = point_cloud_to_mesh_trimesh(point_cloud)
    if mesh is not None:
        n_faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0
        print(f"[Mesh] Trimesh fallback: {n_faces} faces")
    return mesh


def save_point_cloud_ply(point_cloud, output_path):
    """Lưu point cloud ra .ply file."""
    xyz = point_cloud[:, :3]
    rgb = np.clip(point_cloud[:, 3:], 0, 1)
    colors = np.column_stack([
        (rgb * 255).astype(np.uint8),
        np.full(len(rgb), 255, dtype=np.uint8)
    ])
    pcd = trimesh.PointCloud(xyz, colors=colors)
    pcd.export(output_path)


def save_mesh(mesh, output_path):
    """Lưu mesh ra file (.obj, .glb, .ply)."""
    if mesh is not None:
        mesh.export(output_path)


class DemoApp:
    def __init__(self, config_path, checkpoint_path, device=None):
        # Resolve relative paths từ project root (thư mục cha của src/)
        project_root = Path(__file__).resolve().parent.parent
        if not os.path.isabs(config_path):
            config_path = os.path.join(project_root, config_path)
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(project_root, checkpoint_path)

        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = build_model(self.config).to(self.device)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            epoch = ckpt.get("epoch", "?")
            best_cd = ckpt.get("best_val_cd", "?")
            print(f"Loaded checkpoint: {checkpoint_path} (epoch={epoch}, best_cd={best_cd})")
        else:
            print(f"WARNING: No checkpoint at {checkpoint_path}, using random weights")

        self.model.eval()
        print(f"Device: {self.device}, Open3D: {HAS_OPEN3D}")

        # Output dir
        project_root = Path(__file__).resolve().parent.parent
        self.output_dir = os.path.join(project_root, "results", "demo")
        os.makedirs(self.output_dir, exist_ok=True)

    def _postprocess_point_cloud(self, pc, quality="high"):
        """
        Post-process point cloud: densify + smooth.
        quality: "low", "medium", "high"
        Tự động điều chỉnh densify dựa trên số điểm đầu vào:
          - >= 1024 pts (A100 model): ít/không densify vì đã đủ chi tiết
          - < 1024 pts (CPU model): densify mạnh hơn để bù thưa
        """
        n_raw = len(pc)
        high_res = n_raw >= 1024  # A100: 2048 pts, CPU: 512 pts

        if quality == "low":
            pc = smooth_point_cloud(pc, num_neighbors=6, iterations=1)
        elif quality == "medium":
            if high_res:
                pc = smooth_point_cloud(pc, num_neighbors=8, iterations=1)
            else:
                pc = densify_point_cloud(pc, iterations=1, num_neighbors=3, max_dist=0.10)
                pc = smooth_point_cloud(pc, num_neighbors=8, iterations=1)
        else:  # high
            if high_res:
                pc = densify_point_cloud(pc, iterations=1, num_neighbors=5, max_dist=0.10)
                pc = smooth_point_cloud(pc, num_neighbors=9, iterations=1)
            else:
                pc = densify_point_cloud(pc, iterations=2, num_neighbors=5, max_dist=0.12)
                pc = smooth_point_cloud(pc, num_neighbors=9, iterations=1)

        # Lọc outliers sau khi densify/smooth
        pc = remove_statistical_outliers(pc, num_neighbors=18, std_ratio=2.4)

        # Giữ point cloud gọn hơn để nhìn rõ hình khối
        if len(pc) > 28000:
            pc = voxel_downsample(pc, voxel_size=0.006)

        return pc

    def _create_plotly_figure(self, pc, title="3D Point Cloud"):
        """Tạo Plotly figure chất lượng cao."""
        import plotly.graph_objects as go

        display_pc = pc
        max_show_points = 20000
        if len(pc) > max_show_points:
            idx = np.random.choice(len(pc), max_show_points, replace=False)
            display_pc = pc[idx]

        xyz = display_pc[:, :3]
        rgb = np.clip(display_pc[:, 3:], 0, 1)

        # Point cloud scatter
        colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in rgb]

        marker_size = 4.5 if len(display_pc) < 1000 else (3.5 if len(display_pc) < 5000 else 3.0)

        fig = go.Figure(data=[go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=colors,
                opacity=0.95,
                line=dict(width=0),
            ),
            hoverinfo='skip',
        )])

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False),
                aspectmode='data',
                bgcolor='rgb(20,20,25)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0),
                    up=dict(x=0, y=1, z=0),
                ),
            ),
            paper_bgcolor='rgb(20,20,25)',
            margin=dict(l=0, r=0, b=0, t=30),
            height=550,
            title=dict(text=f"{title} ({len(display_pc)}/{len(pc)} points)", font=dict(color='white', size=14)),
        )

        return fig

    def _create_mesh_figure(self, mesh):
        """Tạo Plotly figure từ mesh."""
        import plotly.graph_objects as go

        if mesh is None or not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            return None

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)

        # Lấy vertex colors
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vc = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0
            facecolor = [f'rgb({int(vc[f].mean(0)[0]*255)},{int(vc[f].mean(0)[1]*255)},{int(vc[f].mean(0)[2]*255)})' for f in faces]
        else:
            facecolor = None

        fig = go.Figure(data=[go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            facecolor=facecolor,
            opacity=0.95,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5),
            lightposition=dict(x=100, y=200, z=300),
        )])

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False),
                aspectmode='data',
                bgcolor='rgb(20,20,25)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            ),
            paper_bgcolor='rgb(20,20,25)',
            margin=dict(l=0, r=0, b=0, t=30),
            height=550,
            title=dict(
                text=f"3D Mesh ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)",
                font=dict(color='white', size=14),
            ),
        )

        return fig

    def generate_3d(self, image, text, quality="high"):
        """
        Generate 3D point cloud + mesh từ ảnh + text.

        Returns:
            pc_figure, mesh_figure, ply_path, obj_path, glb_path, status
        """
        import plotly.graph_objects as go
        from PIL import Image
        from torchvision import transforms

        if image is None and (text is None or text.strip() == ""):
            return None, None, None, None, None, "Vui lòng cung cấp ảnh hoặc mô tả văn bản!"

        t0 = time.time()

        # Chuẩn bị input
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

        num_views = self.config.get("data", {}).get("num_views", 1)

        if image is not None:
            img = Image.fromarray(image).convert("RGB") if not isinstance(image, Image.Image) else image.convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        else:
            img_tensor = torch.zeros(1, 3, 224, 224, device=self.device)

        if num_views > 1:
            # Multi-view: lặp lại ảnh duy nhất K lần → [1, K, 3, 224, 224]
            img_tensor = img_tensor.unsqueeze(1).expand(-1, num_views, -1, -1, -1).contiguous()

        if text is None or text.strip() == "":
            text = "a 3d object"

        # Tokenize
        text_tokens = self.model.tokenize([text]).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor, text_tokens)

        # Lấy point cloud gốc
        raw_pc = outputs["point_cloud"][0].cpu().numpy()  # [N, 6]

        t_infer = time.time() - t0

        # Post-process: densify + smooth
        pc = self._postprocess_point_cloud(raw_pc, quality=quality)
        t_post = time.time() - t0 - t_infer

        # Tạo point cloud figure
        pc_fig = self._create_plotly_figure(pc, "Point Cloud")

        # Mesh reconstruction
        mesh = point_cloud_to_mesh(pc)
        mesh_fig = self._create_mesh_figure(mesh)
        t_mesh = time.time() - t0 - t_infer - t_post

        # Lưu files
        timestamp = int(time.time())
        ply_path = os.path.join(self.output_dir, f"output_{timestamp}.ply")
        obj_path = os.path.join(self.output_dir, f"output_{timestamp}.obj")
        glb_path = os.path.join(self.output_dir, f"output_{timestamp}.glb")

        save_point_cloud_ply(pc, ply_path)

        if mesh is not None:
            try:
                save_mesh(mesh, obj_path)
            except Exception:
                obj_path = None
            try:
                save_mesh(mesh, glb_path)
            except Exception:
                glb_path = None
        else:
            obj_path = None
            glb_path = None

        total_time = time.time() - t0
        status = (
            f"✓ Tạo mô hình 3D thành công!\n"
            f"  Points: {len(raw_pc)} → {len(pc)} (sau densify)\n"
            f"  Mesh: {len(mesh.faces) if mesh and hasattr(mesh, 'faces') else 0} faces\n"
            f"  Thời gian: inference {t_infer:.1f}s, post-process {t_post:.1f}s, mesh {t_mesh:.1f}s\n"
            f"  Tổng: {total_time:.1f}s"
        )

        return pc_fig, mesh_fig, ply_path, obj_path, glb_path, status

    def create_ui(self):
        """Tạo Gradio UI chuyên nghiệp."""
        css = """
        .gradio-container { max-width: 1200px !important; }
        .output-3d { min-height: 550px; }
        """

        with gr.Blocks(title="Multimodal 3D Generation", css=css, theme=gr.themes.Soft()) as app:
            gr.Markdown("""
            # 🎨 Tạo Mô Hình 3D từ Ảnh và Mô Tả Văn Bản
            **Hệ thống AI đa phương thức** - Kết hợp CLIP vision + text encoder với Point Cloud Decoder
            để tạo mô hình 3D có màu sắc từ ảnh 2D và/hoặc mô tả văn bản.
            """)

            with gr.Row():
                # --- Input Panel ---
                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    image_input = gr.Image(label="Upload ảnh 2D", type="pil", height=250)
                    text_input = gr.Textbox(
                        label="Mô tả văn bản (tiếng Việt hoặc tiếng Anh)",
                        placeholder="Ví dụ: một chiếc ghế gỗ có bốn chân",
                        lines=2,
                    )
                    quality_input = gr.Radio(
                        choices=["low", "medium", "high"],
                        value="high",
                        label="Chất lượng đầu ra",
                        info="High = nhiều points hơn, mesh đẹp hơn, chậm hơn",
                    )
                    generate_btn = gr.Button("✨ Tạo Mô Hình 3D", variant="primary", size="lg")
                    status = gr.Textbox(label="Trạng thái", lines=5, interactive=False)

                # --- Output Panel ---
                with gr.Column(scale=2):
                    gr.Markdown("### Output 3D")
                    with gr.Tabs():
                        with gr.Tab("Point Cloud"):
                            pc_output = gr.Plot(label="3D Point Cloud", elem_classes="output-3d")
                        with gr.Tab("Mesh"):
                            mesh_output = gr.Plot(label="3D Mesh", elem_classes="output-3d")

                    gr.Markdown("### Download")
                    with gr.Row():
                        ply_file = gr.File(label="Point Cloud (.PLY)")
                        obj_file = gr.File(label="Mesh (.OBJ)")
                        glb_file = gr.File(label="Mesh (.GLB)")

            # --- Examples (tiếng Việt + tiếng Anh) ---
            gr.Markdown("### Ví dụ mô tả (hỗ trợ tiếng Việt)")
            gr.Examples(
                examples=[
                    [None, "một chiếc ghế gỗ có bốn chân"],
                    [None, "một chiếc đèn bàn màu trắng"],
                    [None, "một chiếc xe thể thao màu đỏ"],
                    [None, "một chậu cây xanh"],
                    [None, "a blue cube with red top"],
                    [None, "a wooden shelf with multiple tiers"],
                ],
                inputs=[image_input, text_input],
            )

            # --- Event ---
            generate_btn.click(
                fn=self.generate_3d,
                inputs=[image_input, text_input, quality_input],
                outputs=[pc_output, mesh_output, ply_file, obj_file, glb_file, status],
            )

        return app

    def launch(self, share=False, port=7860):
        app = self.create_ui()
        app.launch(share=share, server_port=port)


def main():
    parser = argparse.ArgumentParser(description="Demo Gradio cho Multimodal 3D")
    parser.add_argument("--config", type=str, default="configs/colab.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--share", action="store_true", help="Tạo public link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = DemoApp(args.config, args.checkpoint)
    demo.launch(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
