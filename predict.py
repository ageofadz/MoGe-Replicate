from typing import Any
from cog import BasePredictor, Input, Path

import cv2
import numpy as np
from pathlib import Path as SysPath
import tempfile

import torch
from moge.model.v2 import MoGeModel

import trimesh
import utils3d


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load MoGe-2 model once per container."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = (
            MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
            .to(self.device)
            .eval()
        )

    def predict(
        self,
        image: Path = Input(description="Input RGB image"),
        max_size: int = Input(
            default=800,
            description="Resize so the longer side is at most this many pixels.",
        ),
        remove_edge: bool = Input(
            default=True,
            description="Apply edge-based cleanup on the mesh mask",
        ),
        resolution_level: int = Input(
            default=9,
            ge=0,
            le=10,
            description="MoGe resolution level (higher = slower but more detail)",
        ),
    ) -> Path:
        """
        Upload an image, get back a textured PLY (vertex-colored mesh).
        """

        # -----------------------------
        # 1) Load image and resize
        # -----------------------------
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image at {image}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        longer = max(h, w)
        if longer > max_size:
            scale = max_size / float(longer)
            img_rgb = cv2.resize(
                img_rgb,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        h, w = img_rgb.shape[:2]

        # -----------------------------
        # 2) Prepare tensor and run MoGe
        # -----------------------------
        img_tensor = (
            torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=self.device)
            .permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        )

        with torch.no_grad():
            output = self.model.infer(
                img_tensor,
                resolution_level=resolution_level,
                apply_mask=True,
            )

        # MoGe output keys: "points", "depth", "mask", "normal" (optional), "intrinsics"
        points = output["points"].detach().cpu().numpy()  # (H, W, 3)
        depth = output["depth"].detach().cpu().numpy()    # (H, W)
        mask = output["mask"].detach().cpu().numpy().astype(bool)  # (H, W)

        # -----------------------------
        # 3) Optional edge cleanup
        # -----------------------------
        if remove_edge:
            normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)

            depth_edge = utils3d.numpy.depth_edge(depth, rtol=0.03, mask=mask)
            normals_edge = utils3d.numpy.normals_edge(
                normals,
                tol=5,
                mask=normals_mask,
            )

            mask = mask & ~(depth_edge & normals_edge)

        # -----------------------------
        # 4) Build a textured mesh
        # -----------------------------
        uv = utils3d.numpy.image_uv(width=w, height=h)

        faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
            points,
            img_rgb.astype(np.float32) / 255.0,  # texture from image
            uv,
            mask=mask,
            tri=True,
        )

        vertices = vertices * np.array([1.0, -1.0, -1.0], dtype=np.float32)
        vertex_uvs = vertex_uvs * np.array([1.0, -1.0], dtype=np.float32) + np.array(
            [0.0, 1.0], dtype=np.float32
        )

        # -----------------------------
        # 5) Export to PLY with vertex colors
        # -----------------------------
        tmpdir = SysPath(tempfile.mkdtemp(prefix="moge_"))
        ply_path = tmpdir / "mesh.ply"

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=(vertex_colors * 255).astype(np.uint8),
            process=False,
        )
        mesh.export(ply_path.as_posix())

        # Cog expects a cog.Path object
        return Path(str(ply_path))