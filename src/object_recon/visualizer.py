import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
import tempfile
import numpy as np
import trimesh
import os
from .pipeline import ObjectReconstructor, ReconstructionConfig, get_rotation_matrix
from .data_types import ObjectPointCloud, ReconstructionRequest, PathLike
from uuid import uuid4


def save_to_glb(
        data: ObjectPointCloud
        ) -> str:
    
    Path(__file__).resolve().parents[2]
    if data.filtered_points.size == 0:
        return "No pointis to save"
    points = data.filtered_points
    colors = data.filtered_colors

    rgb = (colors * 255).astype(np.uint8)
    alpha_channel = np.full((len(rgb), 1), 255, dtype = np.uint8)
    rgba = np.hstack((rgb, alpha_channel))

    trimesh_cloud = trimesh.PointCloud(vertices = points, colors = rgba)

    output_filename = Path(__file__).resolve().parents[2] / f"data/output/output_{str(uuid4())}.glb"
    
    trimesh_cloud.export(output_filename)
    
    return f"Saved succefully: {output_filename}"



def new_rrd_path() -> Path:

    with tempfile.NamedTemporaryFile(prefix = "rrd_", suffix = ".rrd", delete = False) as temp:
        return Path(temp.name)

def delete_rrd(rrd_path) -> None:
    if rrd_path and os.path.isfile(rrd_path):
        os.unlink(rrd_path)



def write_to_rrd(
        data: ObjectPointCloud,
        recording_id: str | None = None,
        output_path: PathLike | None = None,
        show_cam: bool = False
) -> Path:
    
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(name="3D Object Recon", origin="world")
    )
    
    output_path = Path(output_path) if output_path else new_rrd_path()

    recording = rr.RecordingStream(
        application_id = "3D Object Reconstruction",
        recording_id = recording_id
        )
    
    rr.send_blueprint(blueprint, recording = recording)

    parent_path = Path("world")
    
    try:
        recording.save(path = str(output_path))
        recording.log(str(parent_path), rr.Clear(recursive = True))
        recording.log(str(parent_path), rr.ViewCoordinates.RFU, static = True)
        '''recording.log(
            str(parent_path),
            rr.Transform3D(
                rotation = rr.RotationAxisAngle(axis = (0, 1, 0), radians = np.pi /4),   
            ),
            static = True
        )'''
        
        recording.log(
            str(parent_path / "object"),
            rr.Points3D(data.filtered_points, colors = data.filtered_colors)
            )
        
        if show_cam:

            extrinsic = data.extrinsic
            intrinsic = data.intrinsic
            images = data.source_images

            frames_count = extrinsic.shape[0]
            H, W = data.image_shape

            for frame_idx in range(frames_count):

                camera_path = str(parent_path / f"camera_{frame_idx}")

                recording.log(
                    camera_path,
                    rr.Pinhole(image_from_camera = intrinsic[frame_idx], height = H, width = W)
                )
                recording.log(
                    camera_path,
                    rr.Image(image = images[frame_idx])
                )
                recording.log(
                    camera_path,
                    rr.Transform3D(translation = extrinsic[frame_idx, :3, 3], mat3x3 = extrinsic[frame_idx, :3, :3])
                )
                

    finally:
        recording.disconnect()

    return output_path



if __name__ == "__main__":

    path = Path("data/input/images")
    image_paths = sorted(path.glob("*.png"))

    cfg = ReconstructionConfig(
        device = "cpu",
        sam_confidence = 0.7,
        sam_weights_path = "weights/sam3.1_multiplex.pt",
        preprocess_mode = 'crop'
        )
    
    pipeline = ObjectReconstructor(cfg = cfg)
    request = ReconstructionRequest(
        image_paths = image_paths,
        prompt = "toy",
        confidence_perc = 40
        )

    output_obj = pipeline.reconstruct_and_filter(request = request)
    output_obj.filtered_points = output_obj.filtered_points @ get_rotation_matrix(90).T
    output_path = write_to_rrd(data = output_obj, recording_id = str(uuid4()), show_cam = True)
    print(output_path)
    

