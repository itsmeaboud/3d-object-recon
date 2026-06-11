import logging
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import gradio as gr
import torch
from gradio_rerun import Rerun

from .data_types import ObjectPointCloud, PathLike, ReconstructionRequest
from .pipeline import ObjectReconstructor, ReconstructionConfig, filter_by_confidence
from .visualizer import delete_rrd, write_to_rrd, save_to_glb

logger = logging.getLogger(__name__)
DEFAULT_SAM3_WEIGHTS = (
    Path(__file__).resolve().parents[2] / "weights/sam3.1_multiplex.pt"
)
DEFAULT_PROMPT = "main object"
DEFAULT_CONFIDENCE_PERCENTILE = 50.0

StatusOutput = tuple[ObjectPointCloud | None, Path | None, str, Path | None]


def uploaded_file_path(file: object) -> Path:
    return Path(file.name if hasattr(file, "name") else file)


def uploaded_file_paths(files: Sequence[object] | None) -> list[Path]:
    if not files:
        raise gr.Error("Upload at least one image before reconstructing.")

    return sorted(uploaded_file_path(file) for file in files)


def result_status(data: ObjectPointCloud, rrd_path: Path, action: str) -> str:
    points_count = 0 if data.filtered_points is None else len(data.filtered_points)
    return f"{action}. Showing {points_count:,} filtered points. RRD: {rrd_path.name}"


def process_data(
    pipeline: ObjectReconstructor,
    request: ReconstructionRequest,
    old_rrd_path: PathLike = "",
    show_cam: bool = False,
) -> StatusOutput:
    request.image_paths = uploaded_file_paths(request.image_paths)

    try:
        pcd_object = pipeline.reconstruct_and_filter(request)
        rrd_path = write_to_rrd(
            data=pcd_object,
            recording_id=str(uuid4()),
            show_cam=show_cam,
        )

        delete_rrd(old_rrd_path)
        status = result_status(pcd_object, rrd_path, "Reconstruction complete")
        return pcd_object, rrd_path, status, rrd_path
    except Exception as exc:
        logger.exception("Reconstruction failed")
        raise gr.Error(f"Reconstruction failed: {exc}") from exc


def post_process(
    data: ObjectPointCloud | None,
    confidence: float,
    old_rrd_path: PathLike,
    show_cam: bool = False,
) -> StatusOutput:
    if data is None:
        raise gr.Error("Run reconstruction before applying a confidence filter.")

    pcd_object = filter_by_confidence(data, confidence)
    rrd_path = write_to_rrd(pcd_object, str(uuid4()), show_cam=show_cam)

    delete_rrd(old_rrd_path)
    status = result_status(pcd_object, rrd_path, "Filter applied")
    return pcd_object, rrd_path, status, rrd_path

def save_glb(
        data: ObjectPointCloud,
) -> str:

    return save_to_glb(data)
    
def build_pipeline(
    *,
    device: str | None = None,
    sam3_weights_path: PathLike | None = None,
    sam3_confidence: float = 0.7,
) -> ObjectReconstructor:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_weights_path = sam3_weights_path or DEFAULT_SAM3_WEIGHTS

    cfg = ReconstructionConfig(
        device=resolved_device,
        sam_confidence=sam3_confidence,
        sam_weights_path=resolved_weights_path,
    )
    logging.info("Initializing pipeline on %s", resolved_device)
    return ObjectReconstructor(cfg)


def build_demo(
    pipeline: ObjectReconstructor | None = None,
    *,
    device: str | None = None,
    sam3_weights_path: PathLike | None = None,
    sam3_confidence: float = 0.7,
) -> gr.Blocks:
    pipeline = pipeline or build_pipeline(
        device=device,
        sam3_weights_path=sam3_weights_path,
        sam3_confidence=sam3_confidence,
    )

    def on_reconstruct(
        path_list,
        prompt,
        conf,
        old_rrd_path,
        show_cam: bool,
    ):
        request = ReconstructionRequest(
            image_paths=path_list,
            prompt=(prompt or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT,
            confidence_perc=conf,
        )

        return process_data(pipeline, request, old_rrd_path, show_cam)

    def on_filter(
        data: ObjectPointCloud,
        confidence: float,
        old_rrd_path: PathLike,
        show_cam: bool,
    ):
        return post_process(data, confidence, old_rrd_path, show_cam)
    
    def on_save(
            data: ObjectPointCloud):
        return

    css = """
    #viewer {min-height: 720px;}
    #controls {max-width: 380px;}
    .app-note {color: var(--body-text-color-subdued);}
    """

    with gr.Blocks(title="3D Object Reconstruction", css=css) as demo:

        temp_rrd_file = gr.State("")
        pcd = gr.State(None)

        gr.Markdown(
            "# 3D Object Reconstruction\n"
            "Upload an ordered image sequence, reconstruct the target object, "
            "then tune the confidence filter.",
            elem_classes="app-note",
        )

        with gr.Row():

            with gr.Column(scale=1, elem_id="controls"):
                img_input = gr.File(
                    label="Image sequence",
                    file_count="multiple",
                    file_types=["image"],
                )

                prompt_input = gr.Textbox(
                    value=DEFAULT_PROMPT,
                    label="Object prompt",
                    placeholder="main object",
                )

                percentile_slider = gr.Slider(
                    minimum=0.0,
                    maximum=100.0,
                    value=DEFAULT_CONFIDENCE_PERCENTILE,
                    step=1.0,
                    label="Confidence percentile",
                )

                show_cam = gr.Checkbox(
                    value=False,
                    label="Show camera frames",
                )

                with gr.Row():
                    run_btn = gr.Button("Reconstruct", variant="primary")
                    filter_btn = gr.Button("Apply filter")

                with gr.Row():
                    save_btn = gr.Button("Save .glb")
                status = gr.Markdown("Ready.")

            with gr.Column(scale=3):
                viewer = Rerun(elem_id="viewer", height=720)

        run_btn.click(
            fn=on_reconstruct,
            inputs=[
                img_input,
                prompt_input,
                percentile_slider,
                temp_rrd_file,
                show_cam,
            ],
            outputs=[pcd, viewer, status, temp_rrd_file],
        )

        filter_btn.click(
            fn=on_filter,
            inputs=[pcd, percentile_slider, temp_rrd_file, show_cam],
            outputs=[pcd, viewer, status, temp_rrd_file],
        )

        save_btn.click(
            fn = on_save,
            inputs = [pcd],
            outputs = [status]
        )

    return demo

demo = build_demo()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        demo.launch(inbrowser=True, share=False)
    finally:
        logger.info("Closing up Gradio")
        demo.close()
