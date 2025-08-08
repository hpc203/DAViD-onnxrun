import argparse
import os
from typing import Optional

import cv2
import numpy as np
from runtime.depth_estimator import RelativeDepthEstimator
from runtime.multi_task_estimator import MultiTaskEstimator
from runtime.soft_foreground_segmenter import SoftForegroundSegmenter
from runtime.surface_normal_estimator import SurfaceNormalEstimator
from runtime.visualize import (
    create_concatenated_display,
    visualize_foreground,
    visualize_normal_maps,
    visualize_relative_depth_map,
)


def main():
    """Main function to run the demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Demo script for depth estimation, foreground segmentation, and surface normal estimation"
    )
    parser.add_argument("--image", default="testimgs/1.jpg", type=str, help="Path to input image")

    # Multi-task model path
    parser.add_argument(
        "--multitask-model",
        type=str, default="weights/multi-task-model-vitl16_384.onnx",
        help="Path to multi-task ONNX model that handles all tasks",
    )

    # Individual model paths (if using individual models)
    parser.add_argument("--depth-model", default="weights/depth-model-vitb16_384.onnx", type=str, help="Path to depth estimation ONNX model")
    parser.add_argument("--foreground-model", default="weights/foreground-segmentation-model-vitb16_384.onnx", type=str, help="Path to foreground segmentation ONNX model"
    )
    parser.add_argument("--normal-model", default="weights/normal-model-vitb16_384.onnx", type=str, help="Path to surface normal estimation ONNX model"
    )

    parser.add_argument("--output_path", help="Save result to a path (optional)")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI (useful for headless servers)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return

    multitask_available = args.multitask_model and os.path.exists(args.multitask_model)
    depth_available = args.depth_model and os.path.exists(args.depth_model)
    foreground_available = args.foreground_model and os.path.exists(
        args.foreground_model
    )
    normal_available = args.normal_model and os.path.exists(args.normal_model)

    if not (
        multitask_available
        or depth_available
        or foreground_available
        or normal_available
    ):
        print("Error: At least one model must be provided and exist.")
        print("Available options:")
        print("  --multitask-model: Multi-task model for all tasks")
        print("  --depth-model: Individual depth estimation model")
        print("  --foreground-model: Individual foreground segmentation model")
        print("  --normal-model: Individual surface normal estimation model")
        return

    if args.output_path and not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read the image from {args.image}")
        return

    results = {}

    has_individual_models = (
        (args.depth_model and os.path.exists(args.depth_model))
        or (args.foreground_model and os.path.exists(args.foreground_model))
        or (args.normal_model and os.path.exists(args.normal_model))
    )

    if has_individual_models:
        print("Using individual models...")
        results["individual"] = process_with_individual_models(
            image, args.depth_model, args.foreground_model, args.normal_model
        )
    if args.multitask_model:
        print("Using multi-task model...")
        results["multitask"] = process_with_multitask_model(image, args.multitask_model)

    display_results(image, results, args.output_path, args.headless)


def process_with_individual_models(
    image: np.ndarray,
    depth_model: Optional[str] = None,
    foreground_model: Optional[str] = None,
    normal_model: Optional[str] = None,
):
    """Process image using individual models for each task."""
    results = {}

    if depth_model:
        print("Estimating depth map...")
        depth_estimator = RelativeDepthEstimator(
            onnx_model=depth_model, is_inverse=True
        )
        results["depth"] = depth_estimator.estimate_relative_depth(image)

    if foreground_model:
        print("Estimating foreground segmentation...")
        foreground_segmenter = SoftForegroundSegmenter(onnx_model=foreground_model)
        results["foreground"] = foreground_segmenter.estimate_foreground_segmentation(
            image
        )

    if normal_model:
        print("Estimating surface normals...")
        normal_estimator = SurfaceNormalEstimator(onnx_model=normal_model)
        results["normal"] = normal_estimator.estimate_normal(image)

    return results


def process_with_multitask_model(image: np.ndarray, multi_task: bool):
    """Process image using multi-task model."""
    multitask_estimator = MultiTaskEstimator(
        onnx_model=multi_task, is_inverse_depth=False
    )
    return multitask_estimator.estimate_all_tasks(image)


def display_results(
    image: np.ndarray,
    results: dict[str, np.ndarray],
    output_path: Optional[str] = None,
    headless: bool = False,
):
    """Display results."""
    if "individual" in results:
        individual_result = display_single_model_results(
            image, results["individual"], prefix="Individual"
        )
        if output_path:
            cv2.imwrite(
                os.path.join(output_path, "individual_results.png"),
                individual_result,
            )
    if "multitask" in results:
        print("Displaying multi-task model results...")
        multitask_results = results["multitask"]
        multitask_result = display_single_model_results(
            image, multitask_results, prefix="Multi-task"
        )
        if output_path:
            cv2.imwrite(
                os.path.join(output_path, "multitask_results.png"),
                multitask_result,
            )

    if "individual" in results and "multitask" in results:
        if len(results["individual"]) == len(results["multitask"]):
            compare_results = cv2.vconcat([individual_result, multitask_result])
            if output_path:
                cv2.imwrite(
                    os.path.join(output_path, "comparison_results.png"),
                    compare_results,
                )

    if not headless:
        if (
            "individual" in results
            and "multitask" in results
            and len(results["individual"]) == len(results["multitask"])
        ):
            cv2.imshow("Comparison: Individual vs Multi-task", compare_results)
        if "individual" in results:
            cv2.imshow("Individual Model Results", individual_result)
        if "multitask" in results:
            cv2.imshow("Multi-task Model Results", multitask_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_single_model_results(image, model_results, prefix=""):
    """Display results from a single model (individual or multitask)."""
    visualizations = [image]
    labels = ["Original"]

    foreground_mask = model_results.get("foreground")

    if "depth" in model_results:
        depth_vis = visualize_relative_depth_map(
            image, model_results["depth"], foreground_mask
        )
        visualizations.append(depth_vis)
        labels.append(f"{prefix}/Depth")

    if "foreground" in model_results:
        foreground_vis = visualize_foreground(image, model_results["foreground"])
        visualizations.append(foreground_vis)
        labels.append(f"{prefix}/Foreground")

    if "normal" in model_results:
        normal_vis = visualize_normal_maps(
            image, model_results["normal"], foreground_mask
        )
        visualizations.append(normal_vis)
        labels.append(f"{prefix}/Normals")

    result = create_concatenated_display(visualizations, labels, downscale=2)

    return result


if __name__ == "__main__":
    main()
