import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Masking")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="Path to config file",
    )
    parser.add_argument("--input-folder", help="Path to folder file.")
    parser.add_argument(
        "--output-folder",
        help="Folder to save output visualizations. "
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    folder_images = os.path.expanduser(args.input_folder)
    images = []
    images = glob.glob(os.path.join(folder_images, "*.png"))
    images.extend(glob.glob(os.path.join(folder_images, "*.jpg")))
    images.extend(glob.glob(os.path.join(folder_images, "*.jpeg")))

    for path in tqdm.tqdm(images):
        img = read_image(path, format="BGR")
        img_name = os.path.splitext(os.path.basename(path))[0] # without file suffix
        start_time = time.time()

        valid_classes = ["bicycle","person","car","truck","bus","motorbike"]

        # binary mask
        predictions, binary_mask = demo.generate_masks_of_classes(img, valid_classes)
        out_filename = os.path.join(args.output_folder, "masked_binary/", img_name + ".png")
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        binary_mask.save(out_filename)

        # masked with alpha
        img_alpha = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        alpha, _, _ = cv2.split(binary_mask.get_image())
        img_alpha[:, :, 3] = alpha
        out_filename = os.path.join(args.output_folder, "masked_alpha/", img_name + ".png")
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        cv2.imwrite(out_filename, img_alpha)

        # masked on black background
        out_filename = os.path.join(args.output_folder, "masked_black/", img_name + ".png")
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        cv2.imwrite(out_filename, cv2.cvtColor(img_alpha, cv2.COLOR_RGBA2RGB))

        # segments
        predictions, segmented = demo.generate_segments_of_classes(img, valid_classes)
        out_filename = os.path.join(args.output_folder, "segmented/", img_name + ".png")
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        segmented.save(out_filename)

        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                path, len(predictions["instances"]), time.time() - start_time
            )
        )