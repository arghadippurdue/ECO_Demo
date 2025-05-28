import subprocess
import itertools
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run the appropriate main script based on mode, framework, and model."
    )

    parser.add_argument(
        "--framework", choices=["ov", "torch"], help="Framework to use (ov or torch)")
    parser.add_argument(
        "--mode", choices=["dataset", "realtime"], help="Mode to run (dataset or L515)")
    parser.add_argument("--backbone", choices=["mit_b2", "mit_b3"], help="Model to use (mit_b2 or mit_b3)")
    parser.add_argument("--quantization", choices=[
                        "fp16", "fp16a", "int8", "int8a"], help="Model Quantization", default="fp16")
    parser.add_argument("--noise", default="0",
                        help="Include the --noise flag in execution")
    parser.add_argument("--depth", action="store_true",
                        help="Include the --depth flag in execution")
    parser.add_argument("--device", default="CPU",
                        help="Device to use (CPU, GPU, or NPU)")
    parser.add_argument("--experiment", default="0")
    parser.add_argument("--reportpower", action="store_true",
                        help="Include the --report-power flag in execution")

    args = parser.parse_args()

    # Check that required arguments are indeed set if we didn’t come from a step:
    if not args.mode or not args.framework or not args.backbone:
        parser.error(
            "Must specify  required arguments (--mode, --framework, --backbone).")

    # Construct filename
    script_filename = f"main.py"


# ---------------------------------------------main Arguments -------------------------------------------------
    # Fixed interpolation methods (DO NOT CHANGE)
    INTER_R = "cubic"
    INTER_D = "nearest"
    # SINGLE TEST PARAMETERS
    SIZE_R_LIST = [468]
    SIZE_D_LIST = [468]
    REFRESH_R_LIST = [1]
    REFRESH_D_LIST = [1]
    QUALITY_R_LIST = [120]
    QUALITY_D_LIST = [120]
    # Set backbone based on the provided model argument
    BACKBONE_LIST = [args.backbone]
    # Set resume path based on model
    backbone_short = args.backbone.split('_')[1]
    RESUME_PATHS = f"ckpt/{backbone_short}_train/model-best.pth.tar"
    # Fixed arguments (No GPU)
    FIXED_ARGS = "--num-workers 0 --save-image 10 --batch-size 1 -c combo_eval"

# Behaviour Arguments--------------------------------------------------------------------------------------

    mode_arg = "--realtime" if args.mode == "realtime" else "--dataset"
    noise_arg = "--noise " + args.noise
    depth_arg = "--depth" if args.depth else ""
    device_arg = "--device " + args.device
    experiment_arg = "--experiment " + args.experiment
    framework_arg = "--framework " + args.framework
    reportpower_arg = "--reportpower" if args.reportpower else ""

    # Loop through parameters and run command
    for backbone, size_r, size_d, refresh_r, refresh_d, quality_r, quality_d in itertools.product(
        BACKBONE_LIST, SIZE_R_LIST, SIZE_D_LIST, REFRESH_R_LIST, REFRESH_D_LIST, QUALITY_R_LIST, QUALITY_D_LIST
    ):
        # print(f"Running test with Backbone: {backbone}, Size: {size_r} x {size_d}, "
        #       f"Refresh: {refresh_r}/{refresh_d}, Quality: {quality_r}/{quality_d}")

        command = (
            f"python {script_filename} {FIXED_ARGS} --input-size {size_r} --input-size-d {size_d} "
            f"--inter {INTER_R} {INTER_D} -ri {refresh_r} {refresh_d} -q {quality_r} {quality_d} "
            f"--cloud --resume {RESUME_PATHS} "
            f" --backbone {backbone} {noise_arg} {depth_arg} {device_arg} {experiment_arg} {reportpower_arg} "
            f"{mode_arg} {framework_arg} "
        )

        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
