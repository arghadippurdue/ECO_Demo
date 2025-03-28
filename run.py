import subprocess
import itertools
import argparse

def parse_step_args(step):
    """
    Returns a dictionary of parsed arguments for a given step.
    You can modify or extend this mapping logic as needed.
    """
    step_map = {
        1: {
            "experiment": "1",
            "mode": "L515", 
            "framework": "torch", 
            "model": "b2", 
            "noise": "0.5", 
            "depth": True, 
            "device": "CPU"
        },
        2: {
            "experiment": "2",
            "mode": "L515", 
            "framework": "ov", 
            "model": "b2", 
            "noise": "0.5", 
            "depth": True, 
            "device": "NPU"
        },
        3: {
            "experiment": "3",
            "mode": "L515", 
            "framework": "ov", 
            "model": "b2", 
            "noise": "0.5", 
            "depth": False, 
            "device": "NPU"
        },
        4: {
            "experiment": "4",
            "mode": "L515", 
            "framework": "ov", 
            "model": "b3", 
            "noise": "0.5", 
            "depth": False, 
            "device": "NPU"
        },
        5: {
            "experiment": "5",
            "mode": "L515", 
            "framework": "ov", 
            "model": "b2", 
            "noise": "99", 
            "depth": True, 
            "device": "NPU"
        },
        6: {
            "experiment": "6",
            "mode": "L515", 
            "framework": "ov", 
            "model": "b3", 
            "noise": "99", 
            "depth": True, 
            "device": "NPU"
        },
    }
    return step_map.get(step, None)

def main():
    parser = argparse.ArgumentParser(
        description="Run the appropriate main script based on mode, framework, and model."
    )
    parser.add_argument("--step", type=int, help="Use a predefined configuration step (1-6).")

    # Make these positional *optional* by using nargs="?" 
    # so that they aren’t strictly required if --step is used.
    parser.add_argument("mode", nargs="?", choices=["dataset", "L515"], help="Mode to run (dataset or L515)")
    parser.add_argument("framework", nargs="?", choices=["ov", "torch"], help="Framework to use (ov or torch)")

    parser.add_argument("--model", choices=["b0", "b1", "b2", "b3"], help="Model to use (b0, b1, b2, or b3)")
    parser.add_argument("--noise", default="0", help="Include the --noise flag in execution")
    parser.add_argument("--depth", action="store_true", help="Include the --depth flag in execution")
    parser.add_argument("--device", default="CPU", help="Device to use (CPU, GPU, or NPU)")
    parser.add_argument("--experiment",default="0")

    args = parser.parse_args()

    # If --step is provided, override everything else
    if args.step is not None:
        step_args = parse_step_args(args.step)
        if not step_args:
            parser.error(f"Invalid step: {args.step}. Must be 1, 2, 3, or 4.")
        
        args.experiment = step_args["experiment"]
        args.mode = step_args["mode"]
        args.framework = step_args["framework"]
        args.model = step_args["model"]
        args.noise = step_args["noise"]
        args.depth = step_args["depth"]
        args.device = step_args["device"]
        

    # At this point, if --step isn’t provided, we rely on the user to have specified:
    #   <mode> <framework> --model <b0/b1/b2/b3> [--noise] [--depth] [--device X]

    # Check that required arguments are indeed set if we didn’t come from a step:
    if not args.mode or not args.framework or not args.model:
        parser.error("Must specify either `--step` OR the required arguments (mode, framework, --model).")

    # Construct filename
    script_filename = f"main_{args.framework}.py"

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
    BACKBONE_LIST = [f"mit_{args.model}"]

    # Set resume path based on model
    RESUME_PATHS = f"ckpt/{args.model}_train/model-best.pth.tar"

    # Fixed arguments (No GPU)
    FIXED_ARGS = "--num-workers 0 --evaluate --save-image 10 --batch-size 1 -c combo_eval"

    # Force CPU execution by unsetting CUDA devices
    CUDA_VISIBLE_DEVICES = ""

    # Determine if --realtime should be added
    realtime_arg = "--realtime" if args.mode == "L515" else ""

    # Conditionally add --noise
    noise_arg = "--noise " + args.noise

    # Conditionally add --depth
    depth_arg = "--depth" if args.depth else ""

    # Set device
    device_arg = "--device " + args.device
    
    experiment_arg = "--experiment " + args.experiment 

    # Loop through parameters and run command
    for backbone, size_r, size_d, refresh_r, refresh_d, quality_r, quality_d in itertools.product(
        BACKBONE_LIST, SIZE_R_LIST, SIZE_D_LIST, REFRESH_R_LIST, REFRESH_D_LIST, QUALITY_R_LIST, QUALITY_D_LIST
    ):
        print(f"Running test with Backbone: {backbone}, Size: {size_r} x {size_d}, "
              f"Refresh: {refresh_r}/{refresh_d}, Quality: {quality_r}/{quality_d}")

        command = (
            f"python {script_filename} {FIXED_ARGS} --input-size {size_r} --input-size-d {size_d} "
            f"--inter {INTER_R} {INTER_D} -ri {refresh_r} {refresh_d} -q {quality_r} {quality_d} "
            f"--cloud --backbone {backbone} --resume {RESUME_PATHS} {realtime_arg} {noise_arg} {depth_arg} {device_arg} {experiment_arg}"
        )

        subprocess.run(command, shell=True)

if __name__ == "__main__":
    main()
