import argparse
import os


def parse_shape(shape_text):
    tokens = shape_text.replace(",", "x").split("x")
    return tuple(int(token) for token in tokens if token.strip())


def replace_dynamic_dims(shape, fallback=1):
    return tuple(fallback if dim < 0 else dim for dim in shape)


def main():
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX model.")
    parser.add_argument("--input", required=True, help="Path to input .onnx model")
    parser.add_argument("--output", required=True, help="Path to output .engine model")
    parser.add_argument("--workspace_gb", type=float, default=2.0, help="TensorRT workspace size in GB")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 build if supported")
    parser.add_argument("--min_shape", default=None, help="Min input shape for dynamic input profile")
    parser.add_argument("--opt_shape", default=None, help="Opt input shape for dynamic input profile")
    parser.add_argument("--max_shape", default=None, help="Max input shape for dynamic input profile")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable TensorRT info-level logging during engine build",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input ONNX model not found: {args.input}")

    import tensorrt as trt

    logger_level = trt.Logger.INFO if args.verbose else trt.Logger.WARNING
    logger = trt.Logger(logger_level)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(args.input, "rb") as onnx_file:
        if not parser.parse(onnx_file.read()):
            print("TensorRT ONNX parser errors:")
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            raise RuntimeError("Failed to parse ONNX model")

    if network.num_inputs != 1:
        raise RuntimeError(f"Expected one input tensor, found {network.num_inputs}")

    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = tuple(input_tensor.shape)
    print(f"ONNX input tensor: {input_name} shape={input_shape}")

    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_gb * (1 << 30))
    if hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    if args.fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled")
        else:
            print("FP16 requested, but this platform does not support fast FP16. Falling back to FP32.")

    if any(dim < 0 for dim in input_shape):
        default_shape = replace_dynamic_dims(input_shape, fallback=1)
        min_shape = parse_shape(args.min_shape) if args.min_shape else default_shape
        opt_shape = parse_shape(args.opt_shape) if args.opt_shape else default_shape
        max_shape = parse_shape(args.max_shape) if args.max_shape else default_shape

        if not (len(min_shape) == len(opt_shape) == len(max_shape) == len(input_shape)):
            raise RuntimeError(
                "Profile shape rank mismatch. min/opt/max shapes must match ONNX input rank."
            )

        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        print(f"Dynamic profile min={min_shape} opt={opt_shape} max={max_shape}")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as engine_file:
        engine_file.write(serialized_engine)

    print(f"Saved TensorRT engine to: {args.output}")


if __name__ == "__main__":
    main()
