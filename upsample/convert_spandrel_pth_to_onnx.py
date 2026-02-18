import argparse
import os

import torch
from spandrel import ModelLoader


def ensure_onnx_installed():
    try:
        import onnx  # noqa: F401
    except Exception as err:
        raise RuntimeError(
            "ONNX is required for torch.onnx export. Install it with: pip install onnx"
        ) from err


def parse_shape(shape_text):
    tokens = shape_text.replace(",", "x").split("x")
    shape = tuple(int(token) for token in tokens if token.strip())
    if len(shape) != 4:
        raise ValueError("Expected shape in NxHxWxC format, e.g. 1x48x80x3")
    return shape


class GTAUpsampleWrapper(torch.nn.Module):
    """
    Adapter for GANTheftAuto upsample path:
    - input: NHWC float32 in [-1, 1]
    - output: NHWC float32 in [-1, 1]
    """

    def __init__(self, sr_model):
        super().__init__()
        self.sr_model = sr_model

    def forward(self, x):
        x = torch.permute(x, (0, 3, 1, 2))  # NHWC -> NCHW
        x = (x + 1.0) * 0.5
        y = self.sr_model(x)
        y = torch.clamp(y, 0.0, 1.0)
        y = y * 2.0 - 1.0
        y = torch.permute(y, (0, 2, 3, 1))  # NCHW -> NHWC
        return y


def export_onnx(wrapper, dummy, output_path, opset):
    export_kwargs = dict(
        export_params=True,
        do_constant_folding=True,
        input_names=["input_0"],
        output_names=["output_0"],
        opset_version=opset,
        dynamic_axes=None,
    )

    # Avoid PyTorch dynamo ONNX exporter bugs with some SwinIR checkpoints.
    try:
        torch.onnx.export(
            wrapper,
            dummy,
            output_path,
            dynamo=False,
            **export_kwargs,
        )
        print("Exported with legacy ONNX exporter (dynamo=False).")
        return
    except TypeError:
        # Older torch versions may not accept the dynamo flag.
        pass

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        **export_kwargs,
    )
    print("Exported with default ONNX exporter.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a spandrel-compatible SR .pth model to ONNX for GANTheftAuto upsample flow."
    )
    parser.add_argument("--input", required=True, help="Path to .pth model")
    parser.add_argument("--output", required=True, help="Path to output .onnx model")
    parser.add_argument("--input_shape", default="1x48x80x3", help="Input shape as NxHxWxC")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Export device")
    args = parser.parse_args()

    ensure_onnx_installed()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input model not found: {args.input}")

    input_shape = parse_shape(args.input_shape)
    if input_shape[-1] != 3:
        raise ValueError("This exporter expects 3-channel RGB input.")

    export_device = torch.device(args.device)
    if export_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested, but CUDA is not available.")

    descriptor = ModelLoader().load_from_file(args.input).eval().to(export_device)
    print(f"Loaded model architecture: {descriptor.architecture.name}")
    print(f"Model scale: x{descriptor.scale}")
    print(f"Input channels: {descriptor.input_channels}, output channels: {descriptor.output_channels}")

    if descriptor.input_channels != 3 or descriptor.output_channels != 3:
        raise RuntimeError(
            f"Expected 3->3 channel SR model. Got {descriptor.input_channels}->{descriptor.output_channels}."
        )

    wrapper = GTAUpsampleWrapper(descriptor.model).eval().to(export_device)
    dummy = torch.randn(*input_shape, dtype=torch.float32, device=export_device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with torch.no_grad():
        export_onnx(wrapper, dummy, args.output, args.opset)

    print(f"Saved ONNX model to: {args.output}")
    print(f"Input shape: {input_shape}")


if __name__ == "__main__":
    main()
