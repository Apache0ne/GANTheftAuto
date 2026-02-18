import argparse
import os
import inspect

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
try:
    from .addons_compat import get_custom_objects
    from .legacy_h5_loader import load_legacy_h5_model
except ImportError:
    from addons_compat import get_custom_objects
    from legacy_h5_loader import load_legacy_h5_model

# NumPy 2.x removed aliases still referenced by older tf2onnx builds.
for _alias, _value in {
    "object": object,
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "str": str,
}.items():
    if _alias not in np.__dict__:
        setattr(np, _alias, _value)

if "cast" not in np.__dict__:
    class _LegacyNpCast:
        def __getitem__(self, dtype):
            return lambda values: np.asarray(values, dtype=dtype)

    np.cast = _LegacyNpCast()

import tf2onnx


def patch_onnx_helper():
    try:
        from onnx import helper as onnx_helper
    except Exception as err:
        print(f"ONNX helper patch skipped: {err}")
        return

    if getattr(onnx_helper.make_node, "_gta_explicit_paddings_patch", False):
        return

    original_make_node = onnx_helper.make_node
    make_node_signature = inspect.signature(original_make_node)
    has_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in make_node_signature.parameters.values()
    )
    if not has_var_kwargs:
        return

    def patched_make_node(*args, **kwargs):
        value = kwargs.get("explicit_paddings")
        if isinstance(value, (list, tuple)) and len(value) == 0:
            kwargs.pop("explicit_paddings", None)
        return original_make_node(*args, **kwargs)

    patched_make_node._gta_explicit_paddings_patch = True
    onnx_helper.make_node = patched_make_node


def parse_shape(shape_text):
    if shape_text is None:
        return None
    tokens = shape_text.replace(",", "x").split("x")
    return tuple(int(token) for token in tokens if token.strip())


def resolve_opset(requested_opset):
    supported = sorted(tf2onnx.constants.OPSET_TO_IR_VERSION.keys())
    if not supported:
        return requested_opset

    if requested_opset in supported:
        return requested_opset

    lower_or_equal = [value for value in supported if value <= requested_opset]
    if lower_or_equal:
        resolved = max(lower_or_equal)
        print(
            f"Requested opset {requested_opset} is not supported by tf2onnx "
            f"{tf2onnx.__version__}. Using opset {resolved}."
        )
        return resolved

    resolved = max(supported)
    print(
        f"Requested opset {requested_opset} is below supported range. "
        f"Using opset {resolved}."
    )
    return resolved


def resolve_input_shape(model, requested_shape, batch_size):
    if requested_shape is not None:
        return requested_shape

    if len(model.inputs) != 1:
        raise RuntimeError(
            f"Expected one model input for conversion, found {len(model.inputs)}"
        )

    input_shape = []
    for idx, dim in enumerate(model.inputs[0].shape):
        if dim is None:
            input_shape.append(batch_size if idx == 0 else 1)
        else:
            input_shape.append(int(dim))
    return tuple(input_shape)


def main():
    parser = argparse.ArgumentParser(description="Convert a Keras .h5 upsampler model to ONNX.")
    parser.add_argument("--input", required=True, help="Path to input .h5 model")
    parser.add_argument("--output", required=True, help="Path to output .onnx model")
    parser.add_argument(
        "--input_shape",
        default=None,
        help="Explicit input shape, e.g. 1x48x80x3. If omitted, inferred from model input.",
    )
    parser.add_argument("--input_name", default="input_0", help="ONNX input tensor name")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch used for inferred dynamic dim")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input model not found: {args.input}")

    requested_shape = parse_shape(args.input_shape)
    patch_onnx_helper()
    try:
        model = load_model(
            args.input,
            custom_objects=get_custom_objects(),
            compile=False,
        )
    except Exception as err:
        print(f"Standard Keras load_model failed: {err}")
        print("Falling back to legacy .h5 graph loader.")
        model = load_legacy_h5_model(args.input)
    model.summary()

    input_shape = resolve_input_shape(model, requested_shape, args.batch_size)
    opset = resolve_opset(args.opset)
    input_signature = [tf.TensorSpec(shape=input_shape, dtype=tf.float32, name=args.input_name)]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=args.output,
    )

    print(f"Saved ONNX model to: {args.output}")
    print(f"Input shape: {input_shape}")
    print("Model outputs:")
    for output in model_proto.graph.output:
        print(f"- {output.name}")


if __name__ == "__main__":
    main()
