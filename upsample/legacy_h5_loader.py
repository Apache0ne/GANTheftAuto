import json

import h5py
import tensorflow as tf

try:
    from .addons_compat import InstanceNormalization, SpectralNormalization
except ImportError:
    from addons_compat import InstanceNormalization, SpectralNormalization


def _read_model_config(model_path):
    with h5py.File(model_path, "r") as model_file:
        model_config = model_file.attrs.get("model_config")
    if model_config is None:
        raise ValueError(f"No model_config found in {model_path}")
    if isinstance(model_config, bytes):
        model_config = model_config.decode("utf-8")
    return json.loads(model_config)


def _to_inbound_refs(layer_cfg):
    inbound_nodes = layer_cfg.get("inbound_nodes", [])
    if not inbound_nodes:
        return []
    first_node = inbound_nodes[0]
    if not first_node:
        return []
    if isinstance(first_node[0], list):
        return first_node
    if isinstance(first_node[0], str):
        return [first_node]
    return []


def _resolve_ref(tensor_map, ref):
    if isinstance(ref, (list, tuple)) and len(ref) >= 1:
        ref_name = ref[0]
    else:
        ref_name = ref
    if ref_name not in tensor_map:
        raise KeyError(f"Unknown inbound tensor reference: {ref_name}")
    return tensor_map[ref_name]


def _create_layer(class_name, config):
    if class_name == "Addons>InstanceNormalization":
        return InstanceNormalization.from_config(config)
    if class_name == "Addons>SpectralNormalization":
        return SpectralNormalization.from_config(config)
    if class_name == "Activation":
        return tf.keras.layers.Activation.from_config(config)
    if class_name == "UpSampling2D":
        return tf.keras.layers.UpSampling2D.from_config(config)
    raise ValueError(f"Unsupported layer class in legacy loader: {class_name}")


def _build_model_from_legacy_config(model_config):
    if model_config.get("class_name") != "Functional":
        raise ValueError(f"Unsupported model class: {model_config.get('class_name')}")

    cfg = model_config["config"]
    tensor_map = {}

    for layer_cfg in cfg["layers"]:
        class_name = layer_cfg["class_name"]
        layer_params = layer_cfg["config"]
        layer_name = layer_params["name"]

        if class_name == "InputLayer":
            batch_input_shape = layer_params.get("batch_input_shape")
            if not batch_input_shape or len(batch_input_shape) < 2:
                raise ValueError(f"Invalid InputLayer batch_input_shape: {batch_input_shape}")
            tensor_map[layer_name] = tf.keras.Input(
                shape=tuple(batch_input_shape[1:]),
                name=layer_name,
                dtype=layer_params.get("dtype", "float32"),
            )
            continue

        inbound_refs = _to_inbound_refs(layer_cfg)
        if not inbound_refs:
            raise ValueError(f"Layer '{layer_name}' has no inbound references.")

        if class_name == "TFOpLambda":
            function_name = layer_params.get("function")
            if function_name != "__operators__.add":
                raise ValueError(f"Unsupported TFOpLambda function: {function_name}")

            x = _resolve_ref(tensor_map, inbound_refs[0])
            kwargs = inbound_refs[0][3] if len(inbound_refs[0]) > 3 and isinstance(inbound_refs[0][3], dict) else {}
            y_ref = kwargs.get("y")
            if y_ref is None:
                raise ValueError(f"TFOpLambda '{layer_name}' missing 'y' operand.")
            y = _resolve_ref(tensor_map, y_ref)
            tensor_map[layer_name] = tf.keras.layers.Add(name=layer_name)([x, y])
            continue

        layer = _create_layer(class_name, layer_params)
        inputs = [_resolve_ref(tensor_map, ref) for ref in inbound_refs]
        tensor_map[layer_name] = layer(inputs[0] if len(inputs) == 1 else inputs)

    input_tensors = [_resolve_ref(tensor_map, ref) for ref in cfg["input_layers"]]
    output_tensors = [_resolve_ref(tensor_map, ref) for ref in cfg["output_layers"]]
    return tf.keras.Model(inputs=input_tensors, outputs=output_tensors, name=cfg.get("name", "model"))


def load_legacy_h5_model(model_path):
    model_config = _read_model_config(model_path)
    model = _build_model_from_legacy_config(model_config)

    try:
        model.load_weights(model_path)
    except TypeError:
        model.load_weights(model_path, by_name=True, skip_mismatch=False)

    return model
