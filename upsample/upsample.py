import os
import numpy as np

try:
    from .addons_compat import get_custom_objects
    from .legacy_h5_loader import load_legacy_h5_model
except ImportError:
    from addons_compat import get_custom_objects
    from legacy_h5_loader import load_legacy_h5_model


model = None
backend_name = None


def _to_uint8(output):
    output = np.asarray(output, dtype=np.float32)
    if not np.isfinite(output).all():
        print("Warning: upsample output contains NaN/Inf values; clamping to finite range.")
        output = np.nan_to_num(output, nan=-1.0, posinf=1.0, neginf=-1.0)
    output = np.clip((output + 1.0) * 127.5, 0.0, 255.0)
    return output.astype(np.uint8)


class KerasBackend:
    def __init__(self, model_path):
        from tensorflow.keras.models import load_model

        try:
            self.model = load_model(
                model_path,
                custom_objects=get_custom_objects(),
                compile=False,
            )
        except Exception as err:
            print(f"Standard Keras load_model failed: {err}")
            print("Falling back to legacy .h5 graph loader.")
            self.model = load_legacy_h5_model(model_path)
        self.model.summary()

    def predict(self, batch):
        return self.model.predict(batch, verbose=0)


class OnnxBackend:
    def __init__(self, model_path):
        import onnxruntime as ort

        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        providers = [provider for provider in preferred if provider in available]
        if not providers:
            providers = available
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        print(f"Loaded ONNX Runtime providers: {self.session.get_providers()}")

    def predict(self, batch):
        outputs = self.session.run(
            self.output_names,
            {self.input_name: np.asarray(batch, dtype=np.float32)},
        )
        return outputs[0] if len(outputs) == 1 else outputs


class TensorRTBackend:
    def __init__(self, model_path):
        import tensorrt as trt
        import torch

        self.trt = trt
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for TensorRT runtime.")

        self.device = torch.device("cuda")
        self.stream = torch.cuda.Stream(device=self.device)
        self.torch_dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.int64: torch.int64,
            trt.bool: torch.bool,
        }

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(model_path, "rb") as engine_file:
            engine_bytes = engine_file.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {model_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_names.append(tensor_name)
            else:
                self.output_names.append(tensor_name)

        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected exactly one input tensor, found {len(self.input_names)}")

        print("Loaded TensorRT engine")

    def _get_torch_dtype(self, trt_dtype):
        if trt_dtype not in self.torch_dtype_map:
            raise RuntimeError(f"Unsupported TensorRT dtype for torch interop: {trt_dtype}")
        return self.torch_dtype_map[trt_dtype]

    def predict(self, batch):
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        input_name = self.input_names[0]
        input_shape = tuple(batch.shape)
        static_input_shape = tuple(self.engine.get_tensor_shape(input_name))
        if any(dim < 0 for dim in static_input_shape):
            if not self.context.set_input_shape(input_name, input_shape):
                raise RuntimeError(f"Could not set TensorRT input shape to {input_shape}")
        elif static_input_shape != input_shape:
            raise RuntimeError(
                f"Input shape mismatch. Engine expects {static_input_shape}, got {input_shape}"
            )

        with self.torch.cuda.device(self.device):
            input_tensor = self.torch.from_numpy(batch).to(device=self.device)
            if not input_tensor.is_contiguous():
                input_tensor = input_tensor.contiguous()
            self.context.set_tensor_address(input_name, int(input_tensor.data_ptr()))

            output_tensors = []
            for output_name in self.output_names:
                output_shape = tuple(self.context.get_tensor_shape(output_name))
                if any(dim < 0 for dim in output_shape):
                    raise RuntimeError(f"Dynamic output shape not resolved for tensor {output_name}: {output_shape}")
                output_dtype = self._get_torch_dtype(self.engine.get_tensor_dtype(output_name))
                output_tensor = self.torch.empty(output_shape, dtype=output_dtype, device=self.device)
                output_tensors.append(output_tensor)
                self.context.set_tensor_address(output_name, int(output_tensor.data_ptr()))

            with self.torch.cuda.stream(self.stream):
                if not self.context.execute_async_v3(stream_handle=int(self.stream.cuda_stream)):
                    raise RuntimeError("TensorRT execution failed")
            self.stream.synchronize()

            outputs = [tensor.detach().cpu().numpy() for tensor in output_tensors]
            return outputs[0] if len(outputs) == 1 else outputs


def load(model_path):
    global model
    global backend_name

    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".h5":
        model = KerasBackend(model_path)
        backend_name = "keras"
    elif ext == ".onnx":
        model = OnnxBackend(model_path)
        backend_name = "onnxruntime"
    elif ext in (".engine", ".trt", ".plan"):
        model = TensorRTBackend(model_path)
        backend_name = "tensorrt"
    else:
        raise ValueError(
            f"Unsupported upsample model format '{ext}'. Expected .h5, .onnx, or .engine/.trt"
        )
    print(f"Upsample backend: {backend_name}")


def inference(img):
    if model is None:
        raise RuntimeError("Upsample model is not loaded. Call load(model_path) first.")

    batch = np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)
    upsampled = model.predict(batch)
    if isinstance(upsampled, (list, tuple)):
        return [_to_uint8(output) for output in upsampled]
    return _to_uint8(upsampled)
