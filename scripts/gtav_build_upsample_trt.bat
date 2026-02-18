@echo off
setlocal

set H5_MODEL=./trained_models/upsample---[_20]---[______3171]---[_____63420].h5
set ONNX_MODEL=./trained_models/upsample---[_20]---[______3171]---[_____63420].onnx
set TRT_MODEL=./trained_models/upsample---[_20]---[______3171]---[_____63420].engine

python upsample\convert_h5_to_onnx.py ^
 --input %H5_MODEL% ^
 --output %ONNX_MODEL% ^
 --input_shape 1x48x80x3

if errorlevel 1 goto :error

python upsample\build_trt_engine.py ^
 --input %ONNX_MODEL% ^
 --output %TRT_MODEL% ^
 --fp16

if errorlevel 1 goto :error

echo Done. TensorRT engine saved to %TRT_MODEL%
exit /b 0

:error
echo Failed building TensorRT upsample model.
exit /b 1
