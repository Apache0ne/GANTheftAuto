@echo off
setlocal

set PTH_MODEL=C:\Users\xcool\AppData\Roaming\StabilityMatrix\Models\RealESRGAN\003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth
set ONNX_MODEL=./trained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.onnx
set TRT_MODEL=./trained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.engine

python upsample\convert_spandrel_pth_to_onnx.py ^
 --input "%PTH_MODEL%" ^
 --output %ONNX_MODEL% ^
 --input_shape 1x48x80x3 ^
 --opset 17

if errorlevel 1 goto :error

python upsample\build_trt_engine.py ^
 --input %ONNX_MODEL% ^
 --output %TRT_MODEL%

if errorlevel 1 goto :error

echo Done. TensorRT engine saved to %TRT_MODEL%
exit /b 0

:error
echo Failed building TensorRT upsample model from SwinIR .pth.
exit /b 1
