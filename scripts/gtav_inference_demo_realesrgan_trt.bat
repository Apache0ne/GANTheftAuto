python inference.py ^
 --saved_model ./trained_models/gan_5_1_17.pt ^
 --data gtav:./data/gtav/gtagan_2_sample ^
 --inference_image_path ./data/gtav/2.png ^
 --show_base_images True ^
 --upsample_model ./trained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.engine
