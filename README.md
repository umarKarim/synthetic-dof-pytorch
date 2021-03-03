# Synthetic Depth of Field with and without Depth using PyTorch
## Introduction
This respository contains code for applying synthetic depth of field to images. Synthetic depth of field is common in modern smartphones, for example, portrait mode images in iPhones and Pixel phones, and live focus mode in Samsung phones. If depth information is not available then the code uses MiDaS. The code is pretty simple. 

## Requirements
- PyTorch
- NumPy
- OpenCV
- Matplotlib

## How to use
The parameters for the code are given in the *options.py* file. Basic use without depth is as follows.

```
python3 main.py --im_name test_ims/rocks.jpg
```
The depth map is saved in the *depth_dir*. 

For synthetic depth of field with input image and depthmap from Pixel 4 (or any other depth map), usage is as

```
python3 main.py --im_name test_ims/eraser_input.jpg --depth_name test_ims/eraser_depth.png 
```

The result is stored in *output_dir*.

## Some Results
### Pixel 4 result
![Input](https://github.com/umarKarim/synthetic-dof-pytorch/blob/master/test_ims/eraser_input.jpg#thumbnail)
![Depth map](https://github.com/umarKarim/synthetic-dof-pytorch/blob/master/test_ims/eraser_depth.png#thumbnail)
![Result](https://github.com/umarKarim/synthetic-dof-pytorch/blob/master/output_results/eraser_output.png#thumbnail)

### Random image result
![Input](https://github.com/umarKarim/synthetic-dof-pytorch/blob/master/test_ims/rocks.jpg#thumbnail)
![Depth map](https://github.com/umarKarim/synthetic-dof-pytorch/blob/master/depthmaps/rocks_depth.png#thumbnail)
![Result](https://github.com/umarKarim/synthetic-dof-pytorch/blob/master/output_results/rocks_output.png#thumbnail)








