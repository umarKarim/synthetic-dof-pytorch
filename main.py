from depth_extraction import DepthExtractor 
from options import Options 
from bokeh import Bokeh
import matplotlib.pyplot as plt 
import cv2 
import numpy as np


class DoFGen():
    def __init__(self, opts):
        self.depth_name = opts.depth_name 
        self.im_name = opts.im_name 
        self.depth_tolerance = opts.depth_tolerance 
        if self.depth_name is None: 
            self.DepthExtractor = DepthExtractor(opts)
        self.BokehGen = Bokeh(opts)
        self.output_dir = opts.output_dir 
        self.__call__()

    def __call__(self):
        # Extracting the image and depth 
        im = cv2.imread(self.im_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.depth_name is None:
            dmap = self.DepthExtractor(im)
        elif self.depth_name.endswith('.png') or self.depth_name.endswith('.jpg'):
            dmap = cv2.imread(self.depth_name, 0)
        elif self.depth_name.endswith('.npy'):
            dmap = np.load(self.depth_name)
        else:
            raise NameError('unknown depth file type')        
        dmap = (dmap - dmap.min()) / (dmap.max() - dmap.min())
        # extracting roi
        print('Click object of interest')
        plt.imshow(im)
        plt.title('Click object of interest')
        pt = plt.ginput(1) 
        plt.close()
        pt = list(*pt)
        pt = [int(x) for x in pt] 
        foc_depth = np.array(dmap[pt[1], pt[0]])
        depth_roi = np.array([foc_depth - self.depth_tolerance, foc_depth + self.depth_tolerance]) 
        depth_roi[depth_roi < 0.0] = 0.0 
        depth_roi[depth_roi > 1.0] = 1.0 

        # applying the blur 
        out_image = self.BokehGen(im, dmap, depth_roi)
        plt.subplot(1, 3, 1)
        plt.imshow(im)
        plt.title('Input')
        plt.subplot(1, 3, 2)
        plt.imshow(dmap)
        plt.title('Depth')
        plt.subplot(1, 3 ,3)
        plt.imshow(out_image / 255.0)
        plt.title('Result')
        plt.show()
        print('Finished, check {}'.format(self.output_dir))

        # saving the results 
        if self.output_dir is not None:
            out_name = self.output_dir + 'result.png'
            out_image_col = np.zeros_like(out_image)
            out_image_col[:, :, 0] = out_image[:, :, -1]
            out_image_col[:, :, 1] = out_image[:, :, 1]
            out_image_col[:, :, 2] = out_image[:, :, 0]
            # out_im_col = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_name, out_image_col)


if __name__ == '__main__':
    opts = Options().opts
    DoFGen(opts)
