import argparse 


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--output_dir', type=str, default='output_results/')
        parser.add_argument('--depth_blur_size', type=tuple, default=15)
        parser.add_argument('--depth_tolerance', type=float, default=0.2)
        parser.add_argument('--depth_steps', type=int, default=10)
        parser.add_argument('--resolution', type=tuple, default=None)
        parser.add_argument('--save_depth', type=bool, default=True)
        parser.add_argument('--depth_dir', type=str, default='depthmaps/')
        
        parser.add_argument('--depth_name', type=str, default=None)
        parser.add_argument('--im_name', type=str, default='cup.jpg')

        self.opts = parser.parse_args()

    def __call_(self):
        return self.opts 

