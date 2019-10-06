import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    for i in range(1, 119):
        filename = f'{i:03d}'

        img = np.load(os.path.join(args.input_dir,
                                   'images', filename + '.npy'))
        seg = np.load(os.path.join(args.input_dir, 'seg', filename + '.npy'))

        # sanity check
        assert 0 == len(seg[np.logical_and(seg != 0, seg != 1)])

        plt.imshow(img)
        plt.imshow(seg, cmap='Reds', alpha=0.5)
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir',
                        help='the directory for output')

    args = parser.parse_args()

    main(args)
