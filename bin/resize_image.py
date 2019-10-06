import os
import cv2
import numpy as np
import argparse


def main(args):

    # make dir for output
    os.makedirs(args.output_dir, mode=0o755, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'images'),
                mode=0o755, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'seg'),
                mode=0o755, exist_ok=True)

    for i in range(1, 119):
        filename = f'{i:03d}'

        img = cv2.imread(os.path.join('images', filename + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = np.load(os.path.join('groundTruth', filename + '.npy'))

        img_resized = cv2.resize(
            img, (args.dim, args.dim), interpolation=cv2.INTER_LANCZOS4)
        seg_resized = cv2.resize(
            seg, (args.dim, args.dim), interpolation=cv2.INTER_LANCZOS4)

        seg_cleaned = np.zeros_like(seg_resized)
        seg_cleaned[seg_resized == 2] = 1
        seg_cleaned[seg_resized > 3] = 1

        seg_thick = np.zeros_like(seg_cleaned)
        for h in range(args.dim):
            for w in range(args.dim):

                h1 = h - args.thick
                h1 = max(h1, 0)

                w1 = w - args.thick
                w1 = max(w1, 0)

                h2 = h + args.thick
                h2 = min(h2, args.dim - 1)

                w2 = w + args.thick
                w2 = min(w2, args.dim - 1)

                if seg_cleaned[h, w]:
                    seg_thick[h1:h2, w1:w2] = 1

        np.save(os.path.join(args.output_dir, 'images', filename), img_resized)
        np.save(os.path.join(args.output_dir, 'seg', filename), seg_thick)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir',
                        help='the directory for output')
    parser.add_argument('-t', '--thick', type=int, default=7,
                        help='the line thickness for segmentation')
    parser.add_argument('-d', '--dim', type=int, default=512,
                        help='new dimensions for resized images')

    args = parser.parse_args()

    main(args)
