# Local Feature Stencil Code
# Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech with Henry Hu <henryhu@gatech.edu>
# Edited by James Tompkin
# Adapted for python by asabel and jdemari1 (2019)
from matplotlib import pyplot as plt

import argparse

import matplotlib
import numpy as np

from skimage import io, img_as_float32
from skimage.transform import rescale
from skimage.color import rgb2gray

import student as student
from helpers import evaluate_correspondence, cheat_interest_points

matplotlib.use("TkAgg")


# This script
# (1) Loads and resizes images
# (2) Finds interest points in those images                 (you code this)
# (3) Describes each interest point with a local feature    (you code this)
# (4) Finds matching features                               (you code this)
# (5) Visualizes the matches
# (6) Evaluates the matches based on ground truth correspondences

def load_data(file_name):
    




    image1_file = "../data/NotreDame/NotreDame1.jpg"
    image2_file = "../data/NotreDame/NotreDame2.jpg"

    eval_file = "../data/NotreDame/NotreDameEval.mat"

    if file_name == "notre_dame":
        pass
    elif file_name == "mt_rushmore":
        image1_file = "../data/MountRushmore/Mount_Rushmore1.jpg"
        image2_file = "../data/MountRushmore/Mount_Rushmore2.jpg"
        eval_file = "../data/MountRushmore/MountRushmoreEval.mat"
    elif file_name == "e_gaudi":
        image1_file = "../data/EpiscopalGaudi/EGaudi_1.jpg"
        image2_file = "../data/EpiscopalGaudi/EGaudi_2.jpg"
        eval_file = "../data/EpiscopalGaudi/EGaudiEval.mat"

    image1 = img_as_float32(io.imread(image1_file))
    image2 = img_as_float32(io.imread(image2_file))

    return image1, image2, eval_file


def main():
    """
    Reads in the data,

    Command line usage: python main.py -p | --pair <image pair name>

    -p | --pair - flag - required. specifies which image pair to match

    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pair", required=True,
                        help="Either notre_dame, mt_rushmore, or e_gaudi. Specifies which image pair to match")

    args = parser.parse_args()

    image1_color, image2_color, eval_file = load_data(args.pair)

    

    image1 = rgb2gray(image1_color)
    image2 = rgb2gray(image2_color)


    scale_factor = 0.5

    image1 = np.float32(rescale(image1, scale_factor))
    image2 = np.float32(rescale(image2, scale_factor))

    feature_width = 16



    print("Getting interest points...")

    (x1, y1) = student.get_interest_points(image1, feature_width)
    (x2, y2) = student.get_interest_points(image2, feature_width)





    print("Done!")

  

    print("Getting features...")

    image1_features = student.get_features(image1, x1, y1, feature_width)
    image2_features = student.get_features(image2, x2, y2, feature_width)

    print("Done!")


    print("Matching features...")

    matches, confidences = student.match_features(image1_features, image2_features)

    print("Done!")

   

    print("Matches: " + str(matches.shape[0]))

    num_pts_to_visualize = 100

    evaluate_correspondence(image1_color, image2_color, eval_file, scale_factor,
                            x1, y1, x2, y2, matches, confidences, num_pts_to_visualize, args.pair + '_matches.jpg')


if __name__ == '__main__':
    main()
