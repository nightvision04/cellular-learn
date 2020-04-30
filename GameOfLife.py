import numpy as np
import matplotlib.pyplot as plt
import unittest
import copy
from scipy.signal import convolve2d

def life_step(X):
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    ans = (nbrs_count == 3) | (X & (nbrs_count == 2))
    return ans

def life_step_RGB(X):
    """Game of life step using scipy tools
    Expects a numpy array X with the size(xLim,yLim,3)

    Returns a color stripped version

    When GOL cells (white) encounters a colored cell, it sees it
    as a black cell, but the cell is not converted to black.

    """

    assert X.shape[2] == 3
    XX = copy.deepcopy(X)

    # Find where the pixel is black OR white
    blacks = ~(XX.any(axis=2))
    whites = (XX==255).all(axis=2)
    black_white_mask = blacks | whites

    # GOL
    XX = (XX==255).all(axis=2).astype(int)
    nbrs_count = convolve2d(XX, np.ones((3, 3)), mode='same', boundary='wrap') - XX
    XX = (nbrs_count == 3) | (XX & (nbrs_count == 2))

    X[(XX == True) & (black_white_mask)] = [255, 255, 255]
    X[(XX == False) & (black_white_mask)] = [0, 0, 0]
    return X


class TestGOL(unittest.TestCase):

    def test_run(self):
        np.random.seed(0)
        X = np.random.randint(low=0,high=2,size=(5, 5))
        for i in  range(2):
            X = life_step(X)

    def test_run_rgb(self):

        # Same output
        d = [np.zeros((12, 12, 3)) for i in range(4)]
        # Set starting whites
        for layer in range(4):
            for i in range(2):
                for ii in range(2):
                    d[layer][0 + i][0 + ii] = (255, 255, 255)

        for i in  range(2):
            assert (d[0] == life_step_RGB(d[0])).all() ,'not same'

    def test_run_rgb(self):

        # changing output
        d = np.zeros((3, 3, 3))
        # Set starting whites
        d[0][0] = (255, 255, 255)
        d[0][1] = (0, 0, 0)
        d[0][2] = (0, 0, 0)

        d[1][0] = (255, 255, 255)
        d[1][1] =(0, 0, 0)
        d[1][2] = (0, 0, 0)

        d[2][0] = (0, 0, 0)
        d[2][1] = (255, 255, 255)
        d[2][2] = (0, 0, 0)

        for i in range(1):
            assert d.tolist() != life_step_RGB(d).tolist()

    def test_run_rgb3(self):

        # Color mix
            # changing output
            d = [np.zeros((12, 12, 3)) for i in range(4)]
            # Set starting whites
            for layer in range(4):
                for i in range(3):
                    for ii in range(3):
                        d[layer][0 + i][0 + ii] = (255, 255, 255)
                        d[layer][4 + i][4 + ii] = (255, 0, 255)

            for i in range(2):
                assert d[0].any(axis=2).ravel().tolist() \
                != life_step_RGB(d[0]).any(axis=2).ravel().tolist()


    def test_run_rgb_simple(self):

        # if a 4-pixel square is generated, ensure it remains stationary
        d = np.zeros((3, 3, 3))

        d[0][0] = (255, 255, 255)
        d[0][1] = (0, 0, 0)
        d[0][2] = (255, 255, 255)

        d[1][0] = (0, 0, 0)
        d[1][1] = (0, 0, 0)
        d[1][2] = (0, 0, 0)

        d[2][0] = (255, 255, 255)
        d[2][1] = (0, 0, 0)
        d[2][2] = (255, 255, 255)

        for i in range(3):
            d = life_step_RGB(d)
            assert (d == np.array([[[255., 255., 255.],
                [  0.,   0.,   0.],
                [255., 255. ,255.]],

                [[  0. , 0. , 0.],
                [  0. , 0. , 0.],
                [  0. , 0. , 0.]],

                [[255. ,255., 255.],
                [  0. ,  0. , 0.],
                [255., 255. ,255.]]])).all()

    def test_run_rgb_simple_w_colors(self):

        # if a 4-pixel square is generated, ensure it remains stationary
        d = np.zeros((3, 3, 3))

        d[0][0] = (255, 255, 255)
        d[0][1] = (60, 60, 60)
        d[0][2] = (255, 255, 255)

        d[1][0] = (60, 60, 60)
        d[1][1] = (60, 60, 60)
        d[1][2] = (120, 0, 23)

        d[2][0] = (255, 255, 255)
        d[2][1] = (60, 60, 60)
        d[2][2] = (255, 255, 255)

        for i in range(3):
            d = life_step_RGB(d)
            assert (d == np.array([[[255., 255., 255.],
                [  60. , 60. , 60.],
                [255., 255. ,255.]],

                [[  60. , 60. , 60.],
                [  60. , 60. , 60.],
                [  120. , 0. , 23.]],

                [[255. ,255., 255.],
                [  60. , 60. , 60.],
                [255., 255. ,255.]]])).all()

if __name__ == '__main__':
    unittest.main()