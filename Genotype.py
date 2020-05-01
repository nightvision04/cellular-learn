import pandas as pd
import numpy as np
import json
import time
import copy
from GameOfLife import life_step_RGB
import unittest

def read_json(filename):
    f = open(filename, 'r')
    data = f.read()
    f.close()
    return json.loads(data)

class JSONData:
    '''
    Contains JSON data instance
    '''

    def __init__(self):
        self.operator_types = read_json('operator_types.json')
        assert isinstance(self.operator_types, dict), 'Expect self.operator_types to be type DICT'

        self.color_types = read_json('color_types.json')
        assert isinstance(self.color_types, dict), 'Expect self.color_types to be type DICT'

        self.color_classes = read_json('color_classes.json')
        assert isinstance(self.color_classes, list), 'Expect self.color_classes to be type DICT'

        self.link_logic_types = read_json('link_logic_types.json')
        assert isinstance(self.link_logic_types, dict), 'Expect self.link_logic_types to be type DICT'

        self.bit_shift_tuples = read_json('bit_shift_tuples.json')
        assert isinstance(self.bit_shift_tuples, dict), 'Expect self.bit_shift_tuples to be type DICT'


class Genotype:
    '''
    Decodes an INT matrix into vector operation instructions.
    '''

    def __init__(self,
                 data=np.array([]),
                 dna=np.array([]),
                 json_data = JSONData()):

        # Assert first layer is not empty
        assert data[0].shape[0] > 0, 'data matrix was empty'
        for i in range(len(data)):
            assert isinstance(data[i],np.ndarray), 'expected numpy matrix for data[{}]'.format(i)

        # Assert that there are 4 layers
        assert len(data)==4,'Expected 4 layers in data'
        assert dna.shape[0] > 0, 'dna matrix was empty'
        assert json_data,'json_data was not loaded'

        self.data = data
        self.dna = dna
        self.json_data = json_data

        # dna should be integer
        assert dna.dtype in ['int64','int32']

        # Set function dictionary
        self.func = {}
        self.func['bit_truth'] = self.dna[:,0]
        self.func['detect_grid'] = self.dna[:,1:10]
        self.func['target_color'] = [None for i in range(len(self.func['bit_truth']))]
        self.func['action_color'] = [None for i in range(len(self.func['bit_truth']))]
        self.func['target_layer'] = [None for i in range(len(self.func['bit_truth']))]
        self.func['action_layer'] = [None for i in range(len(self.func['bit_truth']))]

        # For each function in genotype
        for i in range(self.dna.shape[0]):

            # Set target & action color
            action_color = self.dna[i][10]
            target_color = self.dna[i][11]
            action_layer = self.dna[i][12]
            target_layer = self.dna[i][13]

            self.func['target_color'][i] = tuple(self.json_data.color_types[str(target_color)]['value'])
            self.func['action_color'][i] = tuple(self.json_data.color_types[str(action_color)]['value'])
            self.func['target_layer'][i] = target_layer
            self.func['action_layer'][i] = action_layer

    def transform(self):
        '''
        Transform the data according to the instruction
        '''

        data = self.data
        starting_data_len = len(data)

        # For each instruction
        for i in range(len(self.func['bit_truth'])):

                # Determine which layers instruction focuses on
                target_layer = self.func['target_layer'][i]
                action_layer = self.func['action_layer'][i]
                assert isinstance(target_layer, (int,np.int64,np.int32))
                assert isinstance(action_layer, (int,np.int64,np.int32))

                # Get truth table for target color
                target_color = list(self.func['target_color'][i])
                target_color_array = (data[target_layer].reshape(-1, 3).astype(int) \
                    == np.array(target_color).astype(int)).all(axis=1)\
                    .reshape(data[target_layer].shape[0],data[target_layer].shape[1]).astype(int)

                assert isinstance(target_color,list)
                assert len(target_color_array.shape) ==2

                # Reshape 3x3 target_color truth array
                target_color_truth = np.array(self.func['detect_grid'][i]).reshape(3,3).astype(int)
                assert len(target_color_truth.shape) == 2

                # If function is using 9-bit truth (0)
                if self.func['bit_truth'][i] in [0]:

                    # Pad a copy of target_color_array so array size is identical after lookup
                    target_color_array_pad = np.pad(target_color_array, ((1, 1), (1, 1)),mode='wrap')
                    target_color_truth_lookup = np.lib.stride_tricks.as_strided(\
                            target_color_array_pad, shape=target_color_array_pad.shape + target_color_truth.shape,
                            strides=2*target_color_array_pad.strides)
                    assert target_color_array_pad.shape[0] == target_color_array.shape[0]+2
                    assert target_color_array_pad.shape[1] == target_color_array.shape[1]+2

                    # Transform and remerge
                    target_color_truth_lookup = \
                        target_color_truth_lookup[:-target_color_truth.shape[0] + 1, :-target_color_truth.shape[1] + 1]
                    target_color_truth_lookup = \
                        (target_color_truth_lookup == target_color_truth).all(axis=(-2, -1)).astype(int)
                    assert target_color_truth_lookup.shape[0] == target_color_array.shape[0]
                    assert target_color_truth_lookup.shape[1] == target_color_array.shape[1]

                # If function is using 1-bit truth (1-9)
                elif self.func['bit_truth'][i] in range(1, 10):

                    # look at the 'bit-shift' tuple to search the selected 9-way direction
                    bit_shift = self.func['bit_truth'][i]
                    bit_shift_tuple = tuple(self.json_data.bit_shift_tuples[str(bit_shift)])
                    target_color_truth_lookup = \
                        (np.roll(target_color_array,shift=bit_shift_tuple,axis=(0,1)) == 1).astype(int)
                    assert target_color_truth_lookup.shape[0] == target_color_array.shape[0]
                    assert target_color_truth_lookup.shape[1] == target_color_array.shape[1]

                else:
                    raise KeyError('dna[0] was not in range(0,10)')

                # Lookup immutable color positions
                for key in self.json_data.color_types:
                    if self.json_data.color_types[key]["immutable"]:
                        immutable_color = list(self.json_data.color_types[key]['value'])

                # Skip current instruction if it is tries to action an immutable color
                if list(self.func['action_color'][i]) == immutable_color:
                    continue

                immutable_color_array = (data[action_layer].reshape(-1, 3).astype(int) \
                                      == np.array(immutable_color).astype(int)).all(axis=1) \
                    .reshape(data[action_layer].shape[0], data[action_layer].shape[1]).astype(int)

                # Update action_color where target_color_lookup and ~immutable_color_array were true
                data[action_layer][(target_color_truth_lookup.astype(bool)) & (~immutable_color_array.astype(bool))] = self.func['action_color'][i]

        # #Iterate layer 4 with Game of Life rules after manipulations from dna
        data[0] == life_step_RGB(data[0])
        data[1] == life_step_RGB(data[1])
        data[2] == life_step_RGB(data[2])
        data[3] == life_step_RGB(data[3])

        # Assert the number of layers has not changed
        assert len(data) == starting_data_len

        self.data = data
        return self

    def update(self):
        '''
        Transform self.data based on self.dna instructions
        '''
        self.transform()
        return self

class TestGenotype(unittest.TestCase):

    def test_transform1(self):
        # Data is set to all color 0 on all layers,
        # except layer 1 which is set to color 1
        data = [np.zeros((3, 3, 3)) for i in range(4)]

        # Fill layer 0 with white
        data[2] = np.ones((3,3,3)) * 255

        dna = np.zeros((1, 14)).astype(int)
        dna[0, 0:1] = 0  # 9-way
        dna[0,1:10] = 1 # All Grid
        dna[0][10] = 1  # Action color
        dna[0][11] = 0 # Target color
        dna[0][12] = 1 # Action layer
        dna[0][13] = 2  # Target layer

        g = Genotype(data=data,
                     dna=dna)
        g.update()
        # Did all spaces turn red on layer 1?
        expected_layer = np.ones((3, 3, 3)) * (255,0,0)
        assert (g.data[1] == expected_layer).all()

    def test_transform2(self):
        data = [np.zeros((3, 3, 3)) for i in range(4)]

        # Fill layer 0 with green
        data[0] = np.ones((3,3,3)) * (0,255,0)

        dna = np.zeros((1, 14)).astype(int)
        dna[0,0:1] = 0 # 9-way
        dna[0,1:10] = 1 # All Grid
        dna[0][10] = 1  # Action color
        dna[0][11] = 2 # Target color
        dna[0][12] = 1 # Action layer
        dna[0][13] = 0  # Target layer

        g = Genotype(data=data,
                     dna=dna)
        g.update()

        # Did all spaces turn red on layer 1?
        expected_layer = np.ones((3, 3, 3)) * (255,0,0)
        assert (g.data[1] == expected_layer).all()

    def test_transform_green_center_cross(self):
        data = [np.zeros((3, 3, 3)) for i in range(4)]

        # Fill layer 0 with green
        data[0] = np.ones((3,3,3)) * (0,255,0)
        # Disable corners
        data[0][0][0] = (0,0,0)
        data[0][0][2] = (0, 0, 0)
        data[0][2][0] = (0, 0, 0)
        data[0][2][2] = (0, 0, 0)

        dna = np.array([[
            0, # 9-way
            0,1,0,
            1,1,1,
            0,1,0,
            1, # Action color
            2, # Target color
            1, # Action layer
            0,  # Target layer
        ]])
        g = Genotype(data=data,dna=dna)
        g.update()

        # Did the center space turn red on layer 1?
        expected_layer = np.zeros((3,3,3))
        expected_layer[1][1] = 1
        expected_layer *= (255, 0, 0)
        assert (g.data[1] == expected_layer).all()

    def test_transform_1bit(self):
        data = [np.ones((3, 3, 3)) * (0,0,0) for i in range(4)]

        # Fill layer 0 with green
        data[0] = np.ones((3,3,3)) * (0,255,0)
        # Disable corners
        data[0][0][0] = (0,0,0)
        data[0][0][2] = (0, 0, 0)
        data[0][2][0] = (0, 0, 0)
        data[0][2][2] = (0, 0, 0)

        dna = np.array([[
            1, # Look at top-left
            0,1,0,
            1,1,1,
            0,1,0,
            1, # Action color
            2, # Target color
            1, # Action layer
            0,  # Target layer
        ]])

        g = Genotype(data=data,dna=dna)
        g.update()

        # Did the shift look in the top-left direction??
        expected_layer = np.ones((3,3,3))
        expected_layer[0][0] = (0,0,0)
        expected_layer[0][1] = (0, 0, 0)
        expected_layer[1][0] = (0, 0, 0)
        expected_layer[1][1] = (0, 0, 0)
        expected_layer *= (255, 0, 0)
        assert (g.data[1] == expected_layer).all()

    def test_transform_1bit_2(self):
        data = [np.ones((3, 3, 3)) * (0, 0, 0) for i in range(4)]

        # Fill layer 0 with green
        data[0] = np.ones((3, 3, 3)) * (0, 255, 0)
        # Disable corners
        data[0][0][0] = (0, 0, 0)
        data[0][0][2] = (0, 0, 0)
        data[0][2][0] = (0, 0, 0)
        data[0][2][2] = (0, 0, 0)

        dna = np.array([[
            1,  # Look at top-left
            0, 1, 0,
            1, 1, 1,
            0, 1, 0,
            5,  # Action color
            2,  # Target color
            1,  # Action layer
            0,  # Target layer
        ]])

        g = Genotype(data=data, dna=dna)
        g.update()

        # Did the shift look in the top-left direction??
        expected_layer = np.ones((3, 3, 3))
        expected_layer[0][0] = (0, 0, 0)
        expected_layer[0][1] = (0, 0, 0)
        expected_layer[1][0] = (0, 0, 0)
        expected_layer[1][1] = (0, 0, 0)
        expected_layer *= (90, 90, 120)

        assert (g.data[1] == expected_layer).all()


if __name__ == '__main__':

    unittest.main()