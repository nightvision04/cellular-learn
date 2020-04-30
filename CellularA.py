import random
import numpy as np
import time
import datetime
import array
import hashlib
import pickle
import Genotype
import unittest
import copy

def hashFromList(arr_,fast=True):
    '''
    Fast generation of hash from list
    '''
    p = pickle.dumps(arr_, -1)
    if fast==True:
        return hash(p)
    else:
        return hashlib.md5(p).hexdigest()

class CellularA():
    '''
    The player is given a vehicle that automatically starts moving.

    Brake using the spacebar to land the vehicle in the highlighted area.
    '''

    def __init__(self,
                 dna=np.array([]),
                 data=np.array([]),
                 json_data = None):


        assert dna.shape[0] !=0, 'Expected non-empty dna numpy array'

        # First layer should be non empty
        assert data[0].shape[0] !=0, 'Expected non-empty data numpy array'
        # assert json_data !=None, 'json_data should be object, got NONE'
        # e.g. data: np.random.random_sample((30,30)).round().astype(int)
        # e.g. dna: np.random.random_sample((10, 104)).round().astype(int)

        self.g = Genotype.Genotype(data=data,
                     dna=dna,
                     json_data=json_data)


        # Initialize values:
        self.data_state = self.g.data

        # Set time success conditions
        #self.period_ms = 240  # ms per period 0.025 == 40fps
        self.period_max = 100  # Max time before self.trial_state=TIMEOUT

        # Begin the assessment
        self.trial_state = 'STARTED'
        self.period = 0

        # Init world array for fitness function
        self.data_array = []
        self.data_array.append(self.data_state.copy())

        # Init hash array for fitness function
        self.data_hash_array = []


        self.trial_results_array = []


    def update(self):
        '''
        Update period in task and step forward events
        '''

        self.period +=1
        self.g.update()
        self.data_state = self.g.data
        return self

    def evaluate(self):
        '''
        Checks the current task state and updates task.evaluated_state
        as necessary
        '''

        # Append world state
        assert len(self.data_state) == 4, "Expected 4 layers in self.data_state"
        self.data_array.append(copy.deepcopy(self.data_state))

        # Check if world state has previously occured
        dataHash = ""
        for i in range(len(self.data_state)):
            dataHash += str(hashFromList(np.array(self.data_state[i]).ravel().tolist()))
        if dataHash in self.data_hash_array:
            self.trial_state = 'REPEATED'


        # Append world hash state
        self.data_hash_array.append(dataHash)
        self.data_hash_array = list(set(self.data_hash_array))

        # If the simulation has matured
        if self.period >= self.period_max:
            self.trial_state = 'TIMEOUT'

        return self


    def run(self):
        '''
        Runs the simulation until a stopping point is reached
        '''

        running=True
        while running==True:

            self.update()
            self.evaluate()

            if self.trial_state !='STARTED':
                running = False

        return {'data_array': self.data_array,
                'dna': self.g.dna}


class TestCellularA(unittest.TestCase):


    def test_numbers(self):
        self.assertEqual(6+6, 12)

    def test_run(self):
        cellularA = CellularA(data=np.random.random_sample((30,30)).round().astype(int),
                                dna=np.random.random_sample((10, 104)).round().astype(int),
                              json_data=Genotype.JSONData())

        res = cellularA.run()



if __name__ == '__main__':
    unittest.main()
