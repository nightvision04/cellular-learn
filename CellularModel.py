import matplotlib.pyplot as plt
import numpy as np
import random
import time
import copy
from Genotype import JSONData
from CellularA import CellularA
import unittest
import Viewer
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

class CellularModel():
    '''

    Evolves DNA CellularAutomata to solve a problem set. Uses scikit learns 
    fit() and predict() standard methods. Genetic algorithms (GA ) are non-deterministic
    and the provided solution is not expected to be optimal. The solution is highly
    sensitive to the initial population.
    
    To determine the simplest stable solution, DNA is first given an instruction
    space of 1, and incremented after being scored. The final solution is determined
    when there is no significant improvement found after adding a parameter or the
    max_depth parameter has been reached during the param_grid search.

    The fit(X,y) method expects that y has more than one class.

    If X is continuous, the training data will be discretized using the
    parameter quantize_size. The default value is (15,15). This is done so that
    each cellular automaton is able to make decisions in discrete unit space.

    Usage:
    model = CellularModel(N=150,
                        mutate_proba=0.05,
                        max_epochs=10,
                        auto_plot=False,
                        quantize_size=(15,15))
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test,y_test)

    Expects:
        N INT the population size used for selection
        mutate_proba FLOAT the probability of mutating a child
        
    '''

    def __init__(
            self,
            N=50,
            mutate_proba=0.01,
            max_epochs=60,
            auto_plot=False,
            min_depth=15,
            max_depth=40,
            quantize_size='auto',
            fitness_func='accuracy',
            silent=False):

        # Size of the world. E.g: 10x10
        self.size = (5,14)
        self.dna_scores = []

        N = int(N)
        mutate_proba = float(mutate_proba)
        self.N = N
        self.mutate_proba = mutate_proba
        self.best_child = None
        self.intitialized = False
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.isFit = False
        self.bags = []
        self.silent = silent
        self.y_classes = None
        self.auto_plot = auto_plot
        self.max_epochs = max_epochs
        self.json_data = JSONData()
        self.fitness_func = fitness_func

        assert isinstance(quantize_size,(tuple,str))
        if isinstance(quantize_size,tuple):
            assert quantize_size[0] > 6 and quantize_size[1] > 6,\
                'quantize_size must be a tuple greater than 6 in X,y dimensions. e.g.(7,7)'
        elif isinstance(quantize_size,str):
            assert quantize_size =='auto',\
                'Unknown quantize_size value: "{}". Use "auto" or tuple, e.g. (15,15)'\
                    .format(quantize_size)
            # Set to default space size
            quantize_size = (15,15)

        self.quantize_size = quantize_size

        # Initialize minmaxscaler for discritizing features/labels
        self.scaler_X = MinMaxScaler(
            feature_range=(2,self.quantize_size[1]-2))
        self.scaler_y = MinMaxScaler(
            feature_range=(2, self.quantize_size[1] - 2))

        assert isinstance(max_depth,int),\
            'Expected max_depth INT'
        assert isinstance(min_depth, int), \
            'Expected min_depth INT'
        assert isinstance(
            auto_plot, bool), 'Expected auto_plot to be BOOL, got type: {}'.format(
            type(auto_plot))
        assert isinstance(
            max_epochs, int), 'Expected max_epochs to be INT, got type: {}'.format(
            type(max_epochs))


    def save(self,fp=''):
        '''
        Save the model instance to a file
        '''

        assert isinstance(fp,str),\
            'Expected fp STR, but got type: {}'.format(type(fp))
        with open(fp, "wb") as f:
            pickle.dump(self, f)

        return self

    def preprocess(self,
                   X=None,
                   y=None,
                   training=True):
        '''
        Discritizes data if continuous

        Expects:
            X np.ndarray int32|int64|float64    training data features
            y np.ndarray int64|int32            training data labels
            training BOOL (default: True)       if False load scaler
        '''

        if not training:
            assert self.isFit,\
                'Expected training=True because model has not run .fit().'

        assert isinstance(X, np.ndarray)
        assert X.shape[0] >=1,\
            'Expected X to have at least 1 sample.'
        if X.dtype not in ['int32','int64','float32','float64']:
            raise TypeError('X is incorrect dtype')

        if training:
            assert isinstance(y, np.ndarray)
            assert y.shape[0] >= 1, 'y was an empty array'
            assert X.shape[0] == y.shape[0], \
                'X and y have incompatible shapes {},{}'.format(X.shape, y.shape)
            assert y.shape[1] == 1, 'y must use the shape ({},1)'.format(y.shape[1])
            if y.dtype not in ['int32', 'int64']:
                raise TypeError('y must be type INT')
            assert len(np.unique(y)) > 1, \
                'There was a single class in y. Expected more than 1, ({})'.format(np.unique(y))

        # Use max min scaler to 90% of self.quantize_space
        if training:
            self.scaler_X.fit(X)
        X = self.scaler_X.transform(X)
        X = np.floor(X)

        assert X.dtype in ['float64','float32']
        if training:
            assert y.dtype in ['int64','int32']
        else:
            y=np.array([])
        return X,y

    def predict(self,
                X,
                plot=False):
        '''
        Predicts the y class of X

        Expects:
            X np.ndarray            Target of class prediction
        '''

        assert isinstance(X, np.ndarray),\
            'X is expected to be of type: np.ndarray'
        assert X.shape[0] >= 1,\
            'X must be a minimum of 1 sample'
        assert self.isFit, \
            'Run .fit(X,y) prior to .predict(X)'

        # Apply scaling
        X,y = self.preprocess(X,training=False)
        p = ProbSpace(X, quantize_size=(15, 15), training=False)
        self.bags = p.stratified_training_bags(X)
        assert isinstance(self.bags,list)
        data = self.bags[0]

        # Init empty y_pred
        y_pred_array = np.empty((len(data),1)).astype(int)

        for m in range(len(data)):

            # Run data in the Cellular Automata env
            res = CellularA(data=copy.deepcopy(data[m]['space']),
                            dna=self.selected_model_dna,
                            json_data=self.json_data).run()

            # Determine estimate with percentage of class-associate cells
            cTypes = self.json_data.color_types
            class_colors = [cTypes[key]['value'] for key in sorted(cTypes.keys()) \
                            if cTypes[key]['type'] == 'class_reserved']

            y_pred_proba = np.empty((len(self.y_classes)))
            y_pred_sums = np.empty((len(self.y_classes)))

            for i in range(len(self.y_classes)):
                y_pred_sums[i] = np.sum([np.sum((res['data_array'][-1][ii].reshape(-1, 3).astype(int) \
                                 == np.array(class_colors[i]).astype(int)).all(axis=1)) for ii in range(4)])

                # Determine the percentage of guess density for each possible y class
                y_pred_proba = y_pred_sums.astype(float) / y_pred_sums.astype(float).sum()
                y_pred_proba = np.array([x if str(x) != 'nan' else 0.01 for x in y_pred_proba])

            y_pred = self.y_classes[np.argmax(y_pred_proba)]

            if plot:
                Viewer.Playback(res['data_array'],
                text='Predict class: '. \
                format(y_pred),
                dna=self.selected_model_dna).run()

            # Temporary rule
            y_pred = int((y_pred /10) +1)
            y_pred_array[m][0] = y_pred

        return y_pred_array

    def fit(self,
            X=np.array([]),
            y=np.array([]),
            silent=False):
        '''
        Increments complexity of the dna and logs the predictive power.

        Expects:
            X np.ndarray int32|int64|float64    training data features
            y np.ndarray int64|int32            training data labels
            silent BOOL (default: True)         silences model fit output
        '''

        assert isinstance(X,np.ndarray)
        assert isinstance(y,np.ndarray)
        self.y_classes = sorted(list(np.unique(y)))

        # 10 generations will be used to score each dna configuration
        self.param_grid = {
            'size': [(i,14) for i in range(self.min_depth,self.max_depth+1)]
        }
        self.param_best_children_array = [None for i in range(self.max_depth-self.min_depth+1)]
        self.param_scores = [None for i in range(self.max_depth-self.min_depth+1)]
        self.param_best_dna = [None for i in range(self.max_depth-self.min_depth+1)]

        fitting=True
        i=0
        while fitting==True:

            # Set parameters from param_grid
            self.size = self.param_grid['size'][i]
            # Reset the population
            self.initialize()

            # Preprocess the data
            X,y = self.preprocess(X,y)

            # Generate the sample spaces compatible with the CA algo
            if self.max_epochs >= 10:
                splits = 10
            else:
                splits = self.max_epochs

            # Multiply by 50
            splits = splits *50


            p = ProbSpace(X, y, quantize_size=(15, 15),
                          splits=splits, training=True)
            self.bags = p.stratified_training_bags(X, y)

            if not self.silent:
                print('Depth:',self.min_depth+i)

            # Run - This should contain class stratified batches of 30
            self.run()

            # Store the results
            self.param_best_children_array[i] = self.generation_list
            self.param_scores[i] = self.generation_fitness
            self.param_best_dna[i] = [x.world for x in self.generation_list]

            assert isinstance(self.generation_list, list)
            assert isinstance(self.generation_fitness,list)

            if i >= (self.max_depth-self.min_depth):
                fitting = False
                self.isFit = True

            i+=1

        # Set model dna
        best_depth_index = np.argmax(\
            [np.amax(self.param_scores[i]) for i in range(self.max_depth-self.min_depth+1)])
        assert isinstance(best_depth_index, (int,np.int64,np.int32)),\
            'best_depth_index was type: {}'.format(type(best_depth_index))
        self.selected_dna_depth = best_depth_index+1 + self.max_depth

        self.selected_model_fitness = np.amax(self.param_scores[best_depth_index])
        assert isinstance(self.selected_model_fitness,float)

        best_dna_index = np.argmax(self.param_scores[best_depth_index])
        self.selected_model_dna = self.param_best_dna[best_depth_index][best_dna_index]
        assert isinstance(self.selected_model_dna,np.ndarray)


        return self


    def initialize(self):
        'Create population of world elements.'

        self.generation_list = []
        self.generation_fitness = []
        self.current_pop = [World(*self.size,y_classes=self.y_classes).gen_world()
                            for i in range(self.N)]
        self.next_pop = []
        self.intitialized = True
        return self

    def selection(self):
        'Evaluate fitness of population and build mating pool from current pop.'

        gen = len(self.generation_list)
        # Gen is used to determine bag for sampling
        random.seed(gen)
        bag_index = random.choice(range(len(self.bags)))
        selected_bag = self.bags[bag_index]
        self.current_pop_fitness = [
            Fitness(w,
                    json_data=self.json_data,
                    gen=gen,
                    data=selected_bag,
                    y_classes=self.y_classes,
                    fitness_func=self.fitness_func).run().fitness_value for w in self.current_pop]

        # exp func -- Seems to really help polarize best performers!
        # Only works well if the fitness function is already seperating performers.
        # Sensitive to size of N
        exp_func = 3
        self.current_pop_fitness = [
            x**exp_func for x in self.current_pop_fitness]
        return self

    def reproduction(self):
        '''
        Generates and mutates child until the current pop is completed.
        '''

        for i in range(self.N):
            parents = self.pick_parents()
            child = self.crossover(parents)
            child = self.mutation(child)
            self.birth_child(child)

        return self

    def pick_parents(self):
        '''
        Choose parents (probability based on fitness)
        '''

        parents = random.choices(
            population=self.current_pop,
            weights=self.current_pop_fitness,
            k=2)
        return parents

    def crossover(self, parents):
        '''
        Combine DNA of parents.
        '''

        # For the target 2d array size, take random
        # DNA from either of the 2 parents
        child = World(*self.size,y_classes=self.y_classes)
        child.gen_world()

        i = 0
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                parent_selection = random.choice([0, 1])
                child.world[y][x] = parents[parent_selection].world[y][x]
                i += 1

        # Make sure that all data points have been iterated over.
        parent_len = parents[0].world.shape[0] * parents[0].world.shape[1]
        assert i == parent_len,\
            'All data points were not iterated over. Got: {} Expected: {}'.\
                format(i, parent_len)

        return child

    def mutation(self, child):
        '''
        Slightly mutate the child based on probability of self.mutate_proba
        '''

        # Each array index uses unique mutation ranges
        mutate_map = [[0,1,2,3,4,5,6,7,8,9],  # Bit direction
                      [0,1],        # top-left position contains white cell
                      [0,1],        # top-center position contains white cell
                      [0,1],        # top-right position contains white cell
                      [0,1],        # mid-left position contains white cell
                      [0,1],        # mid-center position contains white cell
                      [0,1],        # mid-right position contains white cell
                      [0,1],        # bottom-left position contains white cell
                      [0,1],        # bottom-center position contains white cell
                      [0,1],        # bottom-right position contains white cell
                      range(len(self.y_classes)+3), # Target color
                      range(len(self.y_classes)+3), # Action color
                      [0,1,2,3],     # Target layer
                      [0,1,2,3]]     # Action layer

        i = 0
        for y in range(self.size[0]):
            for x in range(self.size[1]):

                # Only mutate if the self.mutate_proba event is picked.
                chance_to_mutate = random.random() < self.mutate_proba
                if chance_to_mutate == 1:
                    child.world[y][x] = random.choice(mutate_map[x])
                i += 1

        child_len = child.world.shape[0] * child.world.shape[1]
        assert i == child_len,\
            'All data points were not iterated over. Got: {} Expected: {}'\
                .format(i, parent_len)
        return child

    def birth_child(self, child):
        '''
        Add a child to the next population generation.
        '''

        self.next_pop.append(child)
        return self

    def update_pop(self):
        '''
        Replace the current pop with nextGen pop.
        '''

        self.current_pop = self.next_pop
        self.next_pop = []

        return self

    def run(self):

        if not self.intitialized:
            self.initialize()
        while True:

            self.selection()
            self.reproduction()

            # Save self.best_child
            self.best_child = self.current_pop[np.argmax(
                self.current_pop_fitness)]
            self.generation_list.append(self.best_child)

            # Get fitness of best child
            gen = len(self.generation_list)
            # Gen is used to determine bag for sampling
            random.seed(gen)
            bag_index = random.choice(range(len(self.bags)))
            selected_bag = self.bags[bag_index]
            f = Fitness(self.best_child,
                        json_data=self.json_data,
                        gen=gen,
                        data=selected_bag,
                        y_classes=self.y_classes,
                        fitness_func=self.fitness_func)
            f.run()
            best_child_fitness = round(f.fitness_value, 2)
            self.generation_fitness.append(best_child_fitness)

            if len(self.generation_list) % 1 == 0:
                if not self.silent:
                    print('Generation: {} Fitness: {}'.format(
                        len(self.generation_list), best_child_fitness))

                if self.auto_plot:
                    # Plays through data in a pygame window

                    Viewer.Playback(f.fitness_data_array,
                        text='Generation: {}  Fitness: {}'.\
                        format(len(self.generation_list),best_child_fitness),
                        dna=f.world.world).run()

            # Update to the next generation
            self.update_pop()

            # If the max_epochs is met, then stop.
            if self.max_epochs != -1:
                if len(self.generation_list) > self.max_epochs:
                    break
        return self

class ProbSpace():
    '''
    Builds a LIST of each LIST of problem spaces in the
    shuffle split bag to be used by cellular automata.

    Expects:
            X np.ndarray int32|int64|float64    training data features
            y np.ndarray int64|int32            training data labels
            training BOOL (default: True)       if False, return single space
            quantize_size TUPLE                 x,y dim of the problem space
            class_weight STR                    "auto" preserves strafication
                                                of classes in population
            splits
    '''

    def __init__(self,
                 X=np.array([]),
                 y=np.array([]),
                 training=True,
                 quantize_size=(15,15), # INHERIT THIS ONCE FIT
                 class_weight='auto',
                 splits=None):

        assert isinstance(X, np.ndarray),\
            'Expected X type np.array, got: {}'.format(type(X))
        assert X.shape[0] >= 1, \
            'Expected X to have at least 1 sample.'
        assert training in [True, False], \
            'Expected training to be type BOOL'
        if X.dtype not in ['int32', 'int64', 'float32', 'float64']:
            raise TypeError('X is incorrect dtype')
        assert isinstance(quantize_size,tuple),\
            'Expected quantize_size to be tuple. e.g. (15,15)'
        assert len(quantize_size) == 2,\
            'Expected quantize_size to have len 2, got: {}'\
                .format(len(quantize_size))

        if training:
            assert isinstance(y, np.ndarray), \
                'Expected y type np.adarry, got: {}'.format(type(y))
            assert y.shape[0] >= 1, 'y was an empty array'
            assert y.shape[1] == 1, 'y must use the shape ({},1)'.format(y.shape[1])
            if y.dtype not in ['int32', 'int64']:
                raise TypeError('y must be type INT, got TYPE {}'.format(y.dtype))
            assert len(np.unique(y)) > 1, \
                'There was a single class in y. Expected more than 1, ({})' \
                    .format(np.unique(y))
            assert class_weight in ['balanced', 'auto'], \
                'Expected class weight to be in ["balanced","auto"], got {}' \
                    .format(class_weight)
            assert isinstance(splits, int), \
                'Expected splits INT to be set, got: {}'.format(splits)
            assert splits > 0, \
                'Expected number of splits to be greater than 0'
            assert X.shape[0] == y.shape[0], \
                'X and y have incompatible shapes {},{}'.format(X.shape, y.shape)


        self.X = X
        self.y = y
        self.training = training
        self.quantize_size = quantize_size
        self.class_weight = class_weight
        self.splits = splits

    def generate_space(self,
                       X=np.array([]),
                       y=None):
        '''
        Expects:
            X np.ndarray        a single X sample
            y                   is ignored
        Returns a LIST of problem spaces for all X samples.
        '''

        # X,y are assumed to have a shape of (,1)
        assert len(X.shape) == 1,\
            'Expected single sample of X, got dimensions {}'.format(X.shape)

        numFeatures = X.shape[0]
        width = (numFeatures * 4) + 3

        # Initialize 2d space
        d = [np.zeros((width,self.quantize_size[1], 3)) for i in range(4)]

        # Set starting whites
        # Layer 0

        for i in range(numFeatures):
            xPos = int(i * 4)-1
            d[0][3+int(i *4)-1][0] =     (255, 255, 255)
            d[0][3+int(i * 4)][0] =      (255, 255, 255)
            d[0][3 + int(i * 4)+1][0] =  (255, 255, 255)
            d[0][3+int(i * 4) - 1][-1] = (255, 255, 255)
            d[0][3+int(i * 4)][-1] =     (255, 255, 255)
            d[0][3 + int(i * 4)+1][-1] = (255, 255, 255)
            d[0][3 + int(i * 4) - 1][-2] = (255, 255, 255)
            d[0][3 + int(i * 4)][-2] = (255, 255, 255)
            d[0][3 + int(i * 4) + 1][-2] = (255, 255, 255)

        # Layer 1
        # for i in range(7):
        #     for ii in range(5):
        #         d[1][0 + i][0 + ii] = (255, 255, 255)
        for i in range(numFeatures):
            d[1][3 + int(i * 4) - 1][1] = (255, 255, 255)
            d[1][3 + int(i * 4)][1] = (255, 255, 255)
            d[1][3 + int(i * 4) + 1][1] = (255, 255, 255)
            d[1][3 + int(i * 4) - 1][0] = (255, 255, 255)
            d[1][3 + int(i * 4)][0] = (255, 255, 255)
            d[1][3 + int(i * 4) + 1][0] = (255, 255, 255)
            d[1][3 + int(i * 4) - 1][-1] = (255, 255, 255)
            d[1][3 + int(i * 4)][-1] = (255, 255, 255)
            d[1][3 + int(i * 4) + 1][-1] = (255, 255, 255)


        # Layer 2
        for i in range(4):
            for ii in range(5):
                d[2][0 + i][0 + ii] = (255, 255, 255)

        # Layer 3
        for i in range(5):
            for ii in range(7):
                d[3][0 + i][0 + ii] = (255, 255, 255)

        # Assert that there are 4 layers
        assert len(d) == 4, 'Expected 4 layers in d'

        # For each feature, attempt to fill layer 0
        for i in range(numFeatures):
            xPos = int(i * 4)
            # If value is outside problem space range (i.e.
            # value was not seen in training data), set to
            # max value of y-axis self.quantize_size.
            if X[i] > (self.quantize_size[1] - 3):
                d[0][xPos+3,2:int(self.quantize_size[1])] = (0, 255, 0)
            # or min value
            elif X[i] < 0:
                d[0][xPos+3,0:3] = (0, 255, 0)
            else:
                d[0][xPos+3,3:int(3+X[i])] = (0, 255, 0)

        return d

    def stratified_training_bags(self,
                                 X=np.array([]),
                                 y=np.array([])):
        '''

        Creates bags of training data which transform X,y into
        the expected {'space','label'} format used in the Fitness()
        and CellularAutomata() classes.

        Works best when there are at least 100 samples per 2 classes

        Expects:
            X np.ndarray int32|int64|float64    training data features
            y np.ndarray int64|int32            training data labels

        Returns:
            bags LIST           has a shape (max_epochs, X.shape[0],2)

        bags format:
        [bag_1 LIST, bag_2 LIST, ... bag_n LIST]

        bag_n format:
        {'space': np.ndarray, 'label': np.ndarray}
        '''

        bags = []

        if self.splits:
            for i in range(self.splits):
                training_bag = []
                if self.class_weight == 'auto':
                    # Use class weights to determine how samples are stratified.
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.30, random_state=i, stratify=y)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.30, random_state=i)


                for ii in range(X_test.shape[0]):
                    space = self.generate_space(X_test[ii])
                    assert isinstance(space,list)
                    training_bag.append({'space':space,
                                      'label':y_test[ii]})
                bags.append(training_bag)

            assert bags, 'training bags were empty'
        else:
            for i in range(X.shape[0]):
                space = self.generate_space(X[i])
                bags.append({'space':space,'label':y})
            bags = [bags]
        if not self.training:
            assert bags, 'space was not generated for X'

        return bags


class Fitness():
    '''
    Defines an viewer object which is able to move through the world.

    Expects:
        world np.ndarray                Containing the DNA of the current child
        json_data   OBJECT              Containing some of the metadata used in model (e.g. color maps)
        gen INT                         The generation of the current child
        data LIST                       A LIST of samples in the
                                        DICT form {'space',LIST,'label',np.ndarray}
        fitness_func                    STR accuracy | accuracy_proba for determining fitness
    '''

    def __init__(self, world,
                 json_data=None,
                 gen=None,
                 data=[],
                 y_classes=[],
                 fitness_func='accuracy'):

        try:
            self.world = copy.deepcopy(world)
        except NotImplementedError:
            self.world = copy.copy(world)

        assert json_data, 'Requires valid json_data for running CellularA()'
        assert isinstance(data,list),\
            ' Expected data to be type LIST, got {}'.format(type(data))
        assert data, 'data was empty'
        assert 'space' in data[0]
        assert y_classes, \
            'y_classes must be set in Fitness()'
        assert isinstance(fitness_func,str),\
            'Expected fitness_func to be type STR, got type {}'.format(type(fitness_func))
        assert fitness_func in ['accuracy','accuracy_proba'],\
            'Did not recognize fitness_func={}'.format(fitness_func)

        self.json_data = json_data
        self.gen=gen # used for randomized fitness variable
        self.data = data
        self.y_classes = y_classes

        # This value should be the height of the data space
        self.fitness_value_init = 0
        self.fitness_func = fitness_func


    def run(self):

        data = self.data
        batch_fitness = 0
        res_array = [] # Contains res_data_arrays
        for m in range(len(data)):

            self.fitness_value = self.fitness_value_init

            # data[m]['space'] == equiv to d
            # data[m]['label'] == answer

            # Define y_true label (and multiply by 10
            y_true = data[m]['label'][0]
            assert isinstance(y_true,(int,np.int64,np.int32)),\
                'Expected t_true label INT, but got TYPE: {}'.format(type(y_true))

            # Run the Cellular Automata simulation
            res = CellularA(data=copy.deepcopy(data[m]['space']),
                            dna=copy.deepcopy(self.world.world),
                            json_data=self.json_data).run()

            # Determine estimate with percentage of class-associate cells
            color_classes = self.json_data.color_classes

            # Initialize empty sums for each possible y class
            y_pred_sums = np.empty((len(self.y_classes)))

            # Sum the guess density for each class
            for i in range(len(self.y_classes)):

                y_pred_sums[i] = np.sum([np.sum((res['data_array'][-1][ii].reshape(-1, 3).astype(int) \
                                                 == np.array(color_classes[i]).astype(int)).all(axis=1)) for ii in
                                         range(4)])

                # Determine the percentage of guess density for each possible y class
            y_pred_proba = y_pred_sums.astype(float) / y_pred_sums.astype(float).sum()
            y_pred_proba = np.array([x if str(x)!='nan' else 0.01 for x in y_pred_proba])

            assert y_pred_sums.shape == y_pred_proba.shape,\
                'y_pred_sums.shape != y_pred_proba.shape, ({}{})'\
                .format(y_pred_sums.shape,y_pred_proba.shape)


            # Calculate residual as the squared difference
            # in true probability (1) and pred_proba for the correct label
            y_pred = self.y_classes[np.argmax(y_pred_proba)]
            squared_residual = (1- y_pred_proba[y_true])**2

            # Using correctness (because CA is a bad estimator)
            if self.fitness_func == 'accuracy':
                if y_pred == y_true:
                    how_right = 1
                else:
                    how_right = 0
            # Using pre_proba residual squares as cost function
            elif self.fitness_func == 'accuracy_proba':
                how_right = 1- squared_residual

            # Update fitness
            self.fitness_value += how_right
            if self.fitness_value < 0:
                self.fitness_value =0

            res_array.append(res['data_array'])
            batch_fitness+= self.fitness_value

        batch_fitness = batch_fitness /float(len(data))

        self.fitness_data_array = random.choice(res_array)
        self.fitness_value = batch_fitness

        return self


class World():
    '''
    Defines an array of line segments which represent square primatives
    for defining world collision surfaces.
    '''

    def __init__(self, x_lim, y_lim,y_classes=[]):

        self.hello = 'world'
        assert isinstance(x_lim, int), 'x_lim must be an INT'
        assert isinstance(y_lim, int), 'y_lim must be an INT'
        self.x_bounds = (0, x_lim)
        self.y_bounds = (0, y_lim)
        self.finished_world = False
        self.y_classes=y_classes
        assert isinstance(y_classes,list)

    def gen_world(self):
        '''
        Initializes a random numpy array
        '''

        # Casting this to astype(int) may cause issues
        self.world = np.random.randint(low=0,high=2,size=(self.x_bounds[1], self.y_bounds[1]))

        # Bit detection 9-way (0) or 1-way on 9-bit grid (1-9)
        self.world[:, 0:1] = np.random.randint(low=0, high=10, size=(self.x_bounds[1], 1))

        # Add colors for each potential class
        # (Less likely to find reliable solution with more classes)
        # Action color
        self.world[:, 10:11] = np.random.randint(low=0, high=len(self.y_classes) + 3, size=(self.x_bounds[1], 1))

        # Target color ( White or green)
        self.world[:, 11:12] = np.random.randint(low=0, high=len(self.y_classes) + 3, size=(self.x_bounds[1], 1))


        # Action Layer
        self.world[:, 12:13] = np.random.randint(low=0,high=4,size=(self.x_bounds[1],1))
        # Target Layer (layer 0)
        self.world[:, 13:14] = np.random.randint(low=0, high=4, size=(self.x_bounds[1], 1))

        self.world = self.world.astype(int)

        return self


class TestGA(unittest.TestCase):


    def test_preprocess(self):

        # Correct INTs
        X = np.array([[1,2,3,4],
                      [5,6,7,8],
                      [9,10,11,12],
                      [13,14,15,16]])
        y = np.array([1,2,3,4]).reshape(4,1)
        model = CellularModel(quantize_size=(15,15))
        X,y = model.preprocess(X,y)

        expected_X = np.array([[ 2.,2.,  2.,  2.],
                     [ 5.,  5.,  5.,  5.],
                     [ 9.,  9.,  9.,  9.],
                     [13., 12., 13., 13.]])
        expected_y = np.array([[ 1],
                     [ 2],
                     [ 3],
                     [4]])
        assert (X == expected_X).all()
        assert (y == expected_y).all()

        # Incorrect y shape
        X = np.random.randint(low=0, high=2, size=(4, 4))
        y = np.array([1,0,1,0]).reshape(1,4)
        model = CellularModel()
        with self.assertRaises(AssertionError):
            model.preprocess(X,y)


        # Empty data
        X = np.array([])
        y = np.random.randint(low=0, high=2, size=(4, 1))
        model = CellularModel()
        with self.assertRaises(AssertionError):
            model.preprocess(X, y)


        # Empty data
        X = np.random.randint(low=0, high=2, size=(4, 4))
        y = np.array([])
        model = CellularModel()
        with self.assertRaises(AssertionError):
            model.preprocess(X, y)


        # Wrong datatype
        X = np.random.randint(low=0, high=2, size=(4, 4)).astype(bool)
        y = np.array([1,0,1,0]).reshape(4,1)
        model = CellularModel()
        with self.assertRaises(TypeError):
            model.preprocess(X, y)

        # Wrong quantizesize
        X = np.random.randint(low=0, high=2, size=(4, 4))
        y = np.array([1,0,1,0]).reshape(4,1)
        with self.assertRaises(AssertionError):
            model = CellularModel(quantize_size=(0,2))


        # applying sclaer before it is fit
        X = np.random.randint(low=0, high=2, size=(4, 4))
        y = np.array([1,0,1,0]).reshape(4,1)
        model = CellularModel()
        with self.assertRaises(AssertionError):
            model.preprocess(X,y,training=False)


        # y has a single class
        X = np.random.randint(low=0, high=2, size=(4, 4))
        y = np.array([0,0,0,0]).reshape(4,1)
        model = CellularModel()
        with self.assertRaises(AssertionError):
            model.preprocess(X,y)

    def test_probspace(self):
        # Correct values - 2 bags
        X = np.random.randint(low=0, high=2, size=(16, 4))
        y = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]).reshape(16,1)
        p = ProbSpace(X,y,quantize_size=(15,15),
                      splits=2,training=True)
        space = p.generate_space(X[0],y[0])
        bags = p.stratified_training_bags(X,y)
        assert len(bags) ==2

        # Correct values - 6 bags
        X = np.random.randint(low=0, high=2, size=(16, 4))
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]).reshape(16, 1)
        p = ProbSpace(X, y, quantize_size=(15, 15),
                      splits=6, training=True)
        bags = p.stratified_training_bags(X, y)
        assert len(bags) == 6

        # Correct values - 1 bag
        X = np.random.randint(low=0, high=2, size=(16, 4))
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]).reshape(16, 1)
        p = ProbSpace(X, y, quantize_size=(15, 15),
                      splits=1, training=True)
        bags = p.stratified_training_bags(X, y)
        assert len(bags) == 1

        # Correct values - 1 bag training=False
        X = np.random.randint(low=0, high=4, size=(16, 4))
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]).reshape(16, 1)
        p = ProbSpace(X, y, quantize_size=(15, 15),
                      splits=0, training=False)
        bags = p.stratified_training_bags(X)
        assert len(bags) == 1

        # Correct values - bagging multiple samples while training=False
        X = np.random.randint(low=0, high=2, size=(16, 4))
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]).reshape(16, 1)
        p = ProbSpace(X, y, quantize_size=(15, 15),
                      splits=0, training=False)
        bags = p.stratified_training_bags(X, y)

        # Correct values, all samples instead of single
        X = np.random.randint(low=0, high=2, size=(4, 4))
        y = np.array([1,0,1,0]).reshape(4,1)
        p = ProbSpace(X, y, quantize_size=(15, 15),splits=2)
        with self.assertRaises(AssertionError):
            space = p.generate_space(X, y)

        # Incorrect training param
        X = np.random.randint(low=0, high=2, size=(4, 4))
        y = np.array([1,0,1,0]).reshape(4,1)
        with self.assertRaises(AssertionError):
            ProbSpace(X, y, training=3,splits=2)

        # Incorrect class_weight param
        X = np.random.randint(low=0, high=2, size=(4, 4))
        y = np.array([1, 0, 1, 0]).reshape(4, 1)
        with self.assertRaises(AssertionError):
            ProbSpace(X, y, class_weight='eggs',splits=2)

        # Incorrect class_weight param
        X = np.random.randint(low=0, high=2, size=(4, 4))
        y = np.array([1, 0, 1, 0]).reshape(4, 1)
        with self.assertRaises(AssertionError):
            ProbSpace(X, y, splits=0)

    def test_fit_predict_simple(self):

        # Correct values
        from sklearn.datasets import load_wine
        X = np.array([
            [0, 1,0,0],[0, 1,0,0],[0, 1,0,0],
            [1, 0,0,0],[1, 0,0,0],[1, 0,0,0],
            [0,0,1,0],[0,0,1,0],[0,0,1,0],
            [0, 0, 0, 1],[0, 0, 0, 1],[0, 0, 0, 1]
        ])
        y = [0,0,0,1,1,1,2,2,2,3,3,3]
        y = np.array(y).reshape(len(y), 1)
        model = CellularModel(N=3,
                            mutate_proba=0.005,
                            max_epochs=1,
                            silent=True,
                            auto_plot=True,
                            min_depth=1,
                            max_depth=1,
                            fitness_func='accuracy')
        model.fit(X, y,silent=True)

        try:
            y_pred = model.predict(X,plot=True)
            from sklearn.metrics import f1_score
            f1 = f1_score(y, y_pred, average='macro')
        except:
            raise AssertionError('To use .predict(plot=True), pip install pygame')
        model.save('examples/example_model.pkl')



if __name__ == '__main__':
    unittest.main()