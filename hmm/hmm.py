# Using HMM to stimulate the movement of robot in grid world
import math
import collections

class grid:
    def __init__(self, file, grid_size, no_towers):
        self.matrix = [[0 for item in range(grid_size)] for elem in range(grid_size)]
        self.free = []
        self.distances = []
        self.grid_size = grid_size
        self.no_of_towers = no_towers
        self.tower_loc = []
        self.prob_states = []
        self.file = file
        self.free_cells()
        self.tower_locations()
        self.tower_distances()
        self.possible_states()
        self.calculate_transition_probability()
        self.viterbi()

    def free_cells(self):
        index = 0
        with open(self.file) as f:
            store = False
            for lines in f:
               lines = lines.strip()
               if not store and lines == 'Grid-World:':
                    store = True
                    continue
               if not store or lines == '':
                     continue
               elements = lines.split()
               for ind, item in enumerate(elements):
                  self.matrix[index][ind] = int(item)
                  if item == '1':
                      self.free.append([index, ind])
               if len(elements) > 0:
                  index += 1
               if index == 10:
                   break

    def tower_locations(self):
        with open(self.file) as f:
            store = False
            for lines in f:
                lines = lines.strip()
                if not store and lines == 'Tower Locations:':
                    store = True
                    continue
                if not store or lines == '':
                    continue
                elements = lines.split(':')
                elements = elements[1].split()
                self.tower_loc.append([int(elements[0]), int(elements[1])])
                if len(self.tower_loc) == self.no_of_towers: # no of towers
                    break

    def tower_distances(self):
        with open(self.file) as f:
            store = False
            for lines in f:
                lines = lines.strip()
                if not store and lines.startswith('Noisy Distances'):
                    store = True
                    continue
                if not store or lines == '':
                    continue
                elements = lines.split()
                #print(elements)
                temp = []
                for item in elements:
                    temp.append(float(item))
                self.distances.append(temp)

    def distance_to_each_tower(self):
        l2 = lambda x, y : math.sqrt(pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2))
        self.distance_range = []
        for i, item in enumerate(self.free):
            temp = []
            for j, loc in enumerate(self.tower_loc):
                dist = l2(item, loc)
                temp.append([.7 * dist, 1.3 * dist])
            self.distance_range.append(temp)
        assert len(self.distance_range) == len(self.free)

    def possible_states(self):
        self.distance_to_each_tower()
        for distance in self.distances:
            temp = []
            for index, points in enumerate(self.free):
                flag = True
                for ind, elem in enumerate(distance):
                    # Check if the given distance is in the possible distance range if so the point is possible for that time stamp
                    if self.distance_range[index][ind][0] <= elem <= self.distance_range[index][ind][1] :
                        continue
                    else:
                        flag = False
                        break
                if flag:
                    temp.append(points)
            assert temp != []
            self.prob_states.append(temp)
        assert len(self.prob_states) == len(self.distances)
        print(self.prob_states)

    @staticmethod
    def find_neighbours(location, grid_size):
        x = location[0]
        y = location[1]
        temp = []

        if x + 1 < grid_size:
            temp.append((x + 1, y))
        if y + 1 < grid_size:
            temp.append((x, y + 1))
        if x - 1 > 0:
            temp.append((x-1, y))
        if y - 1 > 0:
            temp.append((x, y - 1))

        return temp

    def calculate_transition_probability(self):
        d = collections.defaultdict(list)
        for index, item in enumerate(self.prob_states):
            for elem in item:
                d[tuple(elem)].append(index)

        self.prob_dic = collections.defaultdict(dict)

        for item in d:
            indices = d[item]
            neighbours = self.find_neighbours(item, self.grid_size)
            for ind in indices:
                ind += 1
                for elem in neighbours:
                    if elem in d:
                        if ind in d[elem]:
                            if elem not in self.prob_dic[item]:
                                self.prob_dic[item][elem] = 0
                            self.prob_dic[item][elem] += 1
                            if 'total' not in self.prob_dic[item]:
                                self.prob_dic[item]['total'] = 0
                            self.prob_dic[item]['total'] += 1

        for item in self.prob_dic:
            for elem in self.prob_dic[item]:
                if elem != 'total':
                    self.prob_dic[item][elem] /= self.prob_dic[item]['total']
        print(self.prob_dic)

    def viterbi(self):
        # This finds out the most probable path
        level = 0
        dic = {}
        # Add all the initial start points
        dic[level] = {}
        for item in self.prob_states[level]:
            item = tuple(item)
            dic[level][item] = {}
            dic[level][item]['parent'] = ''
            dic[level][item]['prob'] = 1
        for level in range(1, 11):
            dic[level] = {}
            for items in dic[level - 1]:
                # Add all the neighbours of items as the next state if that is seen in the training data
                if items in self.prob_dic:
                    for contends in self.prob_dic[items]:
                        contends_list = list(contends)
                        if contends_list in self.prob_states[level]:
                            if contends not in dic[level]:
                                dic[level][contends] = {}
                                dic[level][contends]['parent'] = items
                                present_prob = dic[level - 1][items]['prob'] * self.prob_dic[items][contends]
                                dic[level][contends]['prob'] = present_prob
                            else:
                                present_prob = dic[level - 1][items]['prob'] * self.prob_dic[items][contends]
                                if present_prob > dic[level][contends]['prob']:
                                    dic[level][contends]['parent'] = items
                                    dic[level][contends]['prob'] = present_prob

        max_prob = -1
        final = ''
        level = 10
        for item in dic[level]:
            if max_prob < dic[level][item]['prob']:
                max_prob = dic[level][item]['prob']
                final = item

        path = []
        path.append(final)
        while True:
            if dic[level][final]['parent'] == '':
                break
            new_final = dic[level][final]['parent']
            path.append(new_final)
            level -= 1
            final = new_final

        print('The path is ', path[::-1])


