'''
Self-Organizing Maps
Last Modify: 07-07-2023
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from random import random, seed
from time import time

#import sys
#sys.setrecursionlimit(999)

def adjust_clusters(clusters:list):
    """
    Adjusts the excess clustering number
    """
    clusters_new = []
    c_ = {}
    c_sort = sorted(clusters)
    i = 0
    for cl in c_sort:
        if not cl in c_:
            c_[cl] = i
            i += 1

    for i in range(len(clusters)):
        clusters_new.append(c_[clusters[i]])
    return clusters_new

def transpose(data):
    return [[data[i][j] for i in range(len(data))] for j in range(len(data[0]))]

def variance_distance(d):
    mean = [0 for i in range(len(d[0]))]
    for i in range(len(d)):
        for j in range(len(d[0])):
            mean[j] += d[i][j]

    var = [0 for i in range(len(d[0]))]
    for i in range(len(d)):
        for j in range(len(d[0])):
            var[j] += (mean[j] - d[i][j])**2

    l = len(d)
    for i in range(len(var)):
        var[i] /= l

    distance = 0
    for k in range(len(var)):
        distance += var[k]

    return distance**(1/2)
            
def rotate45(matrix):
    if type(matrix) == list:
        matrix = np.array(matrix)

    #Obter a quantidade de dimensões da matriz
    n_dim = len(matrix[0])

    r = (2**(1/2))/2
    k = 2
    for i in range(n_dim-k):
        n = np.identity(n_dim)
        n[i][i] = r
        n[i][i+k] = r
        n[i+k][i] = -r
        n[i+k][i+k] = r
        matrix = matrix.dot(n)

        n = np.identity(n_dim)
        n[i+k-1][i+k-1] = r
        n[i+k-1][i+k] = -r
        n[i+k][i+k-1] = r
        n[i+k][i+k] = r
        matrix = matrix.dot(n)

##        n = np.identity(n_dim)
##        n[i][i] = r
##        n[i][i+k-1] = -r
##        n[i+k-1][i] = r
##        n[i+k-1][i+k-1] = r
##        matrix = matrix.dot(n)
        

##    num_dimensions = matrix.ndim
##    permutation = [0, 1] + list(range(2, num_dimensions))  # Permutação das dimensões
##
##    rotated_matrix = np.transpose(matrix, permutation)
##    return rotated_matrix

    matrix = pd.DataFrame(matrix)
    matrix = matrix.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    return matrix.values.tolist() #rotate.tolist()

def rotate():
    n = globals()["number_of_neurons_"]
    
    rot = []
    for i in range(n):
        for j in range(n):
            rot.append(globals()[f"n{i}_{j}"].weights)
            #print(globals()[f"n{i}_{j}"].weights)

    #print(rot)
    rot = rotate45(rot)
    #print(len(rot))

    k = -1
    for i in range(n):
        for j in range(n):
            k += 1
            #print(k, globals()[f"n{i}_{j}"].weights)
            globals()[f"n{i}_{j}"].weights = rot[k]
            #print(globals()[f"n{i}_{j}"].weights)

    #print(rot)
    return

def adjust(data):
    q_1 = sorted(data)[int(len(data)*1/4)]
    q_2 = sorted(data)[int(len(data)*2/4)]
    q_3 = sorted(data)[int(len(data)*3/4)]
    var = (q_1 - 0.25)**2 + (q_2 - 0.5)**2 + (q_3 - 0.75)**2

    p = 1.01
    var_new = (q_1**p - 0.25)**2 + (q_2**p - 0.5)**2 + (q_3**p - 0.75)**2
    while var_new < var: # p > 1
        var = var_new
        p += 0.01
        var_new = (q_1**p - 0.25)**2 + (q_2**p - 0.5)**2 + (q_3**p - 0.75)**2
        if var_new > var:
            for i in range(len(data)):
                data[i] = data[i]**p
            return data
    p = 0.99
    var_new = (q_1**p - 0.25)**2 + (q_2**p - 0.5)**2 + (q_3**p - 0.75)**2
    while var_new < var: # p < 1
        var = var_new
        p *= 0.99
        var_new = (q_1**p - 0.25)**2 + (q_2**p - 0.5)**2 + (q_3**p - 0.75)**2
        if var_new > var:
            for i in range(len(data)):
                data[i] = data[i]**p
            return data

##    if not (q_1 < 0.25 and q_2 < 0.5 and q_1 < 0.75) or (q_1 > 0.25 and q_2 > 0.5 and q_1 > 0.75):
##        return data
##
##    v = 0.5
##    p = 1.01
##    n = 0.001
##    while True:
##        #print(v)
##        try:
##            q_1 = sorted(data)[int(len(data)*1/4)]** (1+v)
##        except:
##            q_1 = 0
##        try:
##            q_2 = sorted(data)[int(len(data)*2/4)]** (1+v)
##        except ZeroDivisionError:
##            q_2 = 0
##        try:
##            q_3 = sorted(data)[int(len(data)*3/4)]** (1+v)
##        except ZeroDivisionError:
##            q_3 = 0
##        var_new = (q_1 - 0.25)**2 + (q_2 - 0.5)**2 + (q_3 - 0.75)**2
##
##        #print(var_new, var)
##
##        if var_new <= var and var_new > n:
##            #print("Case 1")
##            v = v*p
##            var = var_new
##        elif var_new > var and var_new > n:
##            #print("Case 2")
##            v = -v
##            p = p**(1/2)
##            n += 0.0001
##
##        else:      
##            #print("Case 3")
##            for i in range(len(data)):
##                try:
##                    data[i] = data[i] ** (1+v)
##                except ZeroDivisionError:
##                    pass
##            return data

def normalize(data):
    import pandas as pd
    #print(data)
    data = pd.DataFrame(data)
    data = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    #print(data)
##    for i in range(len(data)):
##        for j in range(len(data[i])):
##            data[i][j] = (data[i][j] - min(data[i])) / (max(data[i]) - min(data[j]))

    data = data.values.tolist()

    data = transpose(data)
    
    for i in range(len(data)):
        print(f"Adjust column {i}")
        data[i] = adjust(data[i])

    return transpose(data)


class Neuron:
    """
    Base object for the functioning of the SOM, it is a neuron

    Example of use:
    1) SOM = create_SOM(20, learning = 0.05)
    2) n1_1.design_weights(dados)
    3) n0_0.auto_organizing(epochs = 10, print_ = True)
    4) n1_1.valley(normalize = True, potency = 1)
    5) n1_1.amount_of_wins()    
    """
    def __init__(self, learning = 0.01):
        self.learning = learning
        self.weights = None
        #self.limits = [None, None]

        #Connections
        self.con_r = None
        self.con_l = None
        self.con_u = None
        self.con_d = None

        #Amount of data the neuron gains
        self.number_of_wins = 0

    def connect(self, obj_r = None, obj_l = None, obj_u = None, obj_d = None):
        '''
        Connects the neurons.
        '''

        if obj_r != None:
            self.con_r = obj_r
            obj_r.con_l = self

        if obj_l != None:
            self.con_l = obj_l
            obj_l.con_r = self

        if obj_u != None:
            self.con_u = obj_u
            obj_u.con_d = self

        if obj_d != None:
            self.con_d = obj_d
            obj_d.con_u = self

    def network_size(self):
        '''
        Counts the size of the grid of neurons
        '''
        k = self
        while True:
            if k.con_l != None:
                k = k.con_l
            elif k.con_d != None:
                k = k.con_d
            else:
                break

        n = [1,1]
        while True:
            #print(k.name(),"<<<")
            if k.con_r != None:
                k = k.con_r
                n[0] += 1
                #print("->")
            elif k.con_u != None:
                k = k.con_u
                n[1] += 1
                #print("vv")

            else:
                return n

    def distance(self, obs:list):
        """
        Calculates the distance
        """
        sum_ = 0
        for i in range(len(obs)):
            try:
                sum_ += (self.weights[i] - obs[i])**2
            except Exception as error:
                print(error)
                print(self.name())
                input("Pause...")
        return sum_**(1/2)        

    def auto_organizing(self, data:list = None, epochs:int = 1, print_:bool = False):
        """
        Self-denses, closest data pulls winning neuron, tests for one die at a time.
        
        :param data: dataframe with the variables to be considered in the process
        :type data: dataframe, optional

        :param epochs: number of interactions
        :type epochs: int

        :param print_: whether the images will be printed or not
        :type print_: bool

        ▬ return:
        • None
        """

        def time_count(t):
            if t < 60:
                return f"{str(t)[:6]} seconds to finish..."
            elif t < 60 * 60:
                tm = [int(t/60), t % 60]
                return f"{str(tm[0])} minutes and {str(tm[1])[:6]} seconds to finish..."
            else:
                tm = [int(t/(60*60)), int(t - int(t/(60*60)*60*60)/60), t % 60]
                return f"{str(tm[0])} hours, {str(tm[1])} minutes and {str(tm[2])[:6]} seconds to finish..."
        
        if not "global_som_data" in globals():
            print("\nPlease prepare the SOM for your data: n1_1.design_weights(YOUR_DATA)\n")
            return None
        if "global_som_data" in globals():
            if data == None and globals()["global_som_data"] != None:
                #print("No data with global data")
                data = globals()["global_som_data"]

        t0 = time()
        t_ = time()
        for i in range(epochs):
            if print_:
                try:
                    if i % int(epochs/200) == 0:
                        print(str(i*100/epochs)[:4] + "%")
                except:
                    pass
                try:
                    if i % int(epochs/100) == 0:
                        t1 = time()
                        if t1-t_ > 10:
                            t = (t1-t0)*(epochs/i - 1)
                            print(time_count(t))
                            t_ = time()
                except:
                    pass
            if not "var_data" in globals():
                globals()["var_data"] = variance_distance(data)

            #Descubro o neuronio mais perto desse dados:
            root_true = self
            while True:
                if root_true.con_l != None:
                    root_true = root_true.con_l
                elif root_true.con_d != None:
                    root_true = root_true.con_d
                else:
                    break

            
            for observation in data:
                min_ = root_true
                new_ = root_true
                #new_.number_of_wins = 0
                root = root_true
                min_len = min_.distance(observation)
                
                #Anda por todos neuronios:
                run = True
                while run:
                    #Descobre se é o mínimo:
                    new_dist = new_.distance(observation)
                    new_.number_od_wins = 0
                    if min_len > new_dist: #Winner
                        #print(new_.name(), "-->", min_.name())
                        min_ = new_
                        min_len = new_dist

                    #Anda por todos neuronios                    
                    if new_.con_r != None:
                        new_ = new_.con_r
                    else:
                        if root.con_u != None:
                            new_ = root.con_u
                            root = new_
                        else:
                            run = False

                #print("Dado ->", observation,"\nNeuronio Vencedor ->", min_.name(), "->", min_.weights)

                #Stirring the neuron and its neighbors
                min_.propagation(observation)
                min_.number_of_wins += 1
        
        return

    def attraction(self, move, obs, n):
        """
        Calculates the attraction of neighbors
        """
        s = self
        
        try:
            for i in move:
                s = s @ i

            for k in range(len(obs)):
                s.weights[k] -= (s.weights[k] - obs[k])*s.learning/n
                
        except AttributeError:
            pass

        except TypeError: #If self.direction == None
            pass

    def propagation(self, obs:list):
        '''
        Propagates the sort iteration
        '''
        n = 32
        
        change = []
        for i in range(len(obs)):
            change.append(self.weights[i] - obs[i])

        for k in range(len(change)):
            self.weights[k] -= change[k]*self.learning

        #interation 1
        i1 = n
        self.attraction(["->"],obs, i1)

        self.attraction(["<-"],obs, i1)

        self.attraction(["^"],obs, i1)

        self.attraction(["v"],obs, i1)

        #interation 2
        i2 = n**(3/2)
        self.attraction(["->","v"],obs, i2)

        self.attraction(["->","^"],obs, i2)

        self.attraction(["<-","v"],obs, i2)

        self.attraction(["<-","^"],obs, i2)

        self.attraction(["->","->"],obs, i2)

        self.attraction(["<-","<-"],obs, i2)

        self.attraction(["v","v"],obs, i2)
        
        self.attraction(["^","^"],obs, i2)

        #Interation 3
        i3 = n**(5/2)
        self.attraction(["->","->","^"],obs, i3)

        self.attraction(["->","->","v"],obs, i3)

        self.attraction(["^","^","->"],obs, i3)

        self.attraction(["^","^","<-"],obs, i3)

        self.attraction(["<-","<-","^"],obs, i3)

        self.attraction(["<-","<-","v"],obs, i3)

        self.attraction(["v","v","->"],obs, i3)

        self.attraction(["v","v","<-"],obs, i3)

    def design_weights(self, data):
        '''
        Creates a grid with the network, the grid is evenly divided,
        the weights of the neurons are their positions
        '''

        data = normalize(data)

        globals()["global_som_data"] = data

        n_len, n_hig = self.network_size()
        
        w_min = []
        w_max = []
        data = transpose(data)
        for i in range(len(data)):
            w_min.append(min(data[i]))
            w_max.append(max(data[i]))

        positions = {}
        for k in range(len(w_min)):
            positions[k] = []
            for i in range(n_len):
                positions[k].append((w_max[k]-w_min[k])*1/(n_len-1)*i + w_min[k])
        #print(positions,"\n\n",w_max)
        
        # Add weights
        k = self
        while True:
            if k.con_l != None:
                k = k.con_l
            elif k.con_d != None:
                k = k.con_d
            else:
                break

        root = self
        k = self
        h = [0, 0]
        h.extend([-1 for i in range(len(list(positions))-2)])
        pos = list(positions)
        #print(positions)
        while True:
            print("Add position of", "n"+str(h[0])+"_"+str(h[1]))
            #print(positions, "----------", h, "- _ - _ - _ -", i, ">>>>>>>>>>>>>>>", pos)                

            globals()["n"+str(h[0])+"_"+str(h[1])].weights = [positions[i][h[i]] for i in pos]

            if k.con_r != None:
                k = k.con_r
                h[1] += 1

            else:
                h[1] += 1
                print("Add position of", "n"+str(h[0])+"_"+str(h[1]))
                globals()["n"+str(h[0])+"_"+str(h[1])].weights = [positions[i][h[i]] for i in pos]

##                for i in range(2, len(h)):
##                    if random() > .5:
##                        h[i] += 1
##                        #print(h[i],len(positions[0]) - 1)
##                        h[i] = min(h[i],len(positions[i])-1)

                if root.con_u != None:
                    k = root.con_u
                    root = k
                    h[0] += 1
                    h[1] = 0
                    
                else:
                    h[0] += 1
                    h[1] = 0
                    while True:
                        print("Add position of", "n"+str(h[0])+"_"+str(h[1]))
                        globals()["n"+str(h[0])+"_"+str(h[1])].weights = [positions[i][h[i]] for i in pos]
                        if root.con_d != None:
                            root = root.con_d
                            h[1] += 1
                        else:
                            return rotate()
                            
                    return rotate()

    def valley(self, normalize = False, potency = 1, **args):
        """
        Makes a graph showing the valleys (matrix-U), clearly separated valleys indicate a possible group
        """
        n = globals()["number_of_neurons_"]
        l = n * 2 - 1
        l = [[0 for i in range(l)] for j in range(l)]

        for i in range(len(l)):
            for j in range(len(l)):
                if i % 2 == 0 and j % 2 == 1:
                    l[i][j] = globals()[f"n{int(i/2)}_{int(j/2)}"].distance(globals()[f"n{int(i/2)}_{int(j/2) + 1}"].weights)
                if i % 2 == 1 and j % 2 == 0:
                    l[i][j] = globals()[f"n{int(i/2)}_{int(j/2)}"].distance(globals()[f"n{int(i/2) + 1}_{int(j/2)}"].weights)

        for i in range(len(l)):
            for j in range(len(l[0])):
                if l[i][j] == 0:
                    v_, v__ = 0, 0
                    try:
                        v_ += l[i-1][j]
                        v__ += 1
                    except:
                        pass
                    try:
                        v_ += l[i][j-1]
                        v__ += 1
                    except:
                        pass
                    try:
                        v_ += l[i+1][j]
                        v__ += 1
                    except:
                        pass
                    try:
                        v_ += l[i][j+1]
                        v__ += 1
                    except:
                        pass
                    l[i][j] = v_/v__
                    #print(i,j,l[i][j],v_,v__)

        s = [i[:] for i in l]
    
        m = 0
        if normalize:
            for i in range(len(l)):
                for j in range(len(l[0])):
                    if l[i][j] > m:
                        m = l[i][j]

            for i in range(len(l)):
                for j in range(len(l[0])):
                    l[i][j] = l[i][j]/m * 255

        for i in range(len(l)):
            for j in range(len(l[0])):
                l[i][j] = [int(l[i][j] * potency), 255 - int(l[i][j] * potency), 200]

        if not "figsize" in args:
            args["figsize"] = (7,5)
            
        fig, ax = plt.subplots(figsize = args["figsize"])
        plt.imshow(l)#, cmap='gray')
        plt.axis('off')
        plt.title("Valley Graph (Matrix-U)")
        if "savefig" in args:
            plt.savefig(args["savefig"] + ".png", bbox_inches='tight')
            plt.clf()
            plt.cla()
        else:
            plt.show()

        def separate(s):
            n_s = [[0 for i in range(len(s))] for i in range(len(s))]
            for i in range(len(s)):
                for j in range(len(s)):
                    n_s[i][j] = s[i][j][0]
            return n_s

        x = np.arange(0, globals()["number_of_neurons_"] * 2 - 1)
        y = np.arange(0, globals()["number_of_neurons_"] * 2 - 1)
        x, y = np.meshgrid(x, y)
        altitude = np.array(separate(l))
        
        # Criar a figura e o eixo 3D
        fig = plt.figure(figsize = args["figsize"])
        ax = fig.add_subplot(111, projection='3d')

        # Plotar a superfície 3D
        ax.plot_surface(x, y, altitude, cmap = 'cividis')
        ax.set_xlabel('Neurons')
        ax.set_ylabel('Neurons')
        ax.set_zlabel('Distance')

        if "savefig" in args:
            plt.savefig(args["savefig"] + "_3d.png", bbox_inches='tight')
            plt.clf()
            plt.cla()
        else:
            plt.show()

        return s

    def amount_of_wins(self, **args):
        """
        Shows the number of wins for each neuron
        """
        for i in range(globals()["number_of_neurons_"]):
            for j in range(globals()["number_of_neurons_"]):
                globals()[f"n{i}_{j}"].number_of_wins = 0

        for d in globals()["global_som_data"]:
            min_ = globals()["n0_0"]
            for i in range(globals()["number_of_neurons_"]):
                for j in range(globals()["number_of_neurons_"]):
                    if globals()[f"n{i}_{j}"].distance(d) < min_.distance(d):
                        min_ = globals()[f"n{i}_{j}"]
            min_.number_of_wins += 1            
        
        data = [[0 for i in range(globals()["number_of_neurons_"])] for i in range(globals()["number_of_neurons_"])]
        for i in range(globals()["number_of_neurons_"]):
            for j in range(globals()["number_of_neurons_"]):
                exec(f"data[{i}][{j}] = n{i}_{j}.number_of_wins")

        np.array(data)

        if not "figsize" in args:
            args["figsize"] = (7,5)
            
        fig, ax = plt.subplots(figsize = args["figsize"])
        plt.imshow(data, cmap='hot')
        plt.title("Amount of Wins")
        plt.colorbar()
        if "savefig" in args:
            plt.savefig(args["savefig"] + ".png", bbox_inches='tight')
            plt.clf()
            plt.cla()
        else:
            plt.show()

    def predict(self, labels = None, **args):
        """
        Returns the predictions by neuron in addition to a graph if you pass the labels "labels:list"
        """
        observation = globals()["global_som_data"]
                
        def dist(a, b):
            if not len(a) == len(b):
                print("The dataframe is not the same length")
                return
            sum_ = 0
            for i in range(len(a)):
                sum_ += (a[i] - b[i]) ** 2

            return sum_

        try:
            observation = observation.values.tolist()
        except:
            pass
        
        l_global = []
        i = 0
        while True:
            j, l_local = 0, []
            while True:
                try:
                    l_local.append(globals()[f"n{i}_{j}"].weights)
                    j += 1
                except:
                    break
            if j == 0:
                break
            i += 1
            l_global.append(l_local)

        if len(observation) != len(l_global[0][0]): 
            clusters = []
            for k in range(len(observation)): #Cluster list
                min_ = dist(l_global[0][0], observation[k])
                clusters_temp = 0
                for i in range(len(l_global)):
                    for j in range(len(l_global[0])):
                        if min_ > dist(l_global[i][j], observation[k]):
                            clusters_temp = i * len(l_global) + j
                            min_ = dist(l_global[i][j], observation[k])
                clusters.append(clusters_temp)

            if labels != None: #Map of cluster
                image = {}
                for i in range(len(l_global)):
                    for j in range(len(l_global[0])):
                        min_ = dist(l_global[0][0], observation[k])
                        image[f"{i}_{j}"] = labels[0]
                        for k in range(len(observation)):
                            if min_ > dist(l_global[i][j], observation[k]):
                                image[f"{i}_{j}"] = labels[k]
                                min_ = dist(l_global[i][j], observation[k])

                list_image = []
                for i in range(len(l_global)):
                    list_image_temp = []
                    for j in range(len(l_global[0])):
                        list_image_temp.append(image[f"{i}_{j}"])
                    list_image.append(list_image_temp)


                def div_lista(l:list):
                    t = int(len(l)**(1/2))
                    k = [[0 for i in range(t)] for i in range(t)]
                    for i in range(len(k)):
                        for j in range(len(k[0])):
                            l_ = l[i * len(k) + j]
                            k[i][j] = l_[0]
                    return k
                    
                dados = list_image
                        
                # Obter todos os valores únicos da tabela
                valores_unicos = np.unique([valor for linha in dados for valor in linha])

                # Gerar uma lista de cores correspondente aos valores únicos
                cores = plt.cm.get_cmap('Set3', len(valores_unicos))

                cor = div_lista([[cores(valores_unicos.tolist().index(valor))] for linha in dados for valor in linha])

                if not "figsize" in args:
                    args["figsize"] = (globals()["number_of_neurons_"]/2, globals()["number_of_neurons_"]/3)
                    
                #print(list_image, len(image.keys()))
                fig, ax = plt.subplots(figsize = args["figsize"])
                tabela = ax.table(cellText = list_image,
                                  cellColours = cor,
                                  loc = 'center',
                                  cellLoc = 'center')
                tabela.auto_set_font_size(False)
                tabela.set_fontsize(10)
                tabela.scale(1, 1.7)
                ax.axis('off')
                plt.suptitle("Clustering with Labels")
                if "savefig" in args:
                    plt.savefig(args["savefig"] + ".png", bbox_inches='tight')
                    plt.clf()
                    plt.cla()
                else:
                    plt.show()
                
            return clusters
        
        else:
            min_ = dist(l_global[0][0], observation)
            for i in range(len(l_global)):
                for j in range(len(l_global[0])):
                    if min_ > dist(l_global[i][j], observation):
                        clusters_temp = i * len(l_global) + j
                        min_ = dist(l_global[i][j], observation)
            return clusters_temp

    def name(self):
        """
        Name of Neuron
        """
        c = list(globals())
        for i in c:
            try:
                if self == globals()[i]:
                    return i
            except ValueError:
                pass
        return None

    def __repr__(self):
        """
        Representation of Neuron
        """
        if self.con_l != None:
            return (self.con_l).__repr__()
        if self.con_d != None:
            return (self.con_d).__repr__()

        root = self
        k = self
        n = ""
        while True:
            n += k.name() + " - "

            if k.con_r != None:
                k = k.con_r
            else:
                if root.con_u != None:
                    n += "\n"
                    n = n[:-3] + n[-1:]
                    k = root.con_u
                    root = k
                else:
                    return n[:-2]

    def __matmul__(self, index):
        """
        Operation @
        """
        if index == "->" or index == ">":
            return self.con_r
        if index == "<-" or index == "<":
            return self.con_l
        if index == "v" or index == "V":
            return self.con_d
        if index == "A" or index == "^":
            return self.con_u
            

def uniform(mean, variance, dim = 2, times = 1):
    return [[mean[i] + variance * random() for i in range(dim)] for j in range(times)]

def create_SOM(x:int = 5, learning:float = 0.01):
    """
    Create de Grid of SOM

    :param x: grid width/height size
    :type x: int

    :param learning: learning rate
    :type learning: float

    Example of use:
    1) SOM = create_SOM(20, learning = 0.05)
    2) n1_1.design_weights(dados)
    3) n0_0.auto_organizing(epochs = 10, print_ = True)
    4) n1_1.valley(normalize = True, potency = 1)
    5) n1_1.amount_of_wins()
    6) clusters = n1_1.predict()
    """

    if "number_of_neurons_" in globals():
        print("Deleting old SOM template")
        n_ = globals()["number_of_neurons_"]
        for i in range(n_):
            for j in range(n_):
                del globals()[f"n{i}_{j}"]

        globals()["number_of_neurons_"] = x
    
    #Crio os neuronios:
    SOM = []
    for i in range(x):
        for j in range(x):
            exec(f"globals()['n{i}_{j}'] = Neuron({learning})")
            SOM.append(globals()[f"n{i}_{j}"])

    #Conecto os neuronios:
    for i in range(x):
        for j in range(x):
            try:
                r_ = "n" + str(i+1) + "_" + str(j)
                eval(r_)
            except:
                r_ = None
            try:
                l_ = "n" + str(i-1) + "_" + str(j)
                eval(l_)
            except:
                l_ = None
            try:
                u_ = "n" + str(i) + "_" + str(j+1)
                eval(u_)
            except:
                u_ = None
            try:
                d_ = "n" + str(i) + "_" + str(j-1)
                eval(d_)
            except:
                d_ = None
                
            s = f"n{i}_{j}.connect(obj_r = {r_}, obj_l = {l_}, obj_u = {u_}, obj_d = {d_})"
            exec(s)

    print("Use object n1_1, it is your starting point for SOM...\nTo prepare the SOM for your data: n1_1.design_weights(YOUR_DATA) \nTo start the SOM: n1_1.(dados)\n This function return the list of neurons")

    if not "number_of_neurons_" in globals():
        globals()["number_of_neurons_"] = x
    
    return SOM


#Example:
if __name__ == "__main__":

    seed(12)

    #from random import random
    #a = [random()*random()*random() for i in range(5)]
    #print(sorted(a))
    #print(sorted(adjust(a)))

    SOM = create_SOM(20, learning = 0.05)

##    #Crio dados ficticios
##    dado_1 = uniform([3, 0], 2,times = 10)
##    dado_2 = uniform([-30, 10], 2,times = 10)
##    dado_3 = uniform([5, 5], 2,times = 10)
##    dado_4 = uniform([-5, -10], 5,times = 10)
##
##    dados = []#[*dado_1, *dado_2, *dado_3, *dado_4]
##    for i in range(0, 11, 1):
##        for j in range(0, 11, 1):
##            for k in range(0, 11, 1):
##                dados.append([i,j,k])

    import pandas as pd
    dados = pd.read_csv("DATA_CETESB_BY_CODE.csv")
    rotulos = list(map(lambda x: x[:2],list(dados["code"])))

    c = dados.columns
    dados = dados.drop(columns = [*c[0:3],])
    c = dados.columns
    dados = dados.reindex(columns = [*c[-2:], *c[:-2]])
    #dados = dados.drop(columns = [*c[:-2]])

    #dados = dados.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    dados = dados.values.tolist()

    #Crio a grade que conecta a vizinhança:
    n1_1.design_weights(dados)

    #Uma iteração de auto organização:
    n0_0.auto_organizing(epochs = 5, print_ = True)

    clusters = n1_1.predict(labels = rotulos)
    new_clusters = adjust_clusters(clusters)

    n1_1.valley(normalize = True, potency = 1)

    n1_1.amount_of_wins()

##
##
##
##    SOM = create_SOM(20, learning = 0.005)
##    #Crio a grade que conecta a vizinhança:
##    n1_1.design_weights(dados)
##
##    i = 0
##    n0_0.auto_organizing(epochs = 0, print_ = True)
##    clusters = n1_1.predict(labels = rotulos, savefig = f"C:\\Users\\Usuario\\Desktop\\img_som_int\predict\\\predict_som_{i}")
##    n1_1.valley(normalize = True, potency = 1, savefig = f"C:\\Users\\Usuario\\Desktop\\img_som_int\\valley\\predict_som_{i}")
##    n1_1.amount_of_wins(savefig = f"C:\\Users\\Usuario\\Desktop\\img_som_int\\amount\\predict_som_{i}")
##        
##    #Uma iteração de auto organização:
##    for i in range(1, 500):
##        print(f"Save {i}")
##        n0_0.auto_organizing(epochs = 1, print_ = True)
##        clusters = n1_1.predict(labels = rotulos, savefig = f"C:\\Users\\Usuario\\Desktop\\img_som_int\predict\\\predict_som_{i}")
##        n1_1.valley(normalize = True, potency = 1, savefig = f"C:\\Users\\Usuario\\Desktop\\img_som_int\\valley\\predict_som_{i}")
##        n1_1.amount_of_wins(savefig = f"C:\\Users\\Usuario\\Desktop\\img_som_int\\amount\\predict_som_{i}")
##        
##    new_clusters = adjust_clusters(clusters)   

    

