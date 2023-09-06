import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from PIL import Image


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        scale_factor = 0.5 #using the scale factor 0.45 or 0.5, we obtain the best accuracy

        if scale_factor < 1.0:
            resized_images = []

            for image in train_data:
                img = Image.fromarray(image.astype('uint8'), 'RGB')
                resized_image = img.resize((int(img.width * scale_factor), int(img.height * scale_factor)))
                resized_images.append(np.array(resized_image))

            features = np.array(resized_images).reshape(train_data.shape[0], -1)
        else:
            features = train_data.reshape(train_data.shape[0], -1)

        self.train_data = features

        return self.train_data

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        test_data = test_data.astype(np.float32)
   
        scale_factor = 0.5 #using the scale factor 0.45 or 0.5, we obtain the best accuracy
        if scale_factor <1:
           # Resize test_data to match the shape of train_data
           resized_images = []

           for image in test_data:
               img = Image.fromarray(image.astype('uint8'), 'RGB')
               resized_image = img.resize((int(img.width * scale_factor), int(img.height * scale_factor)))
               resized_images.append(np.array(resized_image))

           resized_test_data = np.array(resized_images).reshape(test_data.shape[0], -1)
       
        else:
           resized_test_data = test_data.reshape(test_data.shape[0], -1)

        distances = cdist(resized_test_data, self.train_data)
        neighbors_indices = np.argsort(distances, axis=1)[:, :k]
        self.neighbors = self.labels[neighbors_indices]
    
        

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
                
              
        """
        unique_labels = np.unique(self.neighbors)
        most_common_labels = []
        most_common_labels_percent = []
        label_occur = {label: 0 for i, label in enumerate(unique_labels)}
        
        for row in self.neighbors:
            for item in row:
                label_occur[item] += 1
            
            most_common = row[0]
            total = 0
            for item in row:
                total += 1
                if label_occur[item] > label_occur[most_common]:
                    most_common = item
            
            #most_common_labels_percent.append(label_occur[item] / total * 100)
            most_common_labels.append(most_common)
            label_occur = {label: 0 for i, label in enumerate(unique_labels)}
            
        #print(self.neighbors)
        #print(np.array(most_common_labels), np.array(most_common_labels_percent))
        return np.array(most_common_labels)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
