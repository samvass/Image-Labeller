import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options


    def _init_X(self, X):

        X = np.array(X, dtype=float)

        if X.ndim > 2:
            X = X.reshape(X.shape[0]*X.shape[1], -1)

        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.
        if 'threshold' not in options:
            options['threshold'] = 20
        if 'distance_type' not in options:
            options['distance_type'] = 'wcd'

        self.options = options

     
    def _init_centroids(self):

        centroids = np.zeros((self.K, 3))
        old_centroids = np.zeros((self.K, 3))

        init_option = self.options['km_init']

        if init_option == 'first':

            i = 0
            j = 0

            while i < self.K:

                pixel = self.X[j]

                valid = True

                for p in centroids:

                    if p[0] == pixel[0] and p[1] == pixel[1] and p[2] == pixel[2]:
                        valid = False
                        break

                if valid:
                    centroids[i] = pixel
                    i += 1  

                # next pixel
                j += 1
               
            
        # split the data into k sections and take one point from each
        elif init_option == 'spread':
            
            group_size = self.X.shape[0]/self.K
            
            j = 0
            
            while i < self.K:
                
                centroids[i] = self.X[j*group_size]
                i += 1
                j += 1
                
                
                
        # alternates between top and bottom of the array
        elif init_option == 'top_bottom':
            
            j = 0
            z = self.X.shape[0]-1
            c = 0
            
            while i < self.K:
                
                if c % 2 == 0:
                
                    centroids[i] = self.X[j]
                    j += 1
                    
                else:
                    
                    centroids[i] = self.X[z]
                    z -= 1
                    
                c += 1
                               
                
                
        elif init_option == 'random':

                index = np.random.randint(self.X.shape[0])

                centroids[0] = self.X[index]

                i = 1

                while i < self.K:

                    index = np.random.randint(self.X.shape[0])

                    pixel = self.X[index]

                    for p in centroids:
                        for d in range(0,self.x.shape[1]):

                            if p[d] == pixel[d]:
                                # not unique
                                valid = False

                    if valid:
                        centroids[i] = pixel
                        i += 1

        elif init_option == 'custom':
            # The custom option is similiar to the first option, but instead of taking
            # the first k rows from X, it takes the last k rows from X
            # To do that it is easier just to flip the numpy array X with the axis = 0
            # so the elements are on the same column position, but we flip the rows
            i = 0
            j = 0

            X_reversed = np.flip(self.X, axis=0)

            while i < self.K:
                pixel = X_reversed[j]

                valid = True

                for p in centroids:

                    if p[0] == pixel[0] and p[1] == pixel[1] and p[2] == pixel[2]:
                        valid = False
                        break

                if valid:
                    centroids[i] = pixel
                    i += 1

                    # next pixel
                j += 1
                
        self.centroids = centroids
        self.old_centroids = old_centroids


    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        distances = np.sum((self.X[:, np.newaxis, :] - self.centroids) ** 2, axis=-1)
        # Assign each point to the closest centroid
        self.labels = np.argmin(distances, axis=1)


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """

        # Pass the value of centroids to old_centroids
        self.old_centroids = np.copy(self.centroids)

        # Loop over each centroid
        for k in range(self.K):

            X_k = self.X[self.labels == k]

            if X_k.shape[0] > 0:
                self.centroids[k] = np.mean(X_k, axis=0)


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        
        comparison = self.centroids == self.old_centroids

        diff = np.count_nonzero(comparison == False)

        if diff <= self.options['tolerance']:
            return True

        else:
            return False

        
        return np.array_equal(self.old_centroids, self.centroids)

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        
        i = 0 
        
        self._init_centroids()
        self.get_labels()
        
        while i < self.options['max_iter'] and not self.converges():
            self.get_centroids()
            self.get_labels()
            i += 1

            
    def withinClassDistance(self):
        """
        returns the within class distance of the current clustering
        """
        total = np.sum((self.X - self.centroids[self.labels]) ** 2, axis=1)
        wcd = np.mean(total)
        self.wcd = wcd
        return wcd
    
    def interClassDistance(self):
        num_classes = np.unique(labels).shape[0]
        icd = 0.0

        for i in range(num_classes):
            for j in range(i+1, num_classes):
                class_i_points = X[labels == i]
                class_j_points = X[labels == j]
                class_distance = np.mean(np.sum((class_i_points[:, None] - class_j_points) ** 2, axis=2))
                icd += class_distance

        return icd
    
    def fischer(self):
        return withinClassDistance()/interClassDistance()

        
    def find_bestK(self, max_K):
        """
        sets the best k analysing the results up to 'max_K' clusters
        """

        self.K = 2
        self.fit()
        
        if self.options['distance_type'] == 'wcd':
            
            wcd_prev = self.withinClassDistance()
            for k in range(3, max_K+1):
                self.K = k
                self.fit()

                wcd = self.withinClassDistance()
                if (wcd_prev - wcd)/wcd_prev * 100 < self.options['threshold']:
                    self.K -= 1
                    break
                
                wcd_prev = wcd
        
        elif self.options['distance_type'] == 'icd':
            icd_prev = self.interClassDistance()
            for k in range(3, max_K+1):
                self.K = k
                self.fit()

                wcd = self.withinClassDistance()
                if (icd_prev - icd)/icd_prev * 100 > self.options['threshold']:
                    self.K -= 1
                    break
                
                icd_prev = icd
            
        elif self.options['distance_type'] == 'fischer':
            f_prev = self.fischer()
            for k in range(3, max_K+1):
                self.K = k
                self.fit()

                f = self.fischer()
                if (f_prev - f)/f_prev * 100 < self.options['threshold']:
                    self.K -= 1
                    break
                
                f_prev = f
        

def distance(X, C):
    """
        Calculates the distance between each pixel and each centroid
        Args:
            X (numpy array): PxD 1st set of data points (usually data points)
            C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

        Returns:
            dist: PxK numpy array position ij is the distance between the
            i-th point of the first set an the j-th point of the second set
        """

    distances = np.zeros((X.shape[0], C.shape[0]))

    x_it = 0
    c_it = 0

    for i in range(X.shape[0]):
        x = X[i]

        for j in range(C.shape[0]):
            c = C[j]

            dist = np.linalg.norm(x - c)

            distances[i][j] = dist

    return distances


def get_colors(centroids):
    """
        for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
        Args:
            centroids (numpy array): KxD 1st set of data points (usually centroid points)

        Returns:
            labels: list of K labels corresponding to one of the 11 basic colors
        """

    labels = []
    label_id = [0] * centroids.shape[0]  # Initialize label_id with zeros
    threshold = 0.2

        # Getting the probabilities in a k x 11 matrix
    colors_prob = utils.get_color_prob(centroids)
        #ix = np.argmax(colors_prob, axis=1)


        # Putting in lable list the color with the biggest prob for every row
    for i in range(centroids.shape[0]):
        max_color_id = np.argmax(colors_prob[i])
        if utils.colors[max_color_id] == 'White':
                # Checking if any other color probability is greater than a threshold
            for c in range(11):
                if c == max_color_id:
                    continue
                if colors_prob[i][c] > threshold:
                        # Store index of the color with the second highest probability
                    label_id[i] = c
                    break
                else:
                        # If no other color probability is greater than the threshold, assign white as the label
                    label_id[i] = max_color_id
        else:
                # Store index of the maximum color probability
            label_id[i] = max_color_id

            # Putting in label list the color corresponding to the label id for every row
    for i in range(centroids.shape[0]):
        labels.append(utils.colors[label_id[i]])

    return labels
