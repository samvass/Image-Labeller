from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
import KNN
import Kmeans
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)
    

    # You can start coding your functions here
    def retrieval_by_color(list_images, tags, color):
        
        retrieved_imgs = []
        
        print(color)
        
        for i,img in enumerate(list_images):
            
            if color in tags[i]:
                
                retrieved_imgs.append(img)
        
        return np.array(retrieved_imgs)

    def retrieval_by_shape(list_images, tags, shape):
        retrieved_imgs = []
        
        for i,img in enumerate(list_images):
            
            if shape == tags[i]:
                
                retrieved_imgs.append(img)
        
        return np.array(retrieved_imgs)
    
    # Get−shape−accuracy: Function that receives as input the tags we obtained when applying 
    #the KNN and the Ground-Truth of these. Returns the correct tag percentage

    
    def get_shape_accuracy(predicted_class_labels, test_class_labels):
        
        size = test_class_labels.size # select a set amount of images from test_imgs and their indices
        
        correct = 0
                
        for i in range(0, size):

            if predicted_class_labels[i] == test_class_labels[i]:
                correct += 1
        
        return correct/size*100
    
    def get_shape_accuracy_individual(predicted_class_labels, test_class_labels):
        unique_items = set(predicted_class_labels).union(set(test_class_labels))
        clothing_items = list(unique_items)
        accuracy = []
        size = test_class_labels.size
        for item in clothing_items:
            total = 0
            correct = 0
            for i in range(0, size):
                if test_class_labels[i] == item:
                    total += 1
                    if predicted_class_labels[i] == test_class_labels[i]:
                        correct += 1
            accuracy.append(correct/total)
        return pd.DataFrame({"Clothing Item": clothing_items, "Accuracy": accuracy})
    
    # ------------- set up ---------------
    
    # train the KNN algorithm    
    knn = KNN.KNN(train_imgs, train_class_labels)
    
    # predict test_class_labels
    predicted_class_labels = knn.predict(test_imgs, 5) # why is K 5?
    
    imgs = test_imgs
    tags = []
    options = {}
    for img in imgs:
        km = Kmeans.KMeans(img, 1, options)
        km.fit()
        colors = Kmeans.get_colors(km.centroids)
        tags.append(colors)
            
    # ------------- qualitative analysis ---------------
    
    pink = retrieval_by_color(test_imgs, tags, 'Pink')
    visualize_retrieval(pink, 10)
    
    jeans = retrieval_by_shape(test_imgs, predicted_class_labels, 'Jeans')
    visualize_retrieval(jeans, 10)
    
    
#     # ------------- quantitative analysis ---------------

    # compare predicted test class labels with ground truth
    print(f"OVERALL SHAPE ACCURACY: {get_shape_accuracy(predicted_class_labels, test_class_labels)}")
   
    # comapre individual shape accuracy
    my_data = get_shape_accuracy_individual(predicted_class_labels, test_class_labels)
    
    # produce barplot with pandas/ seaborn
    sns.barplot(data=my_data, x="Clothing Item", y="Accuracy")
    plt.title("Accuracy of Labeling Software for Clothing Items")
    plt.xlabel("Clothing Item")
    plt.ylabel("Accuracy (%)")
    plt.show()
    

    
    
    
    