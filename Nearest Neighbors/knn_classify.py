import sys
import numpy as np
from scipy.spatial import distance

def euclidean_distance(x1,x2):
    return distance.euclidean(x1, x2)

def KNN(k,x_train,y_train,x_test,y_test):
    obj_id = 0
    sum=0
    for obj in x_test:
        distance = [euclidean_distance(obj,obj2) for obj2 in x_train]
        distance = np.argsort(distance)
        distance = distance[:k]
        cls = [y_train[i]for i in distance]
        ##pred_class
        counter=1
        for i in range(k):
            if(i!=1 and cls[i]==cls[0]):
                counter += 1
        
        pred_class=cls[0]
        true_class = y_test[obj_id]
        classification_accuracy = 0
        if(pred_class == true_class):
            classification_accuracy = float(1.0/counter)
        sum += classification_accuracy
        print("ID=%5d, predicted=%3d, true=%3d"%(obj_id,pred_class,true_class))
        print("classification accuracy=%6.4lf"%( classification_accuracy));
        
        obj_id += 1
    average_classification_accuracy = sum/len(y_test)
    #print(average_classification_accuracy)
    #print("Average Classification Accuracy=%f"%(average_classification_accuracy))
    return average_classification_accuracy

def main():
    
    if(len(sys.argv)==4):
        train = np.genfromtxt(sys.argv[1]+'.txt')
        test = np.genfromtxt(sys.argv[2]+'.txt')
        k = int(sys.argv[3])
    else:
        train = np.genfromtxt('pendigits_training.txt')
        test = np.genfromtxt(' pendigits_test.txt')
        k = 3
        
    X_train = train[:,:16] 
    y_train = train[:,16]
    X_test = test[:,:16]
    y_test = test[:,-1]
    dim_mean = np.mean(X_train, axis = 0)
    dim_std = np.std(X_train,axis = 0)

    # normalizing train data set
    Norm_X_train=[]
    for row in X_train:
        for i in range(16):
            row[i] = (row[i]-dim_mean[i])/dim_std[i]
        Norm_X_train.append(row)
    Norm_X_train = np.array(Norm_X_train)

    # normalizing test data set
    Norm_X_test=[]
    for row in X_test:
        for i in range(16):
            row[i] = (row[i]-dim_mean[i])/dim_std[i]
        Norm_X_test.append(row)
    Norm_X_test = np.array(Norm_X_test)
    
    average_classification_accuracy = KNN(k,Norm_X_train,y_train,Norm_X_test,y_test)
    print("Average Classification Accuracy=%f"%(average_classification_accuracy))
if __name__ == "__main__":
    main()
