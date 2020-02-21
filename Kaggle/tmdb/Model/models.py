# Class that will allow us to facilitate the use of parameters
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

SGDRegressor = linear_model.SGDRegressor



# Number of folds for the cross validation
NFOLDS = 5
# Seed to avoid randomization
SEED = 1
# Time cross validation
tscv = TimeSeriesSplit(n_splits=NFOLDS)


############################################# Auxiliary functions
def plot_graph(x,y,x_name,y_name,title) :
    # Plots a graph with the given values
    f, ax = plt.subplots(figsize=(8, 5))
    plt.plot(x,y, marker='+')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.grid()
    plt.show()

def get_max_matrix(matrix) :
    # Gives the two indexes that maximize the matrix
    i1,i2 = np.where(matrix == np.max(matrix))
    i1,i2 = i1[0],i2[0]
    return i1,i2

def plot_predict_real(regressor, n, y_label, algorithm, ylim,index) :
    # Plots the predicted versus the real values for the n first ones
    f,ax = plt.subplots(figsize=(8, 5))
    ax.scatter([i for i in range(len(regressor.y_test))][:n] , regressor.y_test[:n], marker='+',color='r',\
           label="Real values", s=100)
    c_r = SKlearnHelper(clf = regressor.classifiers[index] ,params = regressor.list_param[index])
    c_r.fit(regressor.X_train, regressor.y_train)
    y_predict_r = c_r.predict(regressor.X_test)
    ax.scatter([i for i in range(len(y_predict_r))][:n] , y_predict_r[:n], marker='o', color='b',\
            label="Predicted values",s=60)
    plt.xlabel("Number of instance")
    plt.ylabel(y_label)
    plt.legend()
    plt.ylim(ylim)
    plt.title("Comparaison of ratio with " + algorithm )






def get_max_acc_param(param, name_param, liste_param , X_train, y_train ,classifier) :
    # Loops through the classifier with the given parameters
    liste_score = []
    for x in liste_param :
        param[name_param] = x
        classif = SKlearnHelper(clf = classifier, params = param)
        score = get_oof_c(classif, X_train, y_train)
        liste_score.append(score)
    index = np.where( liste_score == np.max(liste_score))[0][0]
    param[name_param] = liste_param[index]
    classif = SKlearnHelper(clf = classifier , params = param)
    return param, classif,np.max(liste_score)


def get_max_acc_two_params(parameters, names_param, listes_param, X_train, y_train, classifier) :
    # Loops through the classifier with the given parameters (loops through two set of parameters)
    liste_score = np.zeros((len(listes_param[0]), len(listes_param[1])))
    for index,x in enumerate(listes_param[0]) :
        parameters[names_param[0]] = x
        for index1, y in enumerate(listes_param[1]) :
            parameters[names_param[1]] = y
            classif = SKlearnHelper(clf = classifier, params = parameters)
            score = get_oof_c(classif, X_train, y_train)
            liste_score[index,index1] = score
    i_0, i_1 = get_max_matrix(liste_score)
    parameters[names_param[0]] = listes_param[0][i_0]
    parameters[names_param[1]] = listes_param[1][i_1]
    classif = SKlearnHelper(clf = classifier, params = parameters)
    return parameters, classif, np.max(liste_score),liste_score




############################################ Metrics functions


def get_oof(clf, x_train, y_train) :
    # Out of fold : returns the average after doing TimeSeriesSplit for a regressor
    oof_score = np.zeros( (NFOLDS))
    for i,(train_index, test_index) in enumerate(tscv.split(x_train)) :
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        y_te = y_train[test_index]
        clf.fit(x_tr,y_tr)
        oof_y_predict = clf.predict(x_te)
        oof_score[i] = r2_score(y_te,oof_y_predict)

    return np.mean(oof_score)


def get_oof_c(clf, x_train, y_train) :
    # Out of fold : returns the average after doing TimeSeriesSplit for a classifier
    oof_score = np.zeros( (NFOLDS))
    for i,(train_index, test_index) in enumerate(tscv.split(x_train)) :
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        y_te = y_train[test_index]
        clf.fit(x_tr,y_tr)
        y_pred = clf.predict(x_te)
        oof_score[i] = accuracy_score(y_te, y_pred)

    return np.mean(oof_score)

def get_score_c(model, x_train, y_train, x_test, y_test) :
    # Get the final score after getting the best parameters for classification
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    return accuracy_score(y_test, y_predict)


class SKlearnHelper(object):
    # Class that will help us in our code
    # It just allows to create a classifier or regressor by passing it the parameters
    # This is just to gather all the classifiers / regressors as a same object
    def __init__(self, clf, seed=0, params=None):
        #params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        # Trains with the training set
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        # Predicts the given data
        return self.clf.predict(x)

    def fit(self,x,y):
        # Fits with the given training set
        return self.clf.fit(x,y)










############################### Functions to optimize the parameters for regression and classification

def get_knn(X_train, y_train,to_plot) :
    # If to_plot is True, it will plot the score for each k_neighbors
    # It loops through the algorithms used to choose the neighbors
    knn_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

    liste_score_knn = []
    n=50
    if (len(X_train) < n) :
        n = len(X_train) - 2

    for k in np.arange(1,n,1) :
        knn_params = {
            'n_neighbors':k,
            'algorithm':knn_algorithm[0],
        }
        knn_r = SKlearnHelper(clf = KNeighborsRegressor, params = knn_params)
        oof_score_knn_r = get_oof(knn_r, X_train, y_train)
        liste_score_knn.append(oof_score_knn_r)
    knn_params['n_neighbors'] = np.arange(1,n,1)[np.argmax(liste_score_knn)]
    if (to_plot) :
        plot_graph(np.arange(1,n,1),liste_score_knn,"Number of neighbors","Score of knn",\
           "Choose parameter for knn")
    return  knn_params,  KNeighborsRegressor



def get_sgdr(X_train,y_train,to_plot) :
    # It returns a stochastic gradient descent regressor
    # It loops through the loss and the penalty
    loss_list = ['squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive']
    penality_list = ['l2','l1','elasticnet']

    param_sgd ={
        'loss':loss_list[0],
        'penalty': penality_list[0],
        'shuffle' : False,
        'max_iter': 1000,
        'random_state': SEED
    }


    liste_score_sgd = []
    for x in loss_list :
        param_sgd['loss'] = x
        current_score = []
        for y in penality_list :
            param_sgd['penalty'] = y
            sgd = SKlearnHelper(clf = SGDRegressor, params = param_sgd)
            oof_score_sgd =  get_oof(sgd, X_train, y_train)
            current_score.append(oof_score_sgd)
        liste_score_sgd.append(current_score)
    i_loss, i_penalty =get_max_matrix(liste_score_sgd)
    if to_plot :
        liste_score_sgd = np.array(liste_score_sgd)
        print("Score for different parameters : ")
        sgd_param_data = {}
        for i in range(3) :
            sgd_param_data[penality_list[i]] = liste_score_sgd[:,i]
        sgd_param_df = pd.DataFrame(sgd_param_data, index =loss_list)
        print(sgd_param_df)
    param_sgd['loss'] = loss_list[i_loss]
    param_sgd['penalty'] = penality_list[i_penalty]
    return   param_sgd,SGDRegressor



def get_dtr(X_train,y_train, to_plot) :
    # Returns a decision tree regressor
    # It loops through the criterion and the max depth
    criterion_list = ['mse', 'friedman_mse', 'mae']
    param_tree = {
        'criterion': criterion_list[0],
        'max_depth': 3   ,
        'max_features' : "log2",
        'random_state':SEED
    }
    liste_score_tree = []
    for x in criterion_list :
        liste_inter = []
        param_tree['criterion'] = x
        n = 100
        if (len(X_train) < n) :
            n = len(X_train) - 2

        for i in range(1,n,5) :
            param_tree['max_depth'] = i
            tree = SKlearnHelper(clf = DecisionTreeRegressor , params = param_tree)
            oof_score_tree =  get_oof(tree, X_train, y_train)
            liste_inter.append(oof_score_tree)
        liste_score_tree.append(liste_inter)

    if (to_plot):
        f, ax = plt.subplots(figsize=(8, 8))
        for index,x in enumerate(criterion_list) :
            plt.plot(np.arange(1,n,5),liste_score_tree[index], marker='+',label=x)
        plt.xlabel("Max depth")
        plt.ylabel("Score of Decistion Tree Regressor")
        plt.legend()
        plt.title("Choose parameter for Decision Tree Regressor")
        plt.grid()
        plt.show()

    i1,i2 = get_max_matrix(liste_score_tree)
    param_tree['criterion'] = criterion_list[i1]
    param_tree['max_depth'] = np.arange(1,n,5)[i2]
    return param_tree, DecisionTreeRegressor




def get_rfr(X_train,y_train, to_plot,max_depth=None) :
    # Returns a Random Forest regressor
    # It loops through the number of estimators and through the max depth
    max_f_liste = ["auto","sqrt","log2"]
    param_rfr = {
        'max_depth':None,
        'n_estimators':100,
        'max_features': max_f_liste[0],
    }
    if max_depth is None :
        max_depth = 10
    liste_tot_score = np.zeros((max_depth,19))
    for index,i in enumerate(np.arange(1,max_depth+1)) :
        liste_score_rfr = []
        param_rfr['max_depth']  = i
        if (len(X_train) > i) :
            for index1,j in enumerate(np.arange(1,20,1)) :
                param_rfr['n_estimators'] = j
                rfr = SKlearnHelper(clf = RandomForestRegressor, params = param_rfr)
                oof_score_rfr =  get_oof(rfr, X_train, y_train)
                liste_score_rfr.append(oof_score_rfr)
                liste_tot_score[index,index1] = oof_score_rfr

    i_depth , i_est = get_max_matrix(liste_tot_score)


    param_rfr['max_depth'] = i_depth+1
    param_rfr['n_estimators'] = min(40,i_est+1)
    print("Max depth : " + str(i_depth+1))
    print("Number of estimators : " + str(i_est+1))
    if (to_plot) :
        f, ax = plt.subplots(figsize=(8, 8))
        for i in range(max_depth) :
            plt.plot(np.arange(1,20), liste_tot_score[i], label="Max_depth " + str(i+1),marker= '+')
        plt.xlabel("Number of estimators")
        plt.ylabel("Score of Decistion RandomForestRegressor")
        plt.legend()
        plt.title("Choose parameter for RandomForestRegressor")
        plt.grid()
        plt.show()


    return   param_rfr, RandomForestRegressor

def get_svr(X_train, y_train, to_plot) :
    # Returns a Support Vector Regressor with the best parameters
    # It loops on the kernel used : either polynomial or Radial Basis Function
    # Then it loops also on the gamma value.
    param_svr = {
        'kernel' : 'rbf',
        'C' : 10,
        'gamma' :'auto'
    }
    param_svr_poly = {
        'kernel': 'poly',
        'C':1,
        'gamma':'auto',
        'degree':4,

    }

    kernel_list = ['rbf','poly']
    liste_score_c_iter = [np.arange(1,10), np.arange(1,10) ]
    liste_score_ck = [[],[]]
    for j in range(len(liste_score_c_iter)) :
        print(j)
        param_svr['kernel'] = kernel_list[j]
        for i in liste_score_c_iter[j] :
            if (j==0) :
                param_svr['gamma'] = i
                svr = SKlearnHelper(clf = SVR, params = param_svr)
            else :
                param_svr_poly['degree'] = i
                svr = SKlearnHelper(clf = SVR, params = param_svr_poly)
            oof_score_svr = get_oof(svr, X_train,y_train)
            liste_score_ck[j].append(oof_score_svr)
    if (to_plot) :
        for i in range(len(liste_score_c_iter)) :
            if i == 1 :
                plot_graph(liste_score_c_iter[i],np.array(liste_score_ck)[i],"gamma parameter","Score","Score for SVC using polynomial kernel")
            else :
                plot_graph(liste_score_c_iter[i],np.array(liste_score_ck)[i],"gamma parameter","Score","Score for SVC using rbf kernel")

    i_rbf = np.where(liste_score_ck[0] == np.max(liste_score_ck[0]))[0][0]
    i_poly = np.where(liste_score_ck[1] == np.max(liste_score_ck[1]))[0][0]
    if liste_score_ck[0][i_rbf] > liste_score_ck[1][i_poly] :
        param_svr['C'] = i_rbf + 1
        svr = SKlearnHelper(clf = SVR, params = param_svr)
        return param_svr, SVR
    else :
        param_svr_poly['C'] = i_poly + 1
        return   param_svr_poly,SVR

def get_svr_2(X_train, y_train, to_plot) :
    param_svr = {
        'kernel' : 'rbf',
        'C' : 1,
        'gamma' :'auto'

    }
    param_svr_poly = {
        'kernel': 'poly',
        'C':1,
        'gamma':'auto',
        'degree':3,
        'epsilon':.1,
        'coef0':1
    }

    kernel_list = ['rbf','poly']
    liste_score_c_iter = [np.arange(1,5), np.arange(1,5) ]
    liste_score_ck = [[],[]]
    for j in range(len(liste_score_c_iter)) :
        param_svr['kernel'] = kernel_list[j]
        for i in liste_score_c_iter[j] :
            if (j==0) :
                param_svr['gamma'] = i/10
                svr = SKlearnHelper(clf = SVR, params = param_svr)
            else :
                param_svr_poly['gamma'] = i/10
                svr = SKlearnHelper(clf = SVR, params = param_svr_poly)
            oof_score_svr = get_oof(svr, X_train,y_train)
            liste_score_ck[j].append(oof_score_svr)
    if (to_plot) :
        for i in range(len(liste_score_c_iter)) :
            if i == 1 :
                plot_graph(liste_score_c_iter[i],np.array(liste_score_ck)[i],"gamma parameter","Score","Score for SVC using polynomial kernel")
            else :
                plot_graph(liste_score_c_iter[i],np.array(liste_score_ck)[i],"gamma parameter","Score","Score for SVC using rbf kernel")

    i_rbf = np.where(liste_score_ck[0] == np.max(liste_score_ck[0]))[0][0]
    i_poly = np.where(liste_score_ck[1] == np.max(liste_score_ck[1]))[0][0]
    if liste_score_ck[0][i_rbf] > liste_score_ck[1][i_poly] :
        param_svr['gamma'] = i_rbf + 1
        svr = SKlearnHelper(clf = SVR, params = param_svr)
        return param_svr, SVR
    else :
        param_svr_poly['gamma'] = i_poly + 1
        return   param_svr_poly,SVR





def get_lda(X_train,y_train, to_print) :
    # Returns a lda classifier
    # It loops through the solvers
    liste_solver = ['svd','lsqr']
    liste_score = []
    param_lda = {
        'solver': liste_solver[0]
    }
    param_lda , lda, score_max = get_max_acc_param(param_lda, 'solver',\
                                 liste_solver, X_train, y_train , LinearDiscriminantAnalysis)
    if (to_print) :
        print("Solver chosen : ",param_lda['solver'])
        print("Score after cross validation for LDA ",score_max)
    return param_lda, LinearDiscriminantAnalysis



def get_lreg(X_train,y_train,to_print) :
    # Returns a logistic regressor
    # It loops through the solver
    liste_solver = ['newton-cg', 'sag','saga','lbfgs']
    param_lreg = {
        'random_state':SEED,
        'solver':liste_solver[0],
        'multi_class':'multinomial'
    }

    param_lreg , lreg , max_score = get_max_acc_param(param_lreg, 'solver',\
                                                     liste_solver, X_train,y_train,
                                                     LogisticRegression)

    if (to_print) :
        print("Solver chosen : ",param_lreg['solver'])
        print("Score after cross validation for Logistic Regression ",max_score)
    return param_lreg, LogisticRegression


def get_perc(X_train, y_train, to_print) :
    # Returns a Perceptron
    # It loops through the penalty
    liste_penalty = ['l2','l1','elasticnet']
    param_perc = {
        'penalty' : liste_penalty[0],
        'random_state':SEED
    }

    param_perc, perc, max_score = get_max_acc_param(param_perc, 'penalty', liste_penalty,\
                                                   X_train,y_train, Perceptron)
    if (to_print) :
        print("Penalty chosen : ",param_perc['penalty'])
        print("Score after cross validation for Perceptron",max_score)
    return param_perc, Perceptron


def get_svm(X_train, y_train, to_print) :
    # Returns a support vector machine classifier
    # It loops through the different kernel available
    liste_kernel = [ 'poly', 'rbf', 'sigmoid']

    param_svm = {
        'random_state':SEED,
        'gamma':'auto',
        'kernel' : liste_kernel[0]
    }
    param_svm, svm, max_score = get_max_acc_param(param_svm, 'kernel', liste_kernel,\
                                                   X_train,y_train, SVC)
    if (to_print) :
        print('Kernel used : ', param_svm['kernel'])
        print("Score after cross validation for SVM",max_score)
    return param_svm, SVC


def get_dtc(X_train, y_train, to_print):
    # Returns a Decision Tree Classifier
    # It loops through the max depth
    liste_depth = np.arange(1,100)
    param_dtc = {
        'random_state':SEED,
        'max_depth' : 1,
        'max_features':'auto',
        'splitter' : 'best'

    }
    param_dtc , dtc, score_max = get_max_acc_param(param_dtc,\
                                                        'max_depth', liste_depth, X_train,\
                                                        y_train, DecisionTreeClassifier)

    if (to_print) :
        print("Max depth chosen : ",param_dtc['max_depth'])
        print("Score after cross validation for Decision Tree",score_max)
    return param_dtc,DecisionTreeClassifier


def get_rfc(X_train, y_train, to_print) :
    # Returns a Random Forest Classifier
    # It loops through both the max depth and the number of estimators
    liste_params = [np.arange(1,100,10), np.arange(1,10)]
    name_params = ['n_estimators','max_depth']
    param_rfc = {
        'n_estimators' : 1,
        'max_depth' : 1,
        'random_state':SEED
    }

    param_rfc, rfc,\
    max_score,liste_score = get_max_acc_two_params(param_rfc,name_params, liste_params,\
                           X_train, y_train, RandomForestClassifier)

    if param_rfc['n_estimators'] > 40 :
        param_rfc['n_estimators'] = 40

    if (to_print ) :
        print("Score after cross validation for Random Forest",max_score)
        f, ax = plt.subplots(figsize=(8, 5))
        for index, i in enumerate(liste_params[1]) :
            plt.plot(liste_params[0], liste_score[:,index], label="Max_depth " + str(index+1),marker= '+')
        plt.grid()
        plt.xlabel("Number of estimators")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        print("Number of estimators : ",param_rfc['n_estimators'] )
        print("Max depth : ", param_rfc['max_depth'])
    return param_rfc , RandomForestClassifier

def get_ada(X_train, y_train, to_print) :
    # Returns an AdaBoost Classifier
    # It loops through the number of estimators (low algorithms)
    liste_estimators = np.arange(1,150,10)
    param_ada = {
        'random_state':SEED,
        'n_estimators' : 1,
        'learning_rate':0.5,
        'random_state': SEED
    }
    param_ada,ada ,max_score = get_max_acc_param(param_ada,'n_estimators',liste_estimators,\
                                                X_train,y_train, AdaBoostClassifier)
    if (to_print) :
        print("Number of estimators : ",param_ada['n_estimators'])
        print("Score after cross validation for Adaboost",max_score)

    return param_ada,AdaBoostClassifier
