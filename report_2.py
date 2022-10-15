#%% imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
import torch
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)

sns.set()
# LaTeX textwidth = 469.47049pt
#%% data 
# https://www.kaggle.com/uciml/student-alcohol-consumption
df_mat = pd.read_csv('data/student-mat.csv')
df_por = pd.read_csv('data/student-por.csv')

# we are going to be focusing on aspects that parents can affect
investigated_attributes = [
    "address", # student's home address type (binary: 'U' - urban or 'R' - rural) 
    "Pstatus", # parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
    "Medu", # mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
    "Fedu", # father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
    "Mjob", # mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other') 
    "Fjob", # father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other') 
    "famsup", # family educational support (binary: yes or no) 
    "paid", # extra paid classes within the course subject (Math or Portuguese) (binary: yes or no) 
    "internet", # internet access at home (binary: yes or no) 
    "nursery", # nursery - attended nursery school (binary: yes or no) 
    "famrel", # quality of family relationships (numeric: from 1 - very bad to 5 - excellent),
    "Dalc", # workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
    "Walc" # weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
    ]

# only working with portuguese/mathematics
df = df_por.copy()
X = df[investigated_attributes].copy()
y = df["G3"].copy()

#%% Encoding
X_enc = X.copy()

# encoding nominal boolean variables
ohe = OneHotEncoder()

address_enc = pd.DataFrame(ohe.fit_transform(X[["address"]]).toarray())
address_enc.columns = ["rural", "urban"]
X_enc = X_enc.join(address_enc)

Pstatus_enc = pd.DataFrame(ohe.fit_transform(X[["Pstatus"]]).toarray())
Pstatus_enc.columns = ["apart", "together"]
X_enc = X_enc.join(Pstatus_enc)

famsup_enc = pd.DataFrame(ohe.fit_transform(X[["famsup"]]).toarray())
famsup_enc.columns = ["no support", "support"]
X_enc = X_enc.join(famsup_enc)

paid_enc = pd.DataFrame(ohe.fit_transform(X[["paid"]]).toarray())
paid_enc.columns = ["untutored", "tutored"]
X_enc = X_enc.join(paid_enc)

internet_enc = pd.DataFrame(ohe.fit_transform(X[["internet"]]).toarray())
internet_enc.columns = ["no internet", "online"]
X_enc = X_enc.join(internet_enc)

nursery_enc = pd.DataFrame(ohe.fit_transform(X[["nursery"]]).toarray())
nursery_enc.columns = ["no nursery", "nursed"]
X_enc = X_enc.join(nursery_enc)

# so that the attribute descriptions in correlation don't repeat
X_corr = X_enc.drop(["address", "rural",
                     "Pstatus", "together",
                     "famsup", "no support",
                     "internet", "no internet",
                     "nursery", "no nursery",
                     "paid", "untutored"],      axis=1)

# label variables would take up too much space on the correlation heatmap,
# so we do the correlation without them
Xy = X_corr.join(y)
Xy = Xy.rename(columns={"G3": "Grade"})

# encoding nominal categorical variables
Mjob_enc = pd.DataFrame(ohe.fit_transform(X[["Mjob"]]).toarray())
Mjob_enc.columns = ohe.get_feature_names(["Mjob"])
X_enc = X_enc.join(Mjob_enc)

Fjob_enc = pd.DataFrame(ohe.fit_transform(X[["Fjob"]]).toarray())
Fjob_enc.columns = ohe.get_feature_names(["Fjob"])
X_enc = X_enc.join(Fjob_enc)

# final encoded X array for PCA analysis and ML purposes
X_enc = X_enc.drop(["address", "Pstatus", "famsup", "paid", "internet",
                    "nursery", "Mjob", "Fjob"], axis=1)

# make a pass/fail bool classification of the grade for vis
y_passfail = (y.mask(y < 10, 0))
y_passfail = (y_passfail.mask(y_passfail > 0, 1))
y_passfail = y_passfail.rename("Passed")

# grades distributions
fig, ax = plt.subplots(1, 2, figsize=(10,4))
sns.histplot(data=y, ax=ax[0], binwidth=1, kde=True)
sns.histplot(data=y_passfail, ax=ax[1], binwidth=0.5)
ax[0].set_xlabel("Exam score (/20)")
ax[1].set_xlabel("Fail/Pass")
plt.tight_layout()
fig.savefig('./plots/grades_distro.pdf')

fig, ax = plt.subplots(1, 2, figsize=(10,6), sharey=True)
job_order = ["teacher", "health", "services", "other", "at_home"]
sns.boxplot(Xy["Mjob"], Xy["Grade"], ax=ax[0], order=job_order)
sns.boxplot(Xy["Fjob"], Xy["Grade"], ax=ax[1], order=job_order)
ax[0].set_xlabel("Mother's job")
ax[1].set_xlabel("Father's job")
ax[0].set_ylabel("Student's score (/20)")
ax[1].set_ylabel("")
plt.tight_layout()
fig.savefig('./plots/parents_job.pdf')

fig, ax = plt.subplots(1, 2, figsize=(10,6), sharey=True)
sns.boxplot(Xy["Medu"], Xy["Grade"], ax=ax[0])
sns.boxplot(Xy["Fedu"], Xy["Grade"], ax=ax[1])
ax[0].set_xlabel("Mother's education")
ax[1].set_xlabel("Father's education")
ax[0].set_ylabel("Student's score (/20)")
ax[1].set_ylabel("")
plt.tight_layout()
fig.savefig('./plots/parents_edu.pdf')

correlations = Xy.corr()
ix = abs(correlations).sort_values('Grade', ascending=False).index
correlations = correlations.loc[:, ix]

ut_mask = np.triu(np.ones_like(correlations, dtype=bool)) # Mask for upper triangle
cmap = sns.diverging_palette(220, 20, as_cmap=True) # Custom diverging colormap
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(data=correlations, mask=ut_mask, cmap=cmap, annot=True,
            linewidth=0.1)
fig.savefig('./plots/correlation_heatmap.pdf')

#%% PCA Analysis

N = len(X_enc)
M = len(X_enc.columns)
attributeNames = list(X_enc.columns)

# I have no idea if removing the mean is the thing to do here since we're 
# mainly working with boolean data, not doing it though gives the first
# principal component a variance explained of ~0.85, so this is the right
# way I think
Y = (X_enc - np.ones((N, 1))*(X_enc.mean(axis=0).to_numpy()))/X_enc.std()

U, S, Vh = np.linalg.svd(Y, full_matrices=False)

rho = (S*S) / (S*S).sum()

threshold = 0.9
pc_required = np.argmax(rho.cumsum()>threshold)
fig = plt.figure(figsize=(12,6))
sns.lineplot(x=range(1, len(rho)+1), y=rho, marker='+')
sns.lineplot(x=range(1, len(rho)+1), y=rho.cumsum(), marker='X')
plt.axhline(threshold, color='r', linestyle='-.')
plt.axvline(pc_required, color='r', linestyle='-.', label=f"{pc_required}")
plt.title("Variance Explained by Principal Component")
plt.legend(["Individual", "Cumulative"])
plt.xlabel("Principal Component #")
plt.ylabel("Variance Explained [-]")
fig.savefig('./plots/pca_ve.pdf')


V = Vh.T # Vh is transpose of V, columns of V are ~weights of the attributes
Z = Y @ V # Project centered data into PC space

number_of_pc_analysed = 5
number_of_main_attributes = 4
main_pc_weights = np.zeros((number_of_pc_analysed, number_of_main_attributes),
                           dtype=np.int8)

# this loop spits out the highest weighing attributes in the given principal component
for i in range(number_of_pc_analysed):
    main_pc_weights[i] = (abs(V[:,i]).argsort()[:number_of_main_attributes])
    print(f"The main PC{i} attributes are:")
    print([attributeNames[a] for a in main_pc_weights[i]])

# plot projected centered data into PC space
Z = Y @ V # Project centered data into PC space
Z = Z.values

C = 2 # fail/pass class

# principal component #s to project
i, j = 0, 1

fig = plt.figure(figsize=(10,6))
for c in range(2):
    print(c)
    class_mask = y_passfail==c
    sns.scatterplot(Z[class_mask, i],
                    Z[class_mask, j],
                    alpha=0.7)
plt.legend(["Failed", "Passed"])
plt.xlabel(f'Principal component #{i+1}')
plt.ylabel(f'Principal component #{j+1}')
fig.savefig('./plots/pca_projection.pdf')
#%%
X_enc = X_enc.drop(['urban',
            'together',
            'no support',
            'untutored',
            'no internet',
            'no nursery'],
           axis = 1)

#%% # exercise 8.1.1
y = np.array(y)
y = y.astype(float)
X_enc = np.array(Y)

# Add offset attribute
X              = np.concatenate((np.ones((X_enc.shape[0],1)),X_enc),1) # Changed X in function to X_enc
attributeNames = [u'Offset']+attributeNames
M              = M+1

#%% Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
#CV = model_selection.KFold(K, shuffle=True)
CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-1,6)) # Original: range(-5,9)

# Initialize variables
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X,y):
    
# =============================================================================
#     # extract training and test set for current CV fold
#     X_train = X[train_index]
#     N = len(X_train)
#     i = np.linspace(0,stop=N-1,num=N)
#     y_train = y[train_index]
#     y_train = y_train.reindex(i)
#     y_train = y_train.fillna(0)
#     X_test = X[test_index]
#     y_test = y[test_index]
#     internal_cross_validation = 10    
# =============================================================================

    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10 
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:],0)
    sigma[k, :] = np.std(X_train[:, 1:],0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    
    # Compute mean squared error without regularization
    #Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    #Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    m = lm.LinearRegression().fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        fig.savefig('./plots/LR.pdf')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()

# fig = plt.figure(figsize=(12,6))    
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()

# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

print('Ran Exercise 8.1.1')

#%% exercise 8.2.2
# ANN part

y = np.array([y_passfail])
y = y.transpose()
classNames = ['Class 1','Class 2']
# classNames = ['Class 1','Class 2','Class 3','Class 4'] # Don't know how many classes belong here

X = np.array(X_enc)
print(f"LENGTH = {len(X[0])}")
M = len(X[0])

plt.rcParams.update({'font.size': 12})

# K-fold CrossValidation
K = 5                       # Change back to 10
CV = model_selection.KFold(K,shuffle=True)

# Setup figure for display of the decision boundary for the several crossvalidation folds.
decision_boundaries = plt.figure(1, figsize=(10,10))
# Determine a size of a plot grid that fits visualizations for the chosen number
# of cross-validation splits, if K=4, this is simply a 2-by-2 grid.
subplot_size_1 = int(np.floor(np.sqrt(K))) 
subplot_size_2 = int(np.ceil(K/subplot_size_1))
# Set overall title for all of the subplots
plt.suptitle('Data and model decision boundaries', fontsize=20)
# Change spacing of subplots
plt.subplots_adjust(left=0, bottom=0, right=1, top=.9, wspace=.5, hspace=0.25)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

# Define the model structure
n_hidden_units = 3 # number of hidden units in the signle hidden layer
# The lambda-syntax defines an anonymous function, which is used here to 
# make it easy to make new networks within each cross validation fold
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.Tanh(),                            #torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                    torch.nn.Sigmoid() # final tranfer function
                    )
# Since we're training a neural network for binary classification, we use a 
# binary cross entropy loss (see the help(train_neural_net) for more on
# the loss_fn input to the function)
loss_fn = torch.nn.BCELoss()
# Train for a maximum of 10000 steps, or until convergence (see help for the 
# function train_neural_net() for more on the tolerance/convergence))
max_iter = 10000
print('Training model of type:\n{}\n'.format(str(model())))

# Do cross-validation:
errors = [] # make a list for storing generalizaition error in each loop
# Loop over each cross-validation split. The CV.split-method returns the 
# indices to be used for training and testing in each split, and calling 
# the enumerate-method with this simply returns this indices along with 
# a counter k:
for k, (train_index, test_index) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, 
    # and convert them to PyTorch tensors
    X_train = torch.Tensor(X[train_index,:] )
    y_train = torch.Tensor(y[train_index] )
    X_test = torch.Tensor(X[test_index,:] )
    y_test = torch.Tensor(y[test_index] )
    
    # Go to the file 'toolbox_02450.py' in the Tools sub-folder of the toolbox
    # and see how the network is trained (search for 'def train_neural_net',
    # which is the place the function below is defined)
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=3,  # Change this 
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_sigmoid = net(X_test) # activation of final note, i.e. prediction of network
    y_test_est = (y_sigmoid > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    e = (y_test_est != y_test)
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    errors.append(error_rate) # store error rate for current CV fold 
    
    # Make a subplot for current cross validation fold that displays the 
    # decision boundary over the original data, "background color" corresponds
    # to the output of the sigmoidal transfer function (i.e. before threshold),
    # white areas are areas of uncertainty, and a deaper red/blue means 
    # that the network "is more sure" of a given class.
    plt.figure(decision_boundaries.number)
    plt.subplot(subplot_size_1,subplot_size_2,k+1)
    plt.title('CV fold {0}'.format(k+1),color=color_list[k])
    predict = lambda x: net(torch.tensor(x, dtype=torch.float)).data.numpy()
    
    # Below works ONLY for M=2
    # visualize_decision_boundary(predict, X, y, # provide data, along with function for prediction
    #                             attributeNames, classNames, # provide information on attribute and class names
    #                             train=train_index, test=test_index, # provide information on partioning
    #                             show_legend=k==(K-1)) # only display legend for last plot
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
# Display the error rate across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('Error rate');
summaries_axes[1].set_title('Test misclassification rates')
    
# Show the plots
# plt.show(decision_boundaries.number) # try these lines if the following code fails (depends on package versions)
# plt.show(summaries.number)
plt.show()

# Display a diagram of the best network in last fold
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,3]]
draw_neural_net(weights, biases, tf)

# Print the average classification error rate
print('Artifical Neural Network:')
print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(errors),4)))

print('Linear regression without feature selection:')
print('- Test error:     {0}'.format(Error_test.mean()))

print('Baseline:')
print('- Test error:     {0}'.format(Error_test_nofeatures.mean()))

#% Classification
#%% exercise 8.1.2 - Logistic Regression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

font_size = 15
plt.rcParams.update({'font.size': font_size})

# Load Matlab data file and extract variables of interest
#mat_data = loadmat('../Data/wine2.mat')
X = X_enc
y = y.squeeze()
#attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
#classNames = [name[0][0] for name in mat_data['classNames']]
#N, M = X.shape
#C = len(classNames)

# Create crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test 
K = 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, stratify=y)
# Try to change the test_size to e.g. 50 % and 99 % - how does that change the 
# effect of regularization? How does differetn runs of  test_size=.99 compare 
# to eachother?

# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# Fit regularized logistic regression model to training data to predict 
# the type of wine
lambda_interval = np.logspace(-1, 6, 50) #np.logspace(-8, 2, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))
for k in range(0, len(lambda_interval)):
    mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
    
    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = mdl.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]

plt.figure(figsize=(10,10))
plt.plot(np.log10(lambda_interval), train_error_rate*100)
plt.plot(np.log10(lambda_interval), test_error_rate*100)
plt.plot(np.log10(opt_lambda), min_error*100, 'o')
#plt.semilogx(lambda_interval, train_error_rate*100)
#plt.semilogx(lambda_interval, test_error_rate*100)
#plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-8, 1, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
#plt.ylim([0, 28])
plt.grid()
plt.show()    

plt.figure(figsize=(10,10))
plt.semilogx(lambda_interval, coefficient_norm,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()    

print('Ran Project 2')

#%% exercise 8.3.3 Fit regularized multinomial regression model

y = np.array([y_passfail])
y = y.transpose()
y = y.squeeze()
classNames = ['Class 1','Class 2']
# classNames = ['Class 1','Class 2','Class 3','Class 4'] # Don't know how many classes belong here
X = np.array(X_enc[:,0:2])
M = len(X[0])

# =============================================================================
# X_train = X[train_index,:] 
# y_train = y[train_index] 
# X_test = X[test_index,:] 
# y_test = y[test_index] 
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, stratify=y)
#attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
#classNames = [name[0][0] for name in mat_data['classNames']]

#N, M = X.shape
#C = len(classNames)

#%% Model fitting and prediction
# Standardize data based on training set
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# Fit multinomial logistic regression model
regularization_strength = 1e-5 # 1e-3
#Try a high strength, e.g. 1e5, especially for synth2, synth3 and synth4
mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                               tol=1e-4, random_state=1, 
                               penalty='l2', C=1/regularization_strength)
mdl.fit(X_train,y_train)
y_test_est = mdl.predict(X_test)

test_error_rate = np.sum(y_test_est!=y_test) / len(y_test)

predict = lambda x: np.argmax(mdl.predict_proba(x),1)
plt.figure(2,figsize=(9,9))
visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
plt.title('LogReg decision boundaries')
plt.show()


# Number of miss-classifications
print('Error rate: \n\t {0} % out of {1}'.format(test_error_rate*100,len(y_test)))
# %%

plt.figure(2, figsize=(9,9))
plt.hist([y_train, y_test, y_test_est], color=['red','green','blue'], density=True)
plt.legend(['Training labels','Test labels','Estimated test labels'])


print('Ran Exercise 8.3.2')

