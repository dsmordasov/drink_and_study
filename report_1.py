#%% imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

sns.set()
# LaTeX textwidth = 469.47049pt
#%% data 
# https://www.kaggle.com/uciml/student-alcohol-consumption
df_mat = pd.read_csv('./student-mat.csv')
df_por = pd.read_csv('./student-por.csv')

# The below is the 'recommended' merge from Kaggle, but it makes no sense - described in the report
# =============================================================================
# # below were provided in the R merge file along with the dataset
# shared_attributes = [
#     "school", "sex", "age", "address", "famsize", "Pstatus","Medu", "Fedu",
#     "Mjob","Fjob", "reason", "nursery", "internet"]
# 
# # 382 students belong to both datasets
# # data suffix _x for mathematics, _y for portuguese
# df_r = pd.merge(
#     df_1of2, df_2of2,
#     on=investigated_attributes)
# =============================================================================

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

# =============================================================================
# df = pd.merge(
#     df_1of2, df_2of2,
#     on=investigated_attributes)
# =============================================================================

# only working with portuguese/mathematics
df = df_por.copy()
X = df[investigated_attributes].copy()
y = df["G3"].copy()

# encoding
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
                 "paid", "untutored"],
                    axis=1)

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
                    "nursery", "Mjob", "Fjob"],
                   axis=1)

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



# =============================================================================
# # grades distributions
# fig, ax = plt.subplots(1, 2, figsize=(10,4), sharey=True)
# sns.histplot(data=df['G3_x'], ax=ax[0], binwidth=1, kde=True)
# sns.histplot(data=df['G3_y'], ax=ax[1], binwidth=1, kde=True)
# ax[0].set_xlabel("Maths score (/20)")
# ax[1].set_xlabel("Portuguese score (/20)")
# plt.tight_layout()
# =============================================================================


# =============================================================================
# # grades boxplots wrt address
# one_attribute = "address" #can be changed
# fig, ax = plt.subplots(1, 2, figsize=(10,4), sharey=True)
# sns.boxplot(df[one_attribute], df["G3_x"], ax=ax[0])
# sns.boxplot(df[one_attribute], df["G3_y"], ax=ax[1])
# fig.suptitle(one_attribute)
# ax[0].set_xlabel("Maths")
# ax[1].set_xlabel("Portuguese")
# ax[0].set_ylabel("Score (/20)")
# ax[1].set_ylabel("")
# plt.tight_layout()
# =============================================================================

# =============================================================================
# # boxplots of all the investigated attributes
# fig, ax = plt.subplots(len(investigated_attributes), 2, figsize=(15, 28), sharey=True)
# attribute_index = 0
# for attribute in investigated_attributes:
#     ax[attribute_index, 0].set_ylim([0, 20])
#     sns.boxplot(df[attribute], df["G3_x"], ax=ax[attribute_index, 0])
#     sns.boxplot(df[attribute], df["G3_y"], ax=ax[attribute_index, 1])
#     ax[attribute_index, 0].set_ylabel(attribute)
#     attribute_index += 1
# =============================================================================


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
number_of_main_attributes = 5
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
