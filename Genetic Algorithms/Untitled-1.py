# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
import kagglehub
path = kagglehub.dataset_download("uciml/autompg-dataset")
# print("Path to dataset files:", path)

# %%
autompg = pd.read_csv(f"{path}/auto-mpg.csv")
# autompg

# %%
autompg.replace('?', np.nan, inplace=True)
autompg.dropna(inplace=True)
# autompg

# %%


# %%
# corr_matrix = autompg.corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix of Auto-MPG Dataset')
# plt.show()


# %%
target = autompg['mpg']
autompg.drop('car name', axis=1, inplace=True)
# autompg

# %% [markdown]
# # Linear Regression

# %% [markdown]
# Normalize

# %%
def normalize(X):

    for i in X.columns:
        X[i] = (X[i] - X[i].mean())/X[i].std()
            
    return X

# %% [markdown]
# Gradient Descend

# %%
def gradient_descend(X, y, learning_rate = 0.01, n_iter = 1000):
    X = normalize(X)
    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros(X.shape[1])
    m = X.shape[0]
    
    for i in range(n_iter):
        h = X @ theta
        loss = (h - y)
        gradient = X.T @ loss / m
        theta = theta - (learning_rate * gradient)
    return theta

# %%
# autompg.dtypes

# %%
# numeric_autompg = autompg.drop(columns=['mpg']).apply(pd.to_numeric, errors='coerce')
# numeric_autompg.dropna(axis=1, inplace=True)
# gradient_descend(numeric_autompg, target)


# %% [markdown]
# # Define Fitness Function
# 
# Here we are using $\textbf{Akaike Information Criterion}$ which is given by the formula
#     
# $AIC = 2k - 2ln(\hat{L})$
# 
# where $k$ = number of estimated parameters 
#       $\hat{L}$ = maximised value of likelihood function
#  

# %%
def rss(X,y,theta):
    X = normalize(X)
    X = np.c_[np.ones(X.shape[0]), X]
    residual_sum_squares = sum(((X @ theta) - y)**2)
    return residual_sum_squares

def calculate_aic(X,y,theta):
    n = len(y)
    aic = 2 * X.shape[1] + n * np.log(rss(X,y,theta) / n)
    return aic    

# %% [markdown]
# # Generate Parent Population

# %%
# noinspection PyShadowingNames
def parent_pops(data,size,y):
    papa_pops = []
    df_temp = data.drop(y, axis = 1)
    for i in range(size):
        feature_arrs = np.random.choice([True,False],size = df_temp.shape[1])
        papa_pops.append(df_temp.iloc[:, np.where(feature_arrs)[0]])
    return papa_pops

# %% [markdown]
# # Calculate Fitness Score

# %%
def parent_score(parents, y):
    pop_score = {}
    temp_var = 0
    for parent in parents:
        theta1 = gradient_descend(parent, y, learning_rate = 0.01, n_iter = 1000)
        aic = -1*calculate_aic(parent, y, theta1)
        pop_score[temp_var] = aic
        temp_var += 1
    return pop_score

# %% [markdown]
# # Select n Fittest Parents

# %%
def fittest_parents(population,lamda,y):

    parent_scores = parent_score(population,y)
    fittest_parent = []
    sorted_parents = sorted(parent_scores.items(), key = lambda x:x[1], reverse = True)
    top_lamda_parents_indices = sorted_parents[:lamda]
    top_lamda_parents = [item[0] for item in top_lamda_parents_indices]
    
    for i in top_lamda_parents:
        fittest_parent.append(population[i])
    return fittest_parent

# %% [markdown]
# # Uniform Crossover

# %%
def uniform_crossover(data,target_variable,parent1, parent2):

    X = data.drop(target_variable, axis = 1) 
    parent_list1 = parent1.columns.tolist()
    # temp = 0
    
    bool_arr1 = np.zeros(X.shape[1])
    for i in parent_list1:
        bool_arr1[X.columns.get_loc(i)] = 1
        
    bool_arr2 = np.zeros(X.shape[1])
    parent_list2 = parent2.columns.tolist()
    for i in parent_list2:
        bool_arr2[X.columns.get_loc(i)] = 1
        
    for i in range(len(bool_arr1)):
        if np.random.uniform() >= 0.5:
            temp = bool_arr1[i]
            bool_arr1[i] = bool_arr2[i]
            bool_arr2[i] = temp
            continue
            
    new_parent1 = X.iloc[:, np.where(bool_arr1)[0]]
    new_parent2 = X.iloc[:, np.where(bool_arr2)[0]]
    return  new_parent1, new_parent2

# %% [markdown]
# # Random Bit-Flip Mutation

# %%
def rbf_mutation(data,target_variable,parent1, parent2,mutation_rate):
    X = data.drop(target_variable, axis = 1)
    
    bool_arr1 = np.zeros(X.shape[1])
    parent_list1 = parent1.columns.tolist()
    for i in parent_list1:
        bool_arr1[X.columns.get_loc(i)] = 1
        
    bool_arr2 = np.zeros(X.shape[1])
    parent_list2 = parent2.columns.tolist()
    for i in parent_list2:
        bool_arr2[X.columns.get_loc(i)] = 1
        
    for i in range(len(bool_arr1)):
        if mutation_rate > np.random.normal(loc = .5, scale = .25, size = 1):
            bool_arr1[i] = 1-bool_arr1[i]
            continue
    for i in range(len(bool_arr2)):
        if mutation_rate > np.random.normal(loc = .5, scale = .25, size = 1):
            bool_arr2[i] = 1-bool_arr2[i]
            continue
            
    new_parent1 = X.iloc[:, np.where(bool_arr1)[0]]
    new_parent2 = X.iloc[:, np.where(bool_arr2)[0]]    
    
    return new_parent1, new_parent2

# %% [markdown]
# # The $(\mu , \lambda)$ Genetic Algorithm
# 
# 

# %%
def gen_alg(data,target_variable,pop_size,no_parents_selected,iterations,mutation_rate): 
    # data = data.drop(target_variable, axis = 1)
    population = parent_pops(data,pop_size,target_variable)
    
    for z in range(iterations):
        # print("iteration:",z)
        parents = fittest_parents(population,no_parents_selected,target_variable)
        # print("parents:",parents)
        new_gen = []
        for i in range(len(parents)):
            for j in range(i+1, len(parents)):
                parent1 = parents[i]
                parent2 = parents[j]
                if np.random.normal(loc = .5 , scale = .25 , size = 1) < 0.5:
                    offspring1, offspring2 = uniform_crossover(data, target_variable, parent1, parent2)
                else:
                    offspring1, offspring2 = rbf_mutation(data, target_variable, parent1, parent2, mutation_rate)
                new_gen.append(offspring1)
                new_gen.append(offspring2)
                
        population = new_gen

    return fittest_parents(population,1,target_variable)

# %%
data = autompg.apply(pd.to_numeric, errors='coerce')
data.dropna(axis=1, inplace=True)  # Drop columns that couldn't be converted to numeric
target_variable = 'mpg'
pop_size = 100
no_parents_selected = 20
iterations = 100
mutation_rate = 0.3

# Ensure all columns are numeric
data = data.apply(pd.to_numeric, errors='coerce')

gen_alg(data, target_variable, pop_size, no_parents_selected, iterations, mutation_rate)

# %%



