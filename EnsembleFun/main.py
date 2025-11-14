import numpy as np

# pasted from DecisionTreeFun
header = ["att0", "att1", "att2", "att3"]
attribute_domains = {"att0": ["Junior", "Mid", "Senior"], 
        "att1": ["Java", "Python", "R"],
        "att2": ["no", "yes"], 
        "att3": ["no", "yes"]}
X = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]

y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
# stitch X and y together to make one table
table = [X[i] + [y[i]] for i in range(len(X))]


# Ensemble Lab Task 1: 
# Write a bootstrap function to return a random sample of rows with replacement
# (test your function with the interview dataset)
def compute_bootstrapped_sample(table):
    n = len(table)
    # np.random.randint(low, high) returns random integers from low (inclusive) to high (exclusive)
    sampled_indexes = [np.random.randint(0, n) for _ in range(n)]
    sample = [table[index] for index in sampled_indexes]
    out_of_bag_indexes = [index for index in list(range(n)) if index not in sampled_indexes]
    out_of_bag_sample = [table[index] for index in out_of_bag_indexes]
    return sample, out_of_bag_sample

training_sample, validation_sample = compute_bootstrapped_sample(table)
print("training instances:")
for row in training_sample:
    print(row)
print("validation instances:")
for row in validation_sample:
    print(row)



# Ensemble Lab Task 2:
# Define a python function that selects F random attributes from an attribute list
# could just call np.random.choice()
def compute_random_subset(values, num_values):
    # let's use np.random.shuffle()
    values_copy = values.copy()
    np.random.shuffle(values_copy) # inplace
    return values_copy[:num_values]

# (test your function with att_indexes (or header))
F = 2
print(compute_random_subset(header, F))


# Random Forest Implementation Notes:
# 1. We are building an ensemble of decision trees (all base learners are trees).
# 2. Use bagging: for each tree, generate a bootstrap sample of the training data.
# 3. During tree construction:
#    - At each node, instead of considering all attributes for splitting,
#      randomly select a subset of F attributes.
#    - Call compute_random_subset() to get this subset.
#    - Pass this subset to select_attribute() to choose the best split.
# 4. Repeat for all nodes in the tree.
# 5. Aggregate predictions from all trees (majority vote for classification).
# This decorrelates trees and improves ensemble performance compared to simple bagging.
