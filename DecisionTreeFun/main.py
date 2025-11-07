import numpy as np

X_train = [
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
y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
# TODO: in fit(), programmatically build header and attribute_domains
# using X_train. perhaps store as attributes of MyDecisionTreeClassifier
header = ["att0", "att1", "att2", "att3"]
attribute_domains = {"att0": ["Junior", "Mid", "Senior"], 
        "att1": ["Java", "Python", "R"],
        "att2": ["no", "yes"], 
        "att3": ["no", "yes"]}

# How to represent decision trees in Python:
# 1. common approaches:
#    - Nested data structures (lists or dictionaries)
#    - Object-oriented (e.g., TreeNode class)
#
# 2. We will use nested lists:
#    - Element 0: data type ("Attribute", "Value", "Leaf")
#    - Element 1: data value
#       - "Attribute"-> attribute name (e.g., "att0")
#       - "Value" -> attribute value (e.g., "Senior")
#       - "Leaf" -> class label (e.g., "True")
#    - Remaining elements: depends on the type
# Example tree:
interview_tree_solution =   ["Attribute", "att0",                        # Root node: split on att0
                                ["Value", "Junior",                      # value for att0 = Junior
                                    ["Attribute", "att3",                # Split on att3 inside Junior
                                        ["Value", "no",
                                            ["Leaf", "True", 3, 5]       # Leaf node: class = True
                                        ],
                                        ["Value", "yes",
                                            ["Leaf", "False", 2, 5]
                                        ]
                                    ]
                                ],
                                ["Value", "Mid",
                                    ["Leaf", "True", 4, 14] 
                                ],
                                ["Value", "Senior",
                                    ["Attribute", "att2",
                                        ["Value", "no",
                                            ["Leaf", "False", 3, 5] 
                                        ],
                                        ["Value", "yes",
                                            ["Leaf", "True", 2, 5] 
                                        ]
                                    ] 
                                ]
                            ]

def select_attribute(instances, attributes):
    # TODO: implement the general Enew algorithm for attribute selection
    # for each available attribute
    #     for each value in the attribute's domain
    #          calculate the entropy for the value's partition
    #     calculate the weighted average for the parition entropies
    # select that attribute with the smallest Enew entropy
    # for now, select an attribute randomly
    rand_index = np.random.randint(0, len(attributes))
    return attributes[rand_index]

def partition_instances(instances, attribute):
    # this is group by attribute domain (not values of attribute in instances)
    # Returns a dictionary: {attribute_value: [instances]}
    att_index = header.index(attribute)
    att_domain = attribute_domains[attribute]
    partitions = {}
    for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions

def all_same_class(instances):
    # get the class label of the first instance.
    first_class = instances[0][-1]
    for instance in instances:
        # if any label differs, return False immediately.
        if instance[-1] != first_class:
            return False
        
    # if the loop completes without finding differences, return True.
    return True 

def tdidt(current_instances, available_attributes):
    
    #    Recursively building a decision tree using the TDIDT algorithm.

    #     1. Select the best attribute to split on and create an "Attribute" node.
    #     2. For each value of the selected attribute:
    #         a. Create a "Value" subtree.
    #         b. If all instances in this partition have the same class:
    #             - Append a "Leaf" node
    #         c. If there are no more attributes to select:
    #             - Append a "Leaf" node (handle clash w/majority vote leaf node)
    #         d. If the partition is empty:
    #             - Append a "Leaf" node (backtrack and replace attribute node with majority vote leaf node)
    #         e. Otherwise:
    #             - Recursively build another "Attribute" subtree for this partition
    #               and append it to the "Value" subtree.
    #     3. Append each "Value" subtree to the current "Attribute" node.
    #     4. Return the current tree (nested list structure).

    

    print("available attributes:", available_attributes)
    
    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes)
    print("splitting on:", split_attribute)
    available_attributes.remove(split_attribute) # can't split on this attribute again in this subtree

    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute)
    print("partitions:", partitions)
    
    # for each partition, repeat unless one of the following occurs (base case)
    for att_value in sorted(partitions.keys()): # process in alphabetical order
        att_partition = partitions[att_value]
        value_subtree = ["Value", att_value]

        #    CASE 1: all class labels of the partition are the same
        # => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            print("CASE 1")

        #    CASE 2: no more attributes to select (clash)
        # => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            print("CASE 2")

        #    CASE 3: no more instances to partition (empty partition)
        # => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            print("CASE 3")

        else:
            # none of base cases were true, recurse!!
            subtree = tdidt(att_partition, available_attributes.copy())
            # TODO: append subtree to value_subtree and value_subtree to tree appropriately
    return tree


def tdidt_predict(tree, instance):
    data_type = tree[0]

    # Base case: if this is a leaf, just return its class label
    if data_type == "Leaf":
        label = tree[1]
        return label
    
    # Recursive case:if we are here, this is an Attribute node
    attribute_name = tree[1]
    attribute_index = header.index(attribute_name)
    instance_value = instance[attribute_index]

    # Look for the matching value node
    for values in tree[2:]:
        value = values[1]
        subtree = values[2]
        
        if instance_value == value:
            return tdidt_predict(subtree, instance)

    

def fit_starter_code():
    # note the TODO above
    # here would be a good place to programmatically extract
    # the header and attribute_domains
    # lets stich together X_train and y_train
    train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
    # print(train)
    
    # make a copy a header, b/c python is pass by object reference
    # and tdidt will be removing attributes from available_attributes
    available_attributes = header.copy()
    tree = tdidt(train, available_attributes)
    print("tree:", tree)
    # your unit test will assert tree == interview_tree_solution


def predict_starter_code():
    # we need test instances
    instance1 = ["Junior", "Java", "yes", "no"] # True
    instance2 = ["Junior", "Java", "yes", "yes"] # False
    instance3 = ["Intern", "Java", "yes", "yes"] # None
    prediction = tdidt_predict(interview_tree_solution, instance1)
    print("prediction:", prediction)

fit_starter_code()
predict_starter_code()

