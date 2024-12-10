# Importing important libraries  
import pandas as pd 
from pandas import DataFrame  
# Reading Dataset  
df_tennis = pd.read_csv('C:/Users/Lochan/OneDrive/Documents/P4Data.csv') 
print(df_tennis)

# Function to calculate final Entropy  
def entropy(probs):   
    import math 
    return sum([-prob*math.log(prob,2) for prob in probs]) 

# Function to calculate Probabilities of positive and negative examples  
def entropy_of_list(a_list): 
    from collections import Counter 
    cnt = Counter(x for x in a_list)  
    #Count the positive and negative ex 
    num_instances = len(a_list) 
    #Calculate the probabilities that we required for our entropy formula  
    probs = [x / num_instances for x in cnt.values()]  
    #Calling entropy function for final entropy  
    return entropy(probs) 

total_entropy = entropy_of_list(df_tennis['PT'])
print("\nTotal Entropy of PlayTennis Data Set:",total_entropy)

# Defining Information Gain Function  
def information_gain(df, split_attribute_name, target_attribute_name, trace=0): 
    print("\nInformation Gain Calculation of ",split_attribute_name) 
    print("target_attribute_name:",target_attribute_name) 
    
    # Grouping features of Current Attribute 
    df_split = df.groupby(split_attribute_name) 
    for name,group in df_split:
        print("Name: ",name) 
        print("Group: ",group) 
    nobs = len(df.index) * 1.0 
    print("NOBS",nobs)
    
    # Calculating Entropy of the Attribute and probability part of formula  
    df_agg_ent = df_split.agg(
        Entropy=(target_attribute_name, entropy_of_list), 
        Prob1=(target_attribute_name, lambda x: len(x) / nobs)
    )
    print("df_agg_ent",df_agg_ent) 
    
    # Calculate Information Gain 
    avg_info = sum(df_agg_ent['Entropy'] * df_agg_ent['Prob1']) 
    old_entropy = entropy_of_list(df[target_attribute_name]) 
    return old_entropy - avg_info
    
print('Info-gain for Outlook is : '+str(information_gain(df_tennis, 'Outlook', 'PT')),"\n") 

# Defining ID3  Algorithm Function 
def id3(df, target_attribute_name, attribute_names, default_class=None):
    # Counting Total number of yes and no classes (Positive and negative Ex)
    from collections import Counter 
    cnt = Counter(x for x in df[target_attribute_name]) 
    if len(cnt) == 1:
        return next(iter(cnt))
    # Return None for Empty Data Set  
    elif df.empty or (not attribute_names): 
        return default_class 
    else:
        default_class = max(cnt.keys())
        print("attribute_names:",attribute_names) 
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]  
        # Separating the maximum information gain attribute after calculating the information gain  
        index_of_max = gainz.index(max(gainz)) #Index of Best Attribute  
        best_attr = attribute_names[index_of_max] #choosing best attribute  
        # The tree is initially an empty dictionary 
        tree = {best_attr:{}} # Initiate the tree with best attribute as a node  
        remaining_attribute_names = [i for i in attribute_names if i != best_attr] 
        for attr_val, data_subset in df.groupby(best_attr): 
            subtree = id3(data_subset, target_attribute_name, remaining_attribute_names, default_class)
            tree[best_attr][attr_val] = subtree 
        return tree 
    
# Get Predictor Names (all but 'class') 
attribute_names = list(df_tennis.columns) 
print("List of Attributes:", attribute_names)  
attribute_names.remove('PT') 
# Remove the class attribute  
print("Predicting Attributes:", attribute_names) 

# Run Algorithm (Calling ID3 function) 
from pprint import pprint 
tree = id3(df_tennis,'PT',attribute_names) 
print("\n\nThe Resultant Decision Tree is :\n") 
pprint(tree)
attribute = next(iter(tree)) 
print("Best Attribute :\n",attribute) 
print("Tree Keys:\n",tree[attribute].keys())

# Defining a function to calculate accuracy 
def classify(instance, tree, default=None): 
    attribute = next(iter(tree)) 
    print("Key:",tree.keys()) 
    print("Attribute:",attribute) 
    print("Insance of Attribute :",instance[attribute],attribute) 
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]] 
        print("Instance Attribute:",instance[attribute],"TreeKeys :",tree[attribute].keys()) 
        if isinstance(result, dict):  
            return classify(instance, result) 
        else: 
            return result  
    else: 
        return default 

df_tennis['predicted'] = df_tennis.apply(classify, axis=1, args=(tree,'No') )  
print(df_tennis['predicted']) 
print('\n Accuracy is:\n' + str( sum(df_tennis['PT']==df_tennis['predicted'] ) / (1.0*len(df_tennis.index)) )) 
df_tennis[['PT', 'predicted']] 
 
training_data = df_tennis.iloc[1:-4]  
test_data  = df_tennis.iloc[-4:] 
train_tree = id3(training_data, 'PT', attribute_names) 
test_data['predicted2'] = test_data.apply(   
classify, axis=1, args=(train_tree,'Yes') )  
print ('\n\n Accuracy is : ' + str( sum(test_data['PT']==test_data['predicted2'] ) / (1.0*len(test_data.index)) ))