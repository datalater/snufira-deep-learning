
# coding: utf-8

# In[10]:



import numpy as np
import csv

class reader(object):                           # class를 선언할 때 object를 상속받는다. (convention)
    def __init__(self, data_file="./data/train.csv"):
        self.value = []
        with open(data_file, "rb") as f:        # binary 모드 안 되면 "r"로 해라
            csv_reader = csv.reader(f, delimiter=",")
            
            for i, row in enumerate(csv_reader):
                if i == 0:
                    self.attribute = row         # Q. attribute는 코드에서 안 보이는데 class reader의 함수인지?
                    continue
                
                self.value.append(row)
        
        self.raw_to_vector(self.value)
        self.split(num_validation_examples=100)

        # self.x <= Pclass:[2], Sex:[4], Age:[5], SibSp:[6], Parch:[7], fare:[9]
        # self.y <= Survived:[1]
        
        # self.x_train.shape = (791, 6)
        # self.y_train.shape = (791)
        # self.id_train.shape = (791)

        # self.x_val.shape = (100, 6)
        # self.y_val.shape = (100)
        # self.id_val.shape = (100)

        self.num_examples = len(self.x_train)    # = 791 (train_data)
        self.start_index = 0
        self.shuffle_indices = range(self.num_examples)

        self.num_examples_val = len(self.x_val)  # = 100 (validation_data)
        self.start_index_val = 0
        self.shuffle_indices_val = range(self.num_examples_val)

        
    def raw_to_vector(self, value):
        self.x = []
        self.y = []
        self.id = []
        
        for row in self.value:
            x = np.zeros(6)
            
            x[0] = float(row[2])
            
#             if row[4] == "male":
#                 x[1] = 0
#             else:
#                 x[1] = 1
            
            x[1] = 0 if row[4] == "male" else 1     # male : 0, female : 1
            x[2] = float(row[5]) if row[5] else 30  # 나이가 없는 데이터에는 30세로 넣는다 (prior).
            x[3] = float(row[6])
            x[4] = float(row[7])
            x[5] = float(row[9]) 
            
            y = np.zeros(1)
            y[0] = int(row[1])
            
            id = np.zeros(1)
            id[0] = float(row[0])
        
            self.x.append(x)
            self.y.append(y)
            self.id.append(id)
        
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.id = np.array(self.id)

    
    def split(self, num_validation_examples):
        self.x_train = self.x[ num_validation_examples: ]
        self.x_val = self.x[ : num_validation_examples ]

        self.y_train = self.y[ num_validation_examples: ]
        self.y_val = self.y[ : num_validation_examples ]

        self.id_train = self.id[ num_validation_examples: ]
        self.id_val = self.id[ :num_validation_examples ]
    
    
    def next_batch(self, batch_size, split="train"):

        if split == "train":
            if self.start_index == 0:
                np.random.shuffle(self.shuffle_indices) # shuffle indices
            
            end_index = min([self.num_examples, self.start_index + batch_size])
            batch_indices = [ self.shuffle_indices[idx] for idx in range(self.start_index, end_index) ]

            batch_x = self.x_train[ batch_indices ]
            batch_y = self.y_train[ batch_indices ]
            batch_id = self.id_train[ batch_indices ] 

            if end_index == self.num_examples:
                self.start_index = 0
            else: self.start_index = end_index

            return batch_x, batch_y, batch_id 

        elif split == "val":
            if self.start_index_val == 0:
                np.random.shuffle(self.shuffle_indices_val) # shuffle indices

            end_index = min([self.num_examples_val, self.start_index_val + batch_size])
            batch_indices = [ self.shuffle_indices_val[idx] for idx in range(self.start_index_val, end_index) ]

            batch_x = self.x_val[ batch_indices ]
            batch_y = self.y_val[ batch_indices ]
            batch_id = self.id_val[ batch_indices ] 

            if end_index == self.num_examples_val:
                self.start_index_val = 0
            else: self.start_index_val = end_index

            return batch_x, batch_y, batch_id  
        

#---Execution code---#
    
# c = reader()
# c.next_batch(20, "train")


# In[ ]:



