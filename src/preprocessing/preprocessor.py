import pandas as pd
import copy

class Preprocessor:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        
    def get_encoded_dataset(self, selected_features: list):
        temp_dataset = copy.deepcopy(self.dataset)
        # extract selected features
        temp_dataset = temp_dataset[selected_features]
        # one-hot encoding
        temp_dataset = pd.get_dummies(temp_dataset)
        return temp_dataset
        
    def fillna_by_median(self, column: str):
        self.dataset[column].fillna(self.dataset[column].median(), inplace=True)
        
    def fillna_by_specific_value(self, column: str, value):
        self.dataset[column].fillna(value, inplace=True)
        
    def create_title(self):
        self.dataset['Title'] = self.dataset['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
        self.dataset['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
        self.dataset['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)
        self.dataset['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
        self.dataset['Title'].replace(['Mlle'], 'Miss', inplace=True)
        self.dataset['Title'].replace(['Jonkheer'], 'Master', inplace=True)
        
    def grouping_familiy(self):
        self.dataset['Surname'] = self.dataset['Name'].map(lambda name: name.split(',')[0].strip())
        self.dataset['FamilyGroup'] = self.dataset['Surname'].map(self.dataset['Surname'].value_counts())
        # Familiy = SibSp + Parch + 1 and then categorize
        self.dataset['Family'] = self.dataset['SibSp'] + self.dataset['Parch'] + 1
        self.dataset.loc[(self.dataset['Family']>=2) & (self.dataset['Family']<=4), 'FamilyLabel'] = 2
        self.dataset.loc[(self.dataset['Family']>=5) & (self.dataset['Family']<=7) | (self.dataset['Family']==1), 'FamilyLabel'] = 1
        self.dataset.loc[(self.dataset['Family']>7), 'FamilyLabel'] = 0