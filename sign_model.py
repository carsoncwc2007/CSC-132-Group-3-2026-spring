import numpy as np
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class SignLanguageModel:
    def __init__ (self,model_type='random_forest'):
        self.model_type=model_type
        self.model=None
        self.label_encoder={} # maps letter to index
        self.label_decoder={} # maps index to letter
        if model_type=='random_forest':
            self.model=RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
    def load_training_data(self,gesture_data_dir='gesture_data'):
        """load all csv files from gesture_data directory"""
        x=[]
        y=[]
        csv_files=[f for f in os.listdir(gesture_data_dir) if f.endswith('.csv')]
        print(f"found{len(csv_files)} gesture files")
        for idx,csv_file in enumerate(csv_files):
            ketter=csv_file.split('_')[1].split('.')[0]# extract letter from filename 
            self.label_encoder[letter]=idx
            self.label_decoder[idx]=letter
            filepath=os.path.join(gesture_data_dir,csv_file)
            df=pd.read_csv(filepath)
            
            #extract features (every thing but label and confidence)
            features=df.iloc[:,:63].values
            labels=[idx]*len(features)
            x.extend(features)
            y.extend(labels)
            print(f"loaded{len(features)} samples for letter {letter}")
            return np.array(x),np.array(y)
    def train(self,gesture_data_dir='gesture_data',test_size=0.2):
        """train the model using data from gesture_data directory"""
        print("\n" + "="*50)
        print("Loading training data...")
        print("="*50)
        x,y=self.load_training_data(gesture_data_dir)
        print(f"\nTotal samples: {len(x)}")
        print(f"features per sample: {x.shape[1]}")
        print(f"Classes: {len(self.label_encoder)}")
        
        #split data
        x_train,x_test,y_train,y_test=train_test_split(
            x,y,test_size=test_size, random_state=42, stratify=y
            )
        print(f"\nTraining{len(x_train)} samples")
        print(f"Test set: {len(x_test)} samples")
        print(f"\n" + "="*50)
        print("Training model...")
        print("="*50)
        self.model.fit(x_train,y_train)
        #evaluate
        y_pred=self.model.predict(x_test)
        accuracy=accuracy_score(y_test,y_pred)
        print(f"\n model training complete. Test accuracy: {accuracy:.4f}")
        #detailed metrics per class
        print("\n per_letter metrics:")
        for idx,letter in self.label_decoder.items():
            mask=y_test==idx
            if mask.sum()>0:
                class_accuracy=accuracy_score(y_test[mask],y_pred[mask])
                print(f"letter {letter}:{class_accuracy:.4f}")