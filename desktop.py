#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('C:\\Users\\ishan\\Downloads\\audio-based-violence-detection-dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# In[2]:


get_ipython().system('pip install librosa')


# In[3]:


import numpy as np
import librosa.display, os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)
    
def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)


# In[4]:


print(os.getcwd())


# In[5]:


print(os.listdir("../"))


# In[6]:


metal_excel_dir = "C:\\Users\\ishan\\Downloads\\audio-based-violence-detection-dataset\\VSD.xlsx"


# In[7]:


print(os.listdir("C:\\Users\\ishan\\Downloads\\audio-based-violence-detection-dataset"))


# In[8]:


metadata = pd.read_excel(metal_excel_dir,sheet_name='read_dataset')


# In[9]:


metadata = metadata.drop(metadata[metadata['File_segment_name'] == "NaN"].index)


# In[10]:


metadata.shape


# In[11]:


print(metadata.shape)


# metadata.dropna()

# In[12]:


print(metadata.head(10))


# In[13]:


violence_dir ="C:\\Users\\ishan\\Downloads\\Abuse-Analysis-on-Audio-data-main/violence_data"
non_violence_dir = "C:\\Users\\ishan\\Downloads\\Abuse-Analysis-on-Audio-data-main/non_violence_data"


# In[14]:


os.mkdir(violence_dir)
os.mkdir(non_violence_dir)


# In[15]:


get_ipython().system('pip install pydub')


# In[16]:


from pydub import AudioSegment
import os


violence_dir = "C:\\Users\\ishan\\Downloads\\Abuse-Analysis-on-Audio-data-main/violence_data"
audio_base_dir = "C:\\Users\\ishan\\Downloads\\audio-based-violence-detection-dataset\\audios_VSD"


if not os.path.exists(violence_dir):
    os.makedirs(violence_dir)


for index, record in metadata.iterrows():
 
    audio_file_path = os.path.join(audio_base_dir, f"{record['File_segment_name']}.wav")
    
    if os.path.exists(audio_file_path):
        # Convert start and end times to milliseconds
        t1 = int(1000 * record['Violence_start'])  # Start time in milliseconds
        t2 = int(1000 * record['Violence_end'])    # End time in milliseconds
        try:
            audio = AudioSegment.from_wav(audio_file_path)
            violence_segment = audio[t1:t2]
            output_file_path = os.path.join(violence_dir, f"{record['File_segment_name']}.wav")
            violence_segment.export(output_file_path, format="wav")
            print(f"Processed and exported: {output_file_path}")
        except Exception as e:
            print(f"Failed to process {audio_file_path}: {str(e)}")
    else:
        print(f"File does not exist: {audio_file_path}")


# In[18]:


from pydub import AudioSegment
import os

violence_dir = "C:\\Users\\ishan\\Downloads\\Abuse-Analysis-on-Audio-data-main/violence_data"  # Adjust the path as needed
if not os.path.exists(violence_dir):
    os.makedirs(violence_dir)  # Create the directory if it does not exist

audio_base_dir = "C:\\Users\\ishan\\Downloads\\audio-based-violence-detection-dataset\\audios_VSD"

for index, record in metadata.iterrows():  # Ensure 'metadata' is a DataFrame
    audio_file_path = os.path.join(audio_base_dir, f"{record['File_segment_name']}.wav")
    
    if os.path.exists(audio_file_path):
        t1 = int(1000 * record['Violence_start'])  # Start time in milliseconds
        t2 = int(1000 * record['Violence_end'])    # End time in milliseconds
        try:
            audio = AudioSegment.from_wav(audio_file_path)
            violence_segment = audio[t1:t2]
            output_file_path = os.path.join(violence_dir, f"{record['File_segment_name']}.wav")
            violence_segment.export(output_file_path, format="wav")
            print(f"Processed and exported: {output_file_path}")
        except Exception as e:
            print(f"Failed to process {audio_file_path}: {str(e)}")
    else:
        print(f"File does not exist: {audio_file_path}")


# In[20]:


import librosa

def features_extractor(file):
    #load the file (audio)
    #audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    audio, sample_rate = librosa.load(file) 
    
    
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
  
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features


# In[21]:


print("Check")


# In[22]:


import numpy as np
from tqdm import tqdm

extracted_features=[]
"""for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])"""


for record in metadata.values:
    file_name = os.path.join(os.path.abspath(violence_dir),record[0]+".wav")
    print(file_name)
    final_class_labels=1
    try:
        data=features_extractor(file_name)
    except:
        continue
    extracted_features.append([data,final_class_labels])
   
        
for record in metadata.values:
    file_name = os.path.join(os.path.abspath(non_violence_dir),record[0]+"_non_violence.wav")
    print(file_name)
    final_class_labels=0
    try:
        data=features_extractor(file_name)
    except:
        continue
    extracted_features.append([data,final_class_labels])


# In[ ]:





# In[23]:


extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()


# In[24]:


extracted_features_df["class"].value_counts()


# In[25]:


X=np.array(extracted_features_df["feature"].tolist())
y=np.array(extracted_features_df["class"].tolist())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)


# In[26]:


X_test.shape


# In[28]:


X_train = X_train.reshape(X_train.size // 40, 40)


print(X_train.shape)


# In[29]:


y.shape


# In[58]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics


model = Sequential()

# LSTM layer
model.add(LSTM(128, input_shape=(1, 40), return_sequences=True))
model.add(Dropout(0.5))  

# Additional LSTM layer
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))  

# Flatten layer
model.add(Flatten())

# Dense layers with dropout
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = Adam(lr=0.0005)  # Adjusted learning rate
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Adjusted patience
model_checkpoint = ModelCheckpoint(filepath='./audio_classification.hdf5', verbose=1, save_best_only=True)

# Train the model
num_epochs = 100  # Reduced number of epochs
num_batch_size = 16  # Reduced batch size
history = model.fit(X_train_reshaped, y_train, batch_size=num_batch_size, epochs=num_epochs, 
                    validation_data=(X_test_reshaped, y_test), 
                    callbacks=[early_stopping, model_checkpoint], verbose=1)




# In[55]:


"""import tensorflow as tf

# Create the RNN model
model = tf.keras.Sequential([
  tf.keras.layers.LSTM(128, input_shape=(X_train.shape[1:])),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
"""
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 


model = tf.keras.Sequential([
    LSTM(128, input_shape=(1, X_train.shape[1]), return_sequences=True),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='./audio_classification.hdf5', verbose=1, save_best_only=True)


num_epochs = 10
num_batch_size = 32
start = datetime.now()
model.fit(X_train_reshaped, y_train, batch_size=num_batch_size, epochs=num_epochs, 
          validation_data=(X_test_reshaped, y_test), callbacks=[checkpointer], verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)



# In[60]:


import numpy as np

X_test_reshaped = np.expand_dims(X_test, axis=1)  

test_accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
print(test_accuracy)


# In[64]:


import numpy as np


X_test_reshaped = np.expand_dims(X_test, axis=1)

predict_x = model.predict(X_test_reshaped) 


classes_x = (predict_x > 0.5).astype("int32")


print(classes_x)


# In[65]:





# In[66]:


X_test.shape


# In[67]:


y_test.shape


# In[68]:


predict_x.shape


# In[69]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, classes_x)

