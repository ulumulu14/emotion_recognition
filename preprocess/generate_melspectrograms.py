import os
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import configparser
import json


path = 'D:/licencjat/data/dataframes/berlin_emodb_df.csv'

def generate_melspectrograms(in_df_path, out_path, sampling_rate=16000):
    '''''
    Generate melspectrograms from audio and save as .png
    '''''
    for file in os.listdir(in_df_path):
        input_df = pd.read_csv(in_df_path)

        for idx, row in input_df.iterrows():
            audio_path = row['audio_path']
            audio_data, sr = librosa.load(audio_path, sr=sampling_rate) # Load audio file

            melspectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate) # Generate melspectrogram

            if not os.path.exists(out_path):
                os.makedirs(out_path)

            # Save melspectrogram
            img = librosa.display.specshow(melspectrogram, sr=sr)
            plt.savefig(f'{out_path}/{file_id}.png')
            print(f'Melspectrogram for {out_path}/{file_id} generated succesfully')


if __name__=='__main__':
    config_path = '/mnt/archive03/a.kaus/gender_classification/scripts/extract_features.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    datasets = json.loads(config['generate_melspectrograms']['datasets'])
    in_df_path = config['generate_melspectrograms']['in_df_path']
    audio_path = config['extract_data']['audio_path']
    melspect_out_path = config['extract_data']['melspect_out_path']
    features_out_path = config['extract_data']['features_out_path']
    sampling_rate = int(config['extract_data']['sampling_rate'])
    n_mfcc = int(config['extract_data']['n_mfcc'])