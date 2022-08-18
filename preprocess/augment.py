import os
import os.path
import sys
import pandas as pd
import numpy as np
import librosa
import configparser
import json
import soundfile as sf
from tqdm import tqdm
from audiomentations import TimeStretch, AddGaussianNoise, Compose
#from imblearn.over_sampling import RandomOverSampler


def get_augmented_audio_path(audio_path, augmented_audio_save_dir):
    augmented_audio_filename = f'{os.path.split(audio_path)[1].split(".")[0]}_augmented.wav'
    return f'{augmented_audio_save_dir}/{augmented_audio_filename}'

def augment_audio(datasets, augmented_audio_save_dir, csv_save_dir, sampling_rate):
    '''''
    Apply random gaussian noise and time stretching to oversampled audio samples and save augmented audio file
    '''''

    for in_df_path in datasets:
        try:
            df = pd.read_csv(in_df_path)
        except Exception as e:
            print(e)
            continue
        
        save_paths = []
        print(f'Processing {in_df_path}')

        # Zastanowić się czy time stretching tu zadziała dobrze
        # Transformations to apply
        augment = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.014, p=0.5),
                           TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5)])

        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            audio_path = row['audio_path']

            try:
                audio_data, sr = librosa.load(audio_path, sr=sampling_rate) # Load audio file
            except Exception as e:
                print(e)
                continue

            augmented_audio = augment(samples=audio_data, sample_rate=sr) # Augmentation

            if not os.path.exists(augmented_audio_save_dir):
                os.makedirs(augmented_audio_save_dir)

            augmented_audio_save_path = get_augmented_audio_path(audio_path=audio_path, augmented_audio_save_dir=augmented_audio_save_dir)
            save_paths.append(augmented_audio_save_path)

            sf.write(file=augmented_audio_save_path, samplerate=16000, data=augmented_audio)
        
        if not os.path.exists(csv_save_dir):
                os.makedirs(csv_save_dir)
        
        df['audio_path'] = save_paths
        csv_save_path = f'{csv_save_dir}/{os.path.split(in_df_path)[1][:-4]}_augmented.csv'
        df = df.drop(['Unnamed: 0'], axis='columns')
        print(f'Saving {csv_save_path}')
        df.to_csv(csv_save_path, index=False)
        print('Saved\n')


if __name__=='__main__':
    if len(sys.argv) != 2:
        print('Usage: python augment.py path/to/config')
        exit()
    
    print('Reading config file...\n')

    config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print('Config file not found')
        exit()

    config = configparser.ConfigParser()
    config.read(config_path)

    datasets = json.loads(config['augment']['datasets'])
    audio_save_dir = config['augment']['audio_save_dir']
    csv_save_dir = config['augment']['csv_save_dir']
    sampling_rate = int(config['augment']['sampling_rate'])

    augment_audio(datasets=datasets, augmented_audio_save_dir=audio_save_dir, csv_save_dir=csv_save_dir, sampling_rate=sampling_rate)
