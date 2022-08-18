import os
import os.path
import sys
import numpy as np
import pandas as pd
import librosa
import librosa.display
import configparser
import json
from tqdm import tqdm
from PIL import Image


def get_audio_slices(audio_data, sampling_rate, audio_len):
    '''''
    Split audio into fixed length slices
    '''''

    slices = []
    slice_start = 0
    slice_len = sampling_rate * audio_len
    audio_len = len(audio_data)

    if audio_len < slice_len:
        audio_data = librosa.util.fix_length(audio_data, size=slice_len)
    elif audio_len > slice_len:
        slice_end = int(slice_start + slice_len)
        
        while slice_end < audio_len:
            slice = audio_data[slice_start:slice_end]
            slices.append(slice)
            slice_start = slice_end
            slice_end = int(slice_start + slice_len)
            
        return slices
        
    return [audio_data]

def get_melspect_path(audio_path, melspect_save_path, slice_idx):
    melspect_filename = f'{os.path.split(audio_path)[1].split(".")[0]}_{slice_idx}.png'
    return f'{melspect_save_path}/{melspect_filename}'

def save_melspectrogram(melspectrogram, audio_path, slice_idx):
    img = Image.fromarray(melspectrogram)
    img = img.convert('L')
    img.save(get_melspect_path(audio_path, melspect_save_path, slice_idx))

def gen_melspectrogram(audio_data, sampling_rate, hop_length, n_fft, n_mels):
    # Generate melspectrogram
    melspectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length) 
    melspectrogram = librosa.power_to_db(melspectrogram)
    melspectrogram = np.flip(melspectrogram, axis=0)

    return melspectrogram

def generate_melspectrograms(datasets, melspect_save_path, csv_save_path, sampling_rate, n_fft, hop_length, n_mels, audio_len):
    '''''
    Generate melspectrograms from audio and save as .png
    '''''

    for in_df_path in datasets:
        out_df = []

        try:
            df = pd.read_csv(in_df_path)
        except Exception as e:
            print(e)
            continue
        
        if 'Unnamed: 0' in df.columns:
            df = df.drop(['Unnamed: 0'], axis='columns')

        print(f'Processing {in_df_path}')

        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            audio_path = row['audio_path']

            # Load audio file
            try:
                audio_data, sr = librosa.load(audio_path, sr=sampling_rate) 
            except Exception as e:
                print(e)
                continue
            
            # Split audio file into even slices
            slices = get_audio_slices(audio_data=audio_data, sampling_rate=sampling_rate, audio_len=audio_len)

            for slice_idx, slice in enumerate(slices):
                # Generate melspectrogram
                melspectrogram = gen_melspectrogram(audio_data=slice, sampling_rate=sampling_rate, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)

                if not os.path.exists(melspect_save_path):
                    os.makedirs(melspect_save_path)
                
                if not os.path.exists(csv_save_path):
                    os.makedirs(csv_save_path)

                # Save melspectrogram
                save_melspectrogram(melspectrogram=melspectrogram, audio_path=audio_path, slice_idx=slice_idx)

                melspect_path = get_melspect_path(audio_path, melspect_save_path, slice_idx)
                out_df.append([melspect_path, str(audio_len), row['language'], row['language_family'], row['gender'], row['emotion']])
        
        out_df = pd.DataFrame(out_df, columns=df.columns)
        out_df = out_df.rename({'audio_path' : 'melspect_path'}, axis='columns')

        csv_out_path = f'{csv_save_path}/{os.path.split(in_df_path)[1]}'
        print(f'Saving {csv_out_path}')

        #df['audio_path'] = df['audio_path'].map(lambda x:get_melspect_path(x, melspect_save_path))
        #df = df.rename({'audio_path' : 'melspect_path'}, axis='columns')
        
        out_df.to_csv(csv_out_path, index=False)
        print('Saved\n')


if __name__=='__main__':
    if len(sys.argv) != 2:
        print('Usage: python generate_melspectrograms.py path/to/config')
        exit()
    
    print('Reading config file...\n')
    
    config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print('Config file not found')
        exit()

    config = configparser.ConfigParser()
    config.read(config_path)

    datasets = json.loads(config['generate_melspectrograms']['datasets'])
    melspect_save_path = config['generate_melspectrograms']['melspect_save_path']
    csv_save_path = config['generate_melspectrograms']['csv_save_path']
    sampling_rate = int(config['generate_melspectrograms']['sampling_rate'])
    n_fft = int(config['generate_melspectrograms']['n_fft'])
    hop_length = int(config['generate_melspectrograms']['hop_length'])
    n_mels = int(config['generate_melspectrograms']['n_mels'])
    audio_len = int(config['generate_melspectrograms']['audio_len'])

    # TODO Parametry samego spektrogramu dodać w configu

    generate_melspectrograms(datasets=datasets, melspect_save_path=melspect_save_path, 
                             csv_save_path=csv_save_path, sampling_rate=sampling_rate, 
                             n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, audio_len=audio_len)
