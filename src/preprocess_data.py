import datasets
from tqdm.auto import tqdm
from pathlib import Path
class PreProcessTrainingData():
    def __init__(self):
        pass
    
    def load_dataset(dataset_name, subset:Optional):
        # Loading the dataset using huggingface
        dataset = datasets.load_dataset(dataset_name, subset, trust_remote_code=True)
        return dataset
    # proprocess the data and return the path
    def preprocess(dataset, split_size):
        text_data = []
        file_count = 0
        for sample in tqdm(dataset['train']):
            sample = sample['text'].replace('\n', '')
            text_data.append(sample)
            # split size 5000 meaning the 5000 text data per 1 file
            if len(text_data) == split_size:
                with open(f'data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                    fp.write('\n'.join(text_data))
                text_data = []
                file_count += 1

        # after saving in 5K chunks, we will have ~3808 leftover samples, we save those now too
        with open(f'data/text_{file_count}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_data))
         path = [for x in Path('data').glob('**/*.txt')]
         return path
        



