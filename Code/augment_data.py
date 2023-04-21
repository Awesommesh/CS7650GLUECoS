import random
import pandas as pd
import nltk
import json
import os
nltk.download('wordnet')
from nltk.corpus import wordnet


# Load the data from train
def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    examples = []
    with open(file_path, 'r') as infile:
        lines = infile.read().strip().split('\n\n')
        for example in lines:
            example = example.split('\n')
            words = [line.split('\t')[0] for line in example]
            labels = [line.split('\t')[-1] for line in example]
            examples.append({'words': words, 'labels': labels})
        if mode == 'test':
            for i in range(len(examples)):
                if examples[i]['words'][0] == 'not found':
                    examples[i]['present'] = False
                else:
                    examples[i]['present'] = True
    return examples

train_data = read_examples_from_file('Data/Processed_Data/NER_EN_HI', 'train')

# Define a function to perform data augmentation by randomly replacing some words with their synonyms
def augment_data(data, num_replacements=0.1):
    augmented_data = []
    word_mappings = []
    for i in range(len(data)):
        words = data[i]['words']
        labels = data[i]['labels']
        num_words = len(words)
        num_replaced_words = int(num_words * num_replacements)
        replaced_words_indices = random.sample(range(num_words), num_replaced_words)
        for index in replaced_words_indices:
            # Randomly select a word to replace
            word = words[index]
            # Get synonyms of the selected word using WordNet
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.append(lemma.name())
            # Replace the word with a randomly selected synonym
            if len(synonyms) > 0:
                synonym = random.choice(synonyms)
                words[index] = synonym
                word_mappings.append({'original_word': word, 'synonym': synonym})
        augmented_data.append({'words': words, 'labels': labels})
    return augmented_data, word_mappings


# Perform data augmentation on the train data
augmented_train_data, words_replaced = augment_data(train_data)

# Merge the original and augmented data
merged_data = train_data + augmented_train_data

# Save the merged data in the same format as the original data
augmented_data_path = 'Data/Processed_Data/NER_EN_HI/train_augmented.txt'
with open(augmented_data_path, 'w', encoding='utf-8') as f:
    for sentence in merged_data:
        words = sentence['words']
        labels = sentence['labels']
        for i in range(len(words)):
            f.write(words[i] + '\t' + labels[i] + '\n')
        f.write('\n')

# Save the words replaced and their synonyms
with open('words_replaced.json', 'w', encoding='utf-8') as f:
    json.dump(words_replaced, f, indent=2, ensure_ascii=False)