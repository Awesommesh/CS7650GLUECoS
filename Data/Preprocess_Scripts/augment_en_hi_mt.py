import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
import random

def read_train_file(train_file):
    sentences = []
    with open(train_file, "r") as f:
        for line in f:
            sentence_x = line.strip().split("\t")[0]
            sentence_y = line.strip().split("\t")[1]
            sentences.append((sentence_x, sentence_y))
    return sentences

def augment_sentences(sentences):
    # define augmenter
    augmenter = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=1, aug_max=3, aug_p=0.1, lang='eng',
                     stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, force_reload=False,
                     verbose=0)

    # augment sentences
    augmented_sentences = []
    for sentence_x, sentence_y in sentences:
        # generate 1-2 augmented sentences
        num_aug = random.randint(1, 2)
        for _ in range(num_aug):
            augmented_x = augmenter.augment(sentence_x)
            augmented_x = random.choice(augmented_x)
            augmented_sentence = f"{augmented_x}\t{sentence_y}"
            augmented_sentences.append(augmented_sentence)

    return augmented_sentences

train_file = "../Processed_Data/MT_EN_HI/train.txt"

# read input sentences from train.txt file
sentences = read_train_file(train_file)

# perform synonym augmentation on input sentences
augmented_sentences = augment_sentences(sentences)
print(len(augmented_sentences), "new sentences added")
# write augmented sentences to train.txt file
with open(train_file, "a") as f:
    for sentence in augmented_sentences:
        f.write(sentence + "\n")
