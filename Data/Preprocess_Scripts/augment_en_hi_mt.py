import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
import random
import argparse
from nlpaug.util.file.download import DownloadUtil

def read_train_file(train_file):
    sentences = []
    with open(train_file, "r") as f:
        for line in f:
            sentence_x = line.strip().split("\t")[0]
            sentence_y = line.strip().split("\t")[1]
            sentences.append((sentence_x, sentence_y))
    return sentences

def augment_sentences(sentences, augmentation_method):
    # download word2vec
    if augmentation_method == "word2vec"
        DownloadUtil.download_word2vec(dest_dir='.')

    # define augmenter
    aug_syn = naw.SynonymAug(aug_src='wordnet', model_path=".", name='Synonym_Aug', aug_min=1, aug_max=3, aug_p=0.3, lang='eng',
                     stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, force_reload=False,
                     verbose=0)
    aug_w2v = naw.WordEmbsAug(model_type='word2vec', model=model, action="substitute", aug_min=1, aug_max=3, aug_p=0.3, top_k=10)
    # augment sentences
    augmented_sentences = []
    for sentence_x, sentence_y in sentences:
        # generate 1-2 augmented sentences
        num_aug = random.randint(1, 2)
        for _ in range(num_aug):
            augmented_x = None
            if augmentation_method == 'synonym':
                augmented_x = aug_syn.augment(sentence_x)
            elif augmentation_method == "word_emb":
                augmented_x = aug_w2v.augment(sentence_x)
            else:
                print("Got unexpected augmentation method!")
                return None
            augmented_x = random.choice(augmented_x)
            if augmented_x == sentence_x:
                continue
            augmented_sentence = f"{augmented_x}\t{sentence_y}"
            augmented_sentences.append(augmented_sentence)

    return augmented_sentences

if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--augmentation-method', type=str, choices=['synonym', 'word2vec'], default='synonym', help='type of augmentation method to use')
    args = parser.parse_args()
    train_file = "Data/Processed_Data/MT_EN_HI/train.txt"

    # read input sentences from train.txt file
    sentences = read_train_file(train_file)

    # perform appropriate augmentation on input sentences
    all_augmented_sentences = augment_sentences(sentences, args.augmentation_method)
    print(len(all_augmented_sentences), "new sentences added")
    # write augmented sentences to train.txt file
    with open(train_file, "a") as f:
        f.write("\n")
        for sentence in all_augmented_sentences:
            f.write(sentence + "\n")





