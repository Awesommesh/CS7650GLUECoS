# GLUECoS: An Evaluation Benchmark for Code-Switched NLP
This is the repo for the ACL 2020 paper [GLUECoS: An Evaluation Benchmark for Code-Switched NLP](https://www.aclweb.org/anthology/2020.acl-main.329/)

GLUECoS is a benchmark comprising of multiple code-mixed tasks across 2 langauge pairs (En-Es and En-Hi)

Below are instructions for obtaining the datasets that comprise the benchmark and training transformer based models on this data. Both steps can be run on separate systems and the instructions are structured in such a way. All the user has to do is to copy over the `Data/Processed_Data` folder over to perform training

## Obtaining Datasets
Follow the following instructions to download and process the datasets. All the steps have been tested and should work in a brand new conda environment with `python==3.6.10` or a docker container with the `python:3.6` image
1. Install the requirements for the preprocessing scripts
    ```
    pip install -r requirements.txt
    ```
2. Create a twitter developer account and fill in the 4 keys, one per line,  in `twitter_authentication.txt`. The file should look like this
    ```
    consumer key
    secret key
    access token
    access secret token
    ```
    
3. Obtain a key for microsoft translator. This is needed as the preprocessing steps involve conversion of romanized datasets into devanagari. Instructions for obtaining this key can be found [here](https://docs.microsoft.com/en-us/azure/cognitive-services/translator/translator-how-to-signup). The number of queries made fall within the free tier. This key will be referred to as SUBSCRIPTION_KEY in the next step
4. To finally download the data, run the command below. This will download the original datasets, perform all the preprocessing needed and bring them into a format that the training scripts can use
    ```
    ./download_data.sh SUBSCRIPTION_KEY
    ```
    The dowloaded and processed data is stored in `Data/Processed_Data`. 
    
    Some of the datasets did not have predefined splits, so the splits used for those can be found in `Data/Original_Data`.

    Please note that the labels for the test sets are not the gold labels. They have been assigned a separate token to maintain fairness in the benchmarking.

    This will not download/preprocess the QA dataset. For that, please check the next step

5. The original QA dataset (Chandu et. al, 2018) contains contexts only for some examples. To obtain contexts for the rest, [DrQA](https://github.com/facebookresearch/DrQA) is used to obtain contexts from a Wikipedia dump. To run this, you will need atleast 20GB of disk storage (to store the wikidump) and 16GB+ of RAM (to run DrQA). DrQA uses pytorch, so having a GPU will help speed it up (although it isn't necessary).

    First, install a suitable version of pytorch for your system. In most cases, a `pip install torch` should do

    To download and process the QA dataset, run the following command
    ```
    bash Data/Preprocess_Scripts/preprocess_qa.sh
    ```

## Training models on the data
The code contains 4 different evaluation scripts
1. One script for token level tasks:
    - LID (en_es/en_hi)
    - NER (en_es/en_hi),
    - POS (en_es/en_hi_fg/en_hi_ud)
2. One script for the sentence level tasks:
    - Sentiment (en_es/en_hi)
3. One script for the QA task 
    - QA (en_hi)
4. One script for the NLI task
    - NLI (en_hi)

You can train the models on your system or via Azure Machine Learning. To know more about the latter, please refer to [this README](azure_ml/README.md).

### Install the training requirements  
Note: The requirements for dataset preprocessing and training have been separately mentioned, as you may run them on different systems
1. Install a suitable version of pytorch for your system, `pip install torch` should work in most cases
2. The requirements from the file in `Code/requirements.txt`
    ```
    pip install -r Code/requirements.txt
    ```
### Training
Run the below command to fine-tune your model on any of the task. The scripts support any models based on bert, xlm and xlm-roberta

```
bash train.sh MODEL MODEL_TYPE TASK 
```
Example Usage :
```    
bash train.sh bert-base-multilingual-cased bert POS_EN_HI_FG
```
You can also run fine-tuning for all tasks with the following command :
```
bash train.sh bert-base-multilingual-cased bert ALL
```

## Citation
Please use the following citation if you would like to mention our work
```
@inproceedings{khanuja-etal-2020-gluecos,
    title = "{GLUEC}o{S}: An Evaluation Benchmark for Code-Switched {NLP}",
    author = "Khanuja, Simran  and
      Dandapat, Sandipan  and
      Srinivasan, Anirudh  and
      Sitaram, Sunayana  and
      Choudhury, Monojit",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.329",
    pages = "3575--3585"
}
```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
