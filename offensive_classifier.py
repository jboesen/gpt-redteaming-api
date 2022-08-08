from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import boto3
import os
from pathlib import Path

# source: https://huggingface.co/cardiffnlp/twitter-roberta-base-offensive
s3=boto3.client('s3', 'us-west-1',
                    aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                    aws_secret_access_key=os.environ['AWS_SECRET_KEY'])

def download_s3_dir(bucket_name, remote_dir, local_dir):
    '''download all files in directory from aws s3.'''
    assert remote_dir.endswith('/')
    assert local_dir.endswith('/')
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    objs = bucket.objects.filter(Prefix=remote_dir)
    sorted_objs = sorted(objs, key=attrgetter("key"))
    for obj in sorted_objs:
        path = Path(os.path.dirname(local_dir + obj.key))
        path.mkdir(parents=True, exist_ok=True)
        if not obj.key.endswith("/"):
            bucket.download_file(obj.key, str(path) + "/" + os.path.split(obj.key)[1])

download_s3_dir(
    'gpt-redteaming-models', 'twitter-roberta-base-offensive/', 'twitter-roberta-base-offensive/'
    )

MODEL_PATH = 'cardiffnlp/twitter-roberta-base-offensive'
model = AutoModelForSequenceClassification\
        .from_pretrained(MODEL_PATH,
                        local_files_only=True,
                        cache_dir='./twitter-roberta-base-offensive',)
tokenizer = AutoTokenizer\
        .from_pretrained(MODEL_PATH,
                        local_files_only=True,
                        cache_dir='./twitter-roberta-base-offensive',)

# download label mapping
labels=['not-offensive', 'offensive']

def rate_offensiveness(sentence):
    """sentence through offensiveness model and returns probability from 0-1"""
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)
    results = softmax(output[0][0].detach().numpy())
    sentence_labels_dict = {}
    ranking = np.argsort(results)
    ranking = ranking[::-1]
    for i in range(results.shape[0]):
        label = labels[ranking[i]]
        sentence = results[ranking[i]]
        sentence_labels_dict[label] = np.round(float(sentence), 4)
    return sentence_labels_dict['offensive']

def sort_offensive(predictions_arr):
    """Rates and sorts offensiveness where key sentence and value is offensiveness"""
    offensive_vals = [rate_offensiveness(sentence) for sentence in predictions_arr]
    result_arr = []
    for i, _ in enumerate(predictions_arr):
        result_arr.append({'offensive': offensive_vals[i], 'sentence': predictions_arr[i]})
    return result_arr
