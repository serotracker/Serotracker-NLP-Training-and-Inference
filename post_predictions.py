import requests
import time
import hashlib
import json
import os
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv('.env')
    PREDICTION_FILE_PATH = './output/all_predictions.txt'
    SERVER_IP = os.getenv('SERVER_IP')

    predictionDict = {}
    file = open(PREDICTION_FILE_PATH, 'r')
    lines = file.readlines()[1:]
    file.close()

    request = []
    key_set = set()

    c = 0

    #split list of abstracts into groups of 500 and post them to the server
    for i, line in enumerate(lines):
        blocks = line.split('\t')
        key = blocks[0]

        if(len(blocks) <= 1):
            continue

        if(key in key_set):
            continue

        key_set.add(key)
        inclusionLikelihood = float(blocks[1])
        pio = json.loads(blocks[2])
        request.append({
            'title_hash': key,
            'inclusion_likelihood': inclusionLikelihood,
            'pio': pio
        })
        if(c%500 == 499):
            requests.post(SERVER_IP + '/addPrediction', json = {
                'predictions': request 
            })
            request = []
        
        c += 1
            
    requests.post(SERVER_IP + '/addPrediction', json = {
        'predictions': request 
    })