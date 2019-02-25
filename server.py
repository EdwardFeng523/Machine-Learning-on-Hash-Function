from flask import Flask, jsonify
import pickle
from HashTable import *
import time
import os.path

FILENAME = "persistence.pkl"

app = Flask(__name__)

hashTable = BloomFilter(10000, 10000)

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

@app.route('/', methods=['GET'])
def get_page():
    return "welcome to the comp480 term project server made by Edward Feng and Xinhao Liu"

@app.route('/request/<packet>', methods=['GET'])
def get_sentence(packet):
    print ("got data packet", packet)
    flag = hashTable.test(packet)
    print ("flag is", flag)
    if flag:
        print ("got a cache hit")
        time.sleep(5)
        return jsonify({'status': 'hit', 'packet': packet})
    else:
        print ("got a cache miss")
        return jsonify({'status': 'miss', 'packet': packet})


@app.route('/save_hash', methods=['POST'])
def save_hash():
    save_object(hashTable, FILENAME)
    print ("hash table just got saved")



if __name__ == '__main__':
    hashTable = BloomFilter(10000, 10000)
    if not os.path.isfile(FILENAME):
        save_object(hashTable, FILENAME)

    with open(FILENAME, 'rb') as input:
        hashTable = pickle.load(input)
    app.run(debug=True)
