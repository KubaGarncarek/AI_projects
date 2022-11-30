import nltk
import math
import sys
import os
import string
import operator

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    mapping_dict = {}
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), errors="ignore") as f:
            content = f.read()
            mapping_dict[filename] = content
            f.close()
    return mapping_dict

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    without_stopwords = []

    stopwords = set(nltk.corpus.stopwords.words("english"))
    list_of_words = [word.lower() for word in nltk.word_tokenize(document)]
    
    for word in list_of_words:
        if word not in string.punctuation and word not in stopwords:
            without_stopwords.append(word)
    return without_stopwords

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set()
    for filename in documents:
        words.update(documents[filename])

    idfs = dict()
    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf
        
    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfs = dict()
    for filename, text in files.items():
        for word in query:
            tf = sum(value == word for value in text)
            try:
                tfidfs[filename] += tf*idfs[word]
            except KeyError:
                tfidfs[filename] = tf*idfs[word]
            
    sorted_d = dict( sorted(tfidfs.items(), key=operator.itemgetter(1),reverse=True))
    sorted_list = list(sorted_d.keys())
   
    return sorted_list[0:n]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_rank = {}
    for sentence in sentences:
        sentence_rank[sentence] = {}
        sentence_length = len(sentences[sentence])
        sentence_rank[sentence]["idfs"] = 0
        sentence_rank[sentence]['word_count'] = 0
        for word in query:
            if word in sentences[sentence]:
                sentence_rank[sentence]['idfs'] += idfs[word]
                sentence_rank[sentence]["word_count"] += 1
        sentence_rank[sentence]["qtd"] = float(sentence_rank[sentence]["word_count"]/sentence_length)
        
    sorted_list = sorted(sentence_rank, key=lambda sentence: (sentence_rank[sentence]["idfs"], sentence_rank[sentence]["qtd"]), reverse=True)
        
    return sorted_list[0:n]

    
if __name__ == "__main__":
    main()
