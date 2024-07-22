import re
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def read_and_split(path: str) -> tuple:
    """
    Read the data from a CSV file, split it into training and testing sets, and return the split datasets.

    Args:
        path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the training and testing datasets in the following order:
            - X_train: The training features.
            - y_train: The training labels.
            - X_test: The testing features.
            - y_test: The testing labels.
    """
    try:
        df = pd.read_csv(path, encoding='ISO-8859-1', usecols=[0, 2, 5], header=0, names=['target', 'date', 'text'])
    except Exception as e:
        raise ValueError(f"Error reading the CSV file: {e}")

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=21)

    y_train, y_test = df_train['target'], df_test['target']
    X_train, X_test = df_train.drop(columns='target'), df_test.drop(columns='target')

    return X_train, y_train, X_test, y_test

def replace_emojis(text: str) -> str:
    """
    Replace emojis with keywords EMO_POS and EMO_NEG for positive and negative emojis, respectively.

    Args:
        text (str): The text to be processed.

    Returns:
        str: The processed text.
    """
    # Smile -- :), : ), :-), (:, ( :, (-:
    text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', text)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', text)
    # Love -- <3, :*
    text = re.sub(r'(<3|:\*)', ' EMO_POS ', text)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', text)
    # Sad -- :-(, : (, :(, ):, )-:
    text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', text)
    # Cry -- :,(, :'(, :"(
    text = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', text)
    
    return text

def clean_text(text: str) -> str:
    """
    Process the given text by transforming it to lowercase, replacing emojis, removing mentions and URLs, and replacing multiple spaces with a single space.

    Args:
        text (str): The text to be processed.

    Returns:
        str: The processed text.
    """
    # transform to lowercase
    text = text.lower()
    # replace emojis
    text = replace_emojis(text)
    # remove mentions
    text = re.sub(r'@[\S]+', '', text)
    # remove hashtags
    text = re.sub(r'#[\S]+', '', text)
    # remove urls
    text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', text)
    # replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    return text

def tokenize(text: str) -> list:
    """
    Tokenizes the given text into a list of words.
    
    Args:
        text (str): The input text to be tokenized.
        
    Returns:
        list: A list of words extracted from the input text.
    """
    words = word_tokenize(text)
    
    return words

import re

def is_word_correct(word: str) -> bool:
    """
    Check if a word consists of only alphabetic characters.

    Args:
        word (str): The word to be checked.

    Returns:
        bool: True if the word consists of only alphabetic characters, False otherwise.
    """
    pattern = r"^'?[A-Za-z']+$"
    return re.search(pattern, word) is not None

def remove_duplicate_letters(word: str) -> str:
    """
    Removes duplicate letters from a word.

    Args:
        word (str): The input word.

    Returns:
        str: The word with duplicate letters removed.
    """
    word = re.sub(r'(.)\1+', r'\1\1', word)

    return word

def extract_correct_words(words: list) -> list:
    """
    Extracts the correct words from a list of words, replaces specific contractions
    and removes duplicate letters (more than 2).
    
    Args:
        words (list): A list of words.
        
    Returns:
        list: A list of correct words.
    """
    correct_words = [word for word in words if is_word_correct(word)]

    mapping = {"'s": "is", "ca": "can", "n't": "not", "'ll": "will", "'m": "am", "u": "you"}

    correct_words = [mapping.get(word, word) for word in correct_words]
    correct_words = [remove_duplicate_letters(word) for word in correct_words]

    return correct_words

import re

def remove_stop_words(words: list) -> list:
    """
    Removes stop words from a list of words.

    Args:
        words (list): A list of words.

    Returns:
        list: A list of words with stop words removed.
    """
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if not word in stop_words]

    return filtered_words

def stem_words(words: list) -> list:
    """
    Stem the given list of words using Porter Stemmer algorithm.
    
    Args:
        words (list): A list of words to be stemmed.
        
    Returns:
        list: A list of stemmed words.
    """
    porter_stemmer = PorterStemmer()
    stemmed_words = [porter_stemmer.stem(word) for word in words]
 
    return stemmed_words

def process_tweet(tweet: str) -> list:
    """
    Process a tweet by cleaning, tokenizing, removing stop words, and stemming the words.

    Args:
        tweet (str): The input tweet to be processed.

    Returns:
        list: A list of stemmed words after processing the tweet.
    """
    cleaned_tweet = clean_text(tweet)
    words = tokenize(cleaned_tweet)
    words_extracted = extract_correct_words(words)
    words_filtered = remove_stop_words(words_extracted)
    words_stemmed = stem_words(words_filtered)

    return words_stemmed

def create_dictionary(tweets: pd.Series) -> dict:
    """
    Creates a dictionary of words and their frequencies from a series of tweets.

    Args:
        tweets (pd.Series): A series of tweets.

    Returns:
        dict: A dictionary where the keys are words and the values are their frequencies.
    """
    words_dict = dict()

    for tweet in tweets:
        words = process_tweet(tweet)
        for word in words:
            if word not in words_dict:
                words_dict[word] = 1
            else:
                words_dict[word] += 1
    
    return words_dict 
