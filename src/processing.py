import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

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
        df = pd.read_csv(path, encoding='ISO-8859-1', usecols=[0, 5], header=0, names=['target', 'text'])
    except Exception as e:
        raise ValueError(f"Error reading the CSV file: {e}")

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=21)
    df_train, df_valid = train_test_split(df_train, test_size=0.2, random_state=21)

    y_train, y_valid, y_test = df_train['target'], df_valid['target'], df_test['target']
    X_train, X_valid, X_test = df_train['text'], df_valid['text'], df_test['text']

    return X_train, y_train, X_valid, y_valid, X_test, y_test

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
    Process the given text by transforming it to lowercase, replacing emojis, mentions and URLs, 
    replacing multiple spaces with a single space and fixing common typos.

    Args:
        text (str): The text to be processed.

    Returns:
        str: The processed text.
    """
    text = replace_emojis(text)
    text = text.lower()
    text = re.sub(r'@[\S]+', 'USR_MEN', text)
    text = re.sub(r'#[\S]+', 'HASH', text)
    text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', 'URL', text)
    # Remove &...; which is used in our dataset for example for quotations: &quot;
    text = re.sub(r'&.*?;', '', text)
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

def is_word_correct(word: str) -> bool:
    """
    Check if a word consists of only expected characters.

    Args:
        word (str): The word to be checked.

    Returns:
        bool: True if the word consists of only alphabetic characters, False otherwise.
    """
    pattern = r"^(?=.*[A-Za-z])'?[A-Za-z'-_]+$"
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
    correct_words = [remove_duplicate_letters(word) for word in correct_words]

    return correct_words

def remove_stop_words(words: list) -> list:
    """
    Removes stop words from a list of words.

    Args:
        words (list): A list of words.

    Returns:
        list: A list of words with stop words removed.
    """
    # we remove not from set of stopwords - it might carry some valuable information in our usecase
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    filtered_words = [word for word in words if not word in stop_words]

    return filtered_words

def extract_lemma(words: list, retriever, pos_tags=None) -> list:
    """
    Extracts the lemma or stem of words using the provided retriever.
    
    Args:
        words (list): A list of words to be processed.
        retriever: An instance implementing stem or lemmatize function.
        pos_tags (list, optional): A list of part-of-speech tags corresponding to the words for lemmatization.
        
    Returns:
        list: A list of processed words (either stemmed or lemmatized).
    
    Raises:
        TypeError: If the retriever is not an instance of PorterStemmer or WordNetLemmatizer.
    """
    if hasattr(retriever, 'stem'):
        return [retriever.stem(word) for word in words]
    elif hasattr(retriever, 'lemmatize'):
        if pos_tags:
            return [retriever.lemmatize(word, pos) for word, pos in zip(words, pos_tags)]
        else:
            return [retriever.lemmatize(word) for word in words]
    else:
        raise TypeError('Retriever must have a stem or lemmatize method!')
    
def stem_words(words: list, porter_stemmer: PorterStemmer) -> list:
    """
    Stem the given list of words using Porter Stemmer algorithm.
    
    Args:
        words (list): A list of words to be stemmed.
        
    Returns:
        list: A list of stemmed words.
    """
    stemmed_words = [porter_stemmer.stem(word) for word in words]
 
    return stemmed_words

def lemmatize_words(words: list, lemmatizer: WordNetLemmatizer) -> list:
    """
    Lemmatize the given list of words using WordNetLemmatizer.
    
    Args:
        words (list): A list of words to be stemmed.
        
    Returns:
        list: A list of stemmed words.
    """
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    return lemmatized_words

def process_tweet(tweet: str, retriever) -> list:
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
    words_stemmed = extract_lemma(words_filtered, retriever)

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