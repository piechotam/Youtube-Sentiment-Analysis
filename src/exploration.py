import re
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

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
    
    df['target'] = df['target'].replace(4, 1)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=21)
    df_train, df_valid = train_test_split(df_train, test_size=0.2, random_state=21)

    y_train, y_valid, y_test = df_train['target'], df_valid['target'], df_test['target']
    X_train, X_valid, X_test = df_train['text'], df_valid['text'], df_test['text']

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def find_tweet_with_pattern(tweets, pattern: str) -> str:
    """
    Find and return the first tweet that matches the given pattern.

    Args:
    tweets (list of str): A list of tweet strings to search through.
    pattern (str): A regex pattern to search for within the tweets.

    Returns:
    str: The first tweet that matches the pattern. If no tweet matches, returns None.
    """
    prog = re.compile(pattern)

    for tweet in tweets:
        if prog.search(tweet) is not None:
            return tweet
        
def count_words(tweet: str) -> int:
    """
    Counts the number of words in a given tweet.

    Parameters:
    tweet (str): The tweet to count the words from.

    Returns:
    int: The number of words in the tweet.
    """
    return len(tweet.split(' '))

def length_distribution(tweets: pd.Series, labels: pd.Series, size: int = 10e5):
    """
    Plot the word count distribution in tweets.

    Args:
        tweets (pd.Series): A pandas Series containing the tweets.
        labels (pd.Series): A pandas Series containing the corresponding labels for the tweets.
        size (int, optional): The maximum sample size for each label. Defaults to 100 000.
    """
    temp_df = pd.concat([tweets, labels], axis=1, keys=['tweet', 'label'])
    positive_sample_size = min(size, temp_df[temp_df['label'] == 1].shape[0])
    negative_sample_size = min(size, temp_df[temp_df['label'] == 0].shape[0])

    positive_sample = temp_df[temp_df['label'] == 1].sample(positive_sample_size)
    negative_sample = temp_df[temp_df['label'] == 0].sample(negative_sample_size)

    word_count_pos = positive_sample['tweet'].apply(count_words)
    word_count_neg = negative_sample['tweet'].apply(count_words)
    
    fig, axs = plt.subplots(1, 2, figsize=(9, 5))

    axs[0].hist(word_count_neg, color='red', alpha=0.5)
    axs[0].set_title('Negative Tweets')
    axs[0].set_xlabel('Word Count')
    axs[0].set_ylabel('Count')

    axs[1].hist(word_count_pos, color='blue', alpha=0.5)
    axs[1].set_title('Positive Tweets')
    axs[1].set_xlabel('Word Count')
    axs[1].set_ylabel('Count')

    fig.suptitle('Word Count Distribution in Tweets')
    
    plt.show()