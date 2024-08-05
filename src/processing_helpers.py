import re
from nltk.stem import PorterStemmer
from processing import clean_text, tokenize, extract_correct_words, remove_stop_words, extract_lemma

replacement_patterns = [
        (r" won\'t ", " will not "),
        (r" can\'t ", " cannot "),
        (r" ain\'t ", " is not "),
        (r" (\w+)\'ll ", " \g<1> will "),
        (r" (\w+)n\'t ", " \g<1> not "),
        (r" i\'m ", " i am "),
        (r" (\w+)\'ve ", " \g<1> have "),
        (r" (\w+)\'re ", " \g<1> are "),
        (r" (\w+)\'d ", " \g<1> would "),
    # people might not put ' and verbs finishing in nt should be separated
        (r" wont ", " will not "),
        (r" cant ", " cannot "),
        (r" aint ", " is not "),
        (r" dont ", " do not "),
        (r" didnt ", " did not "),
        (r" mightnt ", " might not "),
        (r" maynt ", " may not "),
        (r" couldnt ", " could not "),
        (r" wouldnt ", " would not "),
        (r" shouldnt ", " should not "),
        (r" wasnt ", " was not "),
        (r" werent ", " were not "),
    # typos with "be" and "have"
        (r" im ", " i am "),
        (r" ive ", " i have "),
        (r" youre ", " you are "),
        (r" youve ", " you have "),
        (r" it\'s ", " it is "),
        (r" he\'s ", " he is "),
        (r" hes ", " he is "),
        (r" she\'s ", " she is "),
        (r" shes ", " she is "),
        (r" were ", " we are "),
        (r" weve ", " we have "),
        (r" theyre ", " they are "),
        (r" theyve ", " they have ")]

class TyposReplacer:
    def __init__(self, patterns=replacement_patterns) -> None:
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text: str):
        s = " " + text + " "
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        
        return s.strip()

class TweetProcessor:
    def __init__(self, typos_replacer, retriever) -> None:
        self.typos_replacer = typos_replacer
        self.retriever = retriever

    def process_tweet(self, tweet: str) -> str:
        """
        Process a tweet by cleaning, tokenizing, removing stop words, and stemming the words.

        Args:
            tweet (str): The input tweet to be processed.

        Returns:
            list: A list of stemmed words after processing the tweet.
        """
        cleaned_tweet = clean_text(tweet)
        clean_tweet_replaced = self.typos_replacer.replace(cleaned_tweet)
        words = tokenize(clean_tweet_replaced)
        words_extracted = extract_correct_words(words)
        words_filtered = remove_stop_words(words_extracted)
        words_stemmed = extract_lemma(words_filtered, self.retriever)

        return ' '.join(words_stemmed)