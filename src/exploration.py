import re

def find_tweet_with_pattern(tweets, pattern: str) -> str:
    prog = re.compile(pattern)

    for tweet in tweets:
        if prog.search(tweet) is not None:
            return tweet