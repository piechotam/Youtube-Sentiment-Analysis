import requests
import time
import pandas as pd

def extract_video_comments(API_KEY: str, VIDEO_ID: str, order: str, n=100) -> pd.Series:
    """
    Returns a DataFrame of n (default 100) comments from a video 
    with the provided VIDEO_ID and API_KEY.
    
    The order parameter specifies the order of comments. Valid values are: 'time', 'relevance'.
    
    Parameters:
        API_KEY (str): YouTube Data API v3 key.
        VIDEO_ID (str): The ID of the YouTube video.
        order (str): Order of comments, either 'time' or 'relevance'.
        n (int, optional): Number of comments to retrieve. Default is 100.
    
    Returns:
        pd.DataFrame: A DataFrame containing the comments with their publish date and text.
    """
    if order not in ['time', 'relevance']:
        raise ValueError("Invalid value for order parameter. Valid values are: 'time', 'relevance'.")

    url = 'https://www.googleapis.com/youtube/v3/commentThreads'
    params = {
        'key': API_KEY,
        'videoId': VIDEO_ID,
        'part': 'snippet',
        'maxResults': 100,  # API max results per page
        'textFormat': 'plainText',
        'order': order
    }

    comments = []
    total_fetched = 0

    while total_fetched < n:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            response.raise_for_status()
        data = response.json()
        time.sleep(1)
        
        for item in data.get('items', []):
            item_details = item['snippet']['topLevelComment']['snippet']
            comments.append(item_details['textDisplay'])
            total_fetched += 1
            if total_fetched >= n:
                break

        if 'nextPageToken' not in data:
            break

        params['pageToken'] = data['nextPageToken']
    
    series = pd.Series(comments)
    return series

if __name__ == '__main__':
    # testing
    API_KEY = 'AIzaSyCU5HvKL7_pZtcT8oCsj2N56gifJJfCiQo'
    VIDEO_ID = 'fklHBWow8vE'

    series = extract_video_comments(API_KEY, VIDEO_ID, order='relevance')
    print(series.head())