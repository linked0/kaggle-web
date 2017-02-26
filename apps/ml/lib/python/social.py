__author__ = 'linked0'
import twitter

CONSUMER_KEY = 'lYoNtxpKyZWk1HgpponsBRECO'
CONSUMER_SECRET = 'hIb4IOcWuESjfHqOYaajrA0liZB0ISOVFomoe01Lj1XKWWDdsy'
OAUTH_TOKEN = '52678165-V1BzVAZrFzCJUUWAgJOgytESYd8Fkz3LMtpuGcOaV'
OAUTH_TOKEN_SECRET = '12TYFfTjhmYNpfXZjGjKY785fSdo3tJK6kzpPeqFNo91Y'

twitter_api = None;

def get_twitter_api():
    global twitter_api

    if twitter_api != None:
        return

    """Retrun twitter_api"""
    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                           CONSUMER_KEY, CONSUMER_SECRET)
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api

def get_korea_woeid():
    return 23424868

def get_us_woeid():
    return 23424977

def get_world_woeid():
    return 1

def get_korea_trends():
    get_twitter_api()
    return twitter_api.trends.place(_id=get_korea_woeid())

def get_us_trends():
    get_twitter_api()
    return twitter_api.trends.place(_id=get_us_woeid())

def get_world_trends():
    get_twitter_api()
    return twitter_api.trends.place(_id=get_world_woeid())

################################################
# examples as below
def ex():
    print 'world_trends = twitter_api.trends.place(_id=WORLD_WOE_ID)'
    print 'us_trends = twitter_api.trends.place(_id=US_WOE_ID)'

    print 'world_trends'

