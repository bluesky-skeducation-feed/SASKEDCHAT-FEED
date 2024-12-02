from atproto import Client
from datetime import datetime, UTC

def create_feed_generator():
    # Initialize client and login
    client = Client()
    client.login('sask-ed-feed.bsky.social', 'zhuv-qgty-yvwr-5qny')

    # Feed generator configuration
    feed_generator = {
        'did': 'did:web:web-production-96221.up.railway.app',
        'displayName': 'SaskEdChat Feed',
        'description': 'A feed aggregating posts with Saskatchewan education-related hashtags',
        'createdAt': datetime.now(UTC).isoformat(),
        'labels': {
            '$type': 'com.atproto.label.defs#selfLabels',
            'values': [
                {
                    'val': 'edu',
                },
            ],
        },
    }

    # Register the feed generator
    response = client.com.atproto.repo.put_record({
        'repo': client.me.did,
        'collection': 'app.bsky.feed.generator',
        'rkey': 'saskedchat',
        'record': feed_generator
    })
    
    print(f"Feed generator created: {response}")

if __name__ == "__main__":
    create_feed_generator()
