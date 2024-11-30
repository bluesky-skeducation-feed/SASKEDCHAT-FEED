import httpx
import json
from datetime import datetime, timedelta

# Base URL for local testing
BASE_URL = "http://localhost:8000"

# Sample test data
test_subscribers = [
    {
        "service": {
            "handle": "alice.bsky.social"
        },
        "subject": {
            "did": "did:plc:alice123"
        },
        "createdAt": datetime.now().isoformat()
    },
    {
        "service": {
            "handle": "bob.bsky.social"
        },
        "subject": {
            "did": "did:plc:bob456"
        },
        "createdAt": datetime.now().isoformat()
    }
]

test_posts = [
    {
        "uri": "at://did:plc:alice123/post/1",
        "cid": "cid123",
        "author": {
            "did": "did:plc:alice123"
        },
        "record": {
            "text": "Testing the #SaskEdChat feed!",
            "createdAt": datetime.now().isoformat()
        }
    },
    {
        "uri": "at://did:plc:alice123/post/2",
        "cid": "cid456",
        "author": {
            "did": "did:plc:alice123"
        },
        "record": {
            "text": "Another #SaskEdChat post for testing",
            "createdAt": (datetime.now() + timedelta(hours=1)).isoformat()
        }
    },
    {
        "uri": "at://did:plc:bob456/post/1",
        "cid": "cid789",
        "author": {
            "did": "did:plc:bob456"
        },
        "record": {
            "text": "Bob's first #SaskEdChat post",
            "createdAt": (datetime.now() + timedelta(hours=2)).isoformat()
        }
    }
]

def test_health():
    """Test the health check endpoint"""
    response = httpx.get(f"{BASE_URL}/healthcheck")
    print("Health check response:", response.json())
    assert response.status_code == 200

def add_subscribers():
    """Add test subscribers"""
    print("\nAdding subscribers...")
    for subscriber in test_subscribers:
        response = httpx.post(f"{BASE_URL}/subscription", json=subscriber)
        print(f"Added subscriber {subscriber['service']['handle']}: {response.json()}")
        assert response.status_code == 200

def add_posts():
    """Add test posts"""
    print("\nAdding posts...")
    for post in test_posts:
        response = httpx.post(f"{BASE_URL}/post", json=post)
        print(f"Added post from {post['author']['did']}: {response.json()}")
        assert response.status_code == 200

def check_feed():
    """Check the feed content"""
    print("\nChecking feed...")
    response = httpx.get(f"{BASE_URL}/feed?limit=10")
    assert response.status_code == 200
    feed_data = response.json()
    
    print("\nFeed contents:")
    for item in feed_data['feed']:
        post = item['post']
        print(f"- {post['record']['text']} (by {post['author']})")
    
    return feed_data

if __name__ == "__main__":
    try:
        # Run tests
        test_health()
        add_subscribers()
        add_posts()
        feed_data = check_feed()
        
        print("\nAll tests completed successfully!")
        
        # Save feed data for reference
        with open("feed_data.json", "w") as f:
            json.dump(feed_data, f, indent=2)
        print("\nFeed data saved to feed_data.json")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")