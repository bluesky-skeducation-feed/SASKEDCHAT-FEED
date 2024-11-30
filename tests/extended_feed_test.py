import httpx
import json
from datetime import datetime, timedelta
import random

# Base URL for local testing
BASE_URL = "http://localhost:8000"


def generate_test_data():
    """Generate realistic test data for feed testing"""
    # Create a list of realistic users
    users = [
        {
            "handle": "teacher_sarah.bsky.social",
            "did": "did:plc:sarah789",
            "role": "Teacher",
        },
        {
            "handle": "principal_mike.bsky.social",
            "did": "did:plc:mike456",
            "role": "Principal",
        },
        {"handle": "edtech_amy.bsky.social", "did": "did:plc:amy123", "role": "EdTech"},
        {
            "handle": "curriculum_john.bsky.social",
            "did": "did:plc:john234",
            "role": "Curriculum",
        },
        {
            "handle": "librarian_pat.bsky.social",
            "did": "did:plc:pat567",
            "role": "Librarian",
        },
        {"handle": "stem_alex.bsky.social", "did": "did:plc:alex890", "role": "STEM"},
        {"handle": "arts_maria.bsky.social", "did": "did:plc:maria432", "role": "Arts"},
        {
            "handle": "sped_david.bsky.social",
            "did": "did:plc:david765",
            "role": "Special Ed",
        },
    ]

    # Create subscribers
    subscribers = [
        {
            "service": {"handle": user["handle"]},
            "subject": {"did": user["did"]},
            "createdAt": (
                datetime.now() - timedelta(days=random.randint(1, 30))
            ).isoformat(),
        }
        for user in users
    ]

    # List of realistic education-related topics and hashtags
    topics = [
        "professional development",
        "student engagement",
        "assessment strategies",
        "digital literacy",
        "inclusive education",
        "project-based learning",
        "classroom management",
        "educational technology",
    ]

    additional_hashtags = [
        "#EdChat",
        "#EdTech",
        "#Education",
        "#Teaching",
        "#ProfDev",
        "#EdLeadership",
        "#Teachers",
        "#SchoolAdmin",
    ]

    # Generate posts with realistic content
    posts = []
    base_time = datetime.now() - timedelta(days=7)

    for i in range(30):  # Generate 30 posts
        user = random.choice(users)
        topic = random.choice(topics)
        extra_hashtag = random.choice(additional_hashtags)

        # Generate realistic post content
        contents = [
            f"Excited to discuss {topic} in tonight's #SaskEdChat! {extra_hashtag}",
            f"Just shared some thoughts on {topic} - what are your experiences? #SaskEdChat {extra_hashtag}",
            f"Looking for resources on {topic}. Any recommendations? #SaskEdChat",
            f"Great conversation about {topic} happening now in #SaskEdChat {extra_hashtag}",
            f"Here's what works for me with {topic} - sharing in #SaskEdChat today!",
            f"Interesting research on {topic} - thoughts? #SaskEdChat {extra_hashtag}",
            f"How do you handle {topic} in your classroom? #SaskEdChat",
            f"Collaborative discussion on {topic} starting now! #SaskEdChat {extra_hashtag}",
        ]

        post_time = base_time + timedelta(
            hours=random.randint(0, 168)
        )  # Random time within past week

        posts.append(
            {
                "uri": f"at://{user['did']}/post/{i+1}",
                "cid": f"cid{random.randint(1000, 9999)}",
                "author": {"did": user["did"]},
                "record": {
                    "text": random.choice(contents),
                    "createdAt": post_time.isoformat(),
                },
            }
        )

    # Sort posts by creation time
    posts.sort(key=lambda x: x["record"]["createdAt"])

    return subscribers, posts


def test_health_detailed():
    """Detailed test of the health check endpoint"""
    try:
        print("\nTesting health check with detailed output...")
        response = httpx.get(f"{BASE_URL}/healthcheck")
        print(f"Status code: {response.status_code}")
        print(f"Response content: {response.text}")
        print(f"Response headers: {dict(response.headers)}")
        return response.status_code == 200
    except httpx.RequestError as e:
        print(f"Request failed: {str(e)}")
        return False


def run_extended_test():
    """Run extended feed testing"""
    subscribers, posts = generate_test_data()

    try:
        # Test health check with more detailed output
        print("\n1. Testing health check...")
        health_response = httpx.get(f"{BASE_URL}/healthcheck")
        print(f"Health check status code: {health_response.status_code}")
        print(f"Health check response: {health_response.text}")

        if health_response.status_code != 200:
            print("\n❌ Health check failed - checking server status...")
            try:
                # Try the root endpoint
                root_response = httpx.get(f"{BASE_URL}/")
                print(f"Root endpoint status: {root_response.status_code}")
                print(f"Root response: {root_response.text}")
            except Exception as e:
                print(f"Could not connect to root endpoint: {str(e)}")

            raise AssertionError(
                f"Health check failed with status {health_response.status_code}"
            )

        print("✓ Health check passed")

        # Add subscribers
        print("\n2. Adding subscribers...")
        for subscriber in subscribers:
            response = httpx.post(f"{BASE_URL}/subscription", json=subscriber)
            print(f"✓ Added subscriber {subscriber['service']['handle']}")
            assert response.status_code == 200

        # Add posts
        print("\n3. Adding posts...")
        posts_added = 0
        for post in posts:
            response = httpx.post(f"{BASE_URL}/post", json=post)
            if response.status_code == 200:
                posts_added += 1
            assert response.status_code == 200
        print(f"✓ Added {posts_added} posts")

        # Test feed with different limits
        print("\n4. Testing feed with different limits...")
        response = httpx.get(f"{BASE_URL}/feed")
        feed_data = response.json()
        print(f"✓ Default feed returned {len(feed_data['feed'])} posts")

        response = httpx.get(f"{BASE_URL}/feed?limit=5")
        feed_data_small = response.json()
        print(f"✓ Limited feed returned {len(feed_data_small['feed'])} posts")

        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feed_test_results_{timestamp}.json"

        results = {
            "timestamp": datetime.now().isoformat(),
            "total_subscribers": len(subscribers),
            "total_posts": len(posts),
            "feed_sample": feed_data,
        }

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ All tests completed successfully! Results saved to {filename}")

    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        raise


if __name__ == "__main__":
    # First, test just the health check
    print("Testing health check endpoint...")
    if test_health_detailed():
        print("Health check OK - proceeding with full test...")
        run_extended_test()
    else:
        print("\nHealth check failed - please check if:")
        print("1. The FastAPI server is running")
        print("2. The server is accessible at", BASE_URL)
        print("3. The healthcheck endpoint is properly defined in main.py")
        print(
            "\nTry accessing",
            f"{BASE_URL}/docs",
            "in your browser to check if the server is running.",
        )
