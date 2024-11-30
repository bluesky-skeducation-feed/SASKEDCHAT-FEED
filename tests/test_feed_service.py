import httpx
import json
from datetime import datetime
import random
import os
import sqlite3
import time
from typing import Dict, List

# Base URL for local testing
BASE_URL = "http://localhost:8000"


class TestSaskEdChatFeed:
    @classmethod
    def setup_class(cls):
        """Setup test database and clear it"""
        print("\nCleaning database before tests...")
        db_path = "app/data/saskedchat.db"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # Clear all tables
            cursor.execute("DELETE FROM posts")
            cursor.execute("DELETE FROM subscribers")
            conn.commit()

            # Verify tables are empty
            cursor.execute("SELECT COUNT(*) FROM posts")
            post_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM subscribers")
            sub_count = cursor.fetchone()[0]
            conn.close()

            print(f"Database cleaned - Posts: {post_count}, Subscribers: {sub_count}")

    def generate_valid_user(self) -> Dict:
        """Generate a valid test user"""
        user_id = random.randint(1000, 9999)
        return {
            "handle": f"test_user_{user_id}.bsky.social",
            "did": f"did:plc:test{user_id}",
        }

    def generate_valid_post(self, user: Dict, hashtags: List[str] = None) -> Dict:
        """Generate a valid test post"""
        if hashtags is None:
            hashtags = ["#SaskEdChat"]

        post_text = f"Test post {random.randint(1, 1000)} {' '.join(hashtags)}"
        return {
            "uri": f"at://{user['did']}/post/{random.randint(1000, 9999)}",
            "cid": f"cid{random.randint(1000, 9999)}",
            "author": {"did": user["did"]},
            "record": {"text": post_text, "createdAt": datetime.now().isoformat()},
        }

    def debug_feed_query(self, user_did):
        """Helper function to debug feed query directly"""
        conn = sqlite3.connect("app/data/saskedchat.db")
        cursor = conn.cursor()

        print("\nDebugging feed query:")

        # Test the exact query used in the feed endpoint
        query = """
            SELECT DISTINCT p.*
            FROM posts p
            INNER JOIN subscribers s ON p.author = s.did
            WHERE p.text LIKE ? OR p.text LIKE ?
            ORDER BY p.timestamp DESC
            LIMIT 50
        """

        cursor.execute(query, ("%#SaskEdChat%", "%#SaskEd%"))
        rows = cursor.fetchall()

        print("\nDirect query results:")
        for row in rows:
            print(f"URI: {row[0]}")
            print(f"Author: {row[2]}")
            print(f"Text: {row[3]}")
            print("---")

        # Also check the join condition
        print("\nChecking join condition:")
        cursor.execute(
            """
            SELECT p.uri, p.author, s.did
            FROM posts p
            INNER JOIN subscribers s ON p.author = s.did
            WHERE p.author = ?
        """,
            (user_did,),
        )

        join_results = cursor.fetchall()
        print(f"Join results for user {user_did}: {join_results}")

        conn.close()

    def test_healthcheck(self):
        """Test health check endpoint with cursor management"""
        print("\nTesting health check...")
        response = httpx.get(f"{BASE_URL}/healthcheck")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        print("✓ Health check passed")

    def test_cache_behavior(self):
        """Test feed cache behavior"""
        print("\nTesting cache behavior...")

        # Add test data
        user = self.generate_valid_user()
        subscription = {
            "service": {"handle": user["handle"]},
            "subject": {"did": user["did"]},
            "createdAt": datetime.now().isoformat(),
        }
        response = httpx.post(f"{BASE_URL}/subscription", json=subscription)
        assert response.status_code == 200, "Failed to create subscription"
        print("✓ Created subscriber")

        # Add some posts
        posts_added = 0
        for i in range(3):
            post = self.generate_valid_post(user)
            response = httpx.post(f"{BASE_URL}/post", json=post)
            assert response.status_code == 200, f"Failed to create post {i+1}"
            if response.json()["status"] == "success":
                posts_added += 1
            time.sleep(0.1)  # Small delay between posts

        print(f"✓ Added {posts_added} posts")

        # Verify posts are in database
        conn = sqlite3.connect("app/data/saskedchat.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM posts WHERE author = ?", (user["did"],))
        db_post_count = cursor.fetchone()[0]
        conn.close()
        print(f"✓ Found {db_post_count} posts in database")

        # Wait a moment for any async processing
        time.sleep(0.5)

        # First request - should hit database
        print("\nMaking first request (database hit)...")
        start_time = time.time()
        response1 = httpx.get(f"{BASE_URL}/feed")
        first_request_time = time.time() - start_time

        if response1.status_code != 200:
            print(f"First request failed with status {response1.status_code}")
            print(f"Response body: {response1.text}")

        assert response1.status_code == 200, f"First request failed: {response1.text}"
        feed_data = response1.json()
        print(f"Feed contains {len(feed_data['feed'])} posts")

        # Small delay to ensure cache is set
        time.sleep(0.1)

        # Second request - should hit cache
        print("\nMaking second request (cache hit)...")
        start_time = time.time()
        response2 = httpx.get(f"{BASE_URL}/feed")
        cached_request_time = time.time() - start_time

        assert response2.status_code == 200, f"Second request failed: {response2.text}"
        assert response1.json() == response2.json(), "Cache returned different data"

        print(f"First request time: {first_request_time:.4f}s")
        print(f"Cached request time: {cached_request_time:.4f}s")
        print("✓ Cache behavior test passed")

        # Clean up test data
        conn = sqlite3.connect("app/data/saskedchat.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM posts WHERE author = ?", (user["did"],))
        cursor.execute("DELETE FROM subscribers WHERE did = ?", (user["did"],))
        conn.commit()
        conn.close()

    def test_cursor_management(self):
        """Test database cursor management"""
        print("\nTesting cursor management...")

        user = self.generate_valid_user()

        # Test subscription with cursor
        subscription = {
            "service": {"handle": user["handle"]},
            "subject": {"did": user["did"]},
            "createdAt": datetime.now().isoformat(),
        }

        response = httpx.post(f"{BASE_URL}/subscription", json=subscription)
        assert response.status_code == 200

        # Test post with cursor
        post = self.generate_valid_post(user)
        response = httpx.post(f"{BASE_URL}/post", json=post)
        assert response.status_code == 200

        print("✓ Cursor management test passed")

    def test_hashtag_monitoring(self):
        """Test hashtag monitoring with various cases"""
        print("\nTesting hashtag monitoring...")

        user = self.generate_valid_user()
        subscription = {
            "service": {"handle": user["handle"]},
            "subject": {"did": user["did"]},
            "createdAt": datetime.now().isoformat(),
        }
        httpx.post(f"{BASE_URL}/subscription", json=subscription)

        test_cases = [
            {
                "name": "Standard hashtag",
                "hashtags": ["#SaskEdChat"],
                "should_accept": True,
            },
            {
                "name": "Multiple valid hashtags",
                "hashtags": ["#SaskEdChat", "#SKEd"],
                "should_accept": True,
            },
            {
                "name": "Unmonitored hashtag",
                "hashtags": ["#UnrelatedTag"],
                "should_accept": False,
            },
            {
                "name": "Mixed hashtags",
                "hashtags": ["#SaskEdChat", "#UnrelatedTag"],
                "should_accept": True,
            },
        ]

        for case in test_cases:
            print(f"\nTesting: {case['name']}")
            post = self.generate_valid_post(user, case["hashtags"])
            response = httpx.post(f"{BASE_URL}/post", json=post)
            assert response.status_code == 200

            if case["should_accept"]:
                assert response.json()["status"] == "success"
            else:
                assert response.json()["status"] == "skipped"

        print("✓ Hashtag monitoring test passed")

    def test_feed_pagination(self):
        """Test feed pagination with cursor management"""
        print("\nTesting feed pagination...")

        # Add test data
        user = self.generate_valid_user()
        subscription = {
            "service": {"handle": user["handle"]},
            "subject": {"did": user["did"]},
            "createdAt": datetime.now().isoformat(),
        }
        httpx.post(f"{BASE_URL}/subscription", json=subscription)

        # Add multiple posts
        for i in range(10):
            post = self.generate_valid_post(user)
            time.sleep(0.1)  # Ensure different timestamps
            httpx.post(f"{BASE_URL}/post", json=post)

        # Test pagination
        first_page = httpx.get(f"{BASE_URL}/feed?limit=5").json()
        assert len(first_page["feed"]) == 5

        second_page = httpx.get(
            f"{BASE_URL}/feed?limit=5&page_cursor={first_page['cursor']}"  # Changed from cursor to page_cursor
        ).json()
        assert len(second_page["feed"]) == 5

        # Verify pages are different
        first_page_ids = [post["post"]["uri"] for post in first_page["feed"]]
        second_page_ids = [post["post"]["uri"] for post in second_page["feed"]]
        assert not set(first_page_ids).intersection(set(second_page_ids))

        print("✓ Feed pagination test passed")

    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        print("\nTesting error handling...")

        # Test invalid subscription
        invalid_sub = {
            "service": {"handle": ""},  # Empty handle
            "subject": {"did": "invalid-did"},  # Invalid DID
            "createdAt": "invalid-date",  # Invalid date
        }
        response = httpx.post(f"{BASE_URL}/subscription", json=invalid_sub)
        assert response.status_code == 422

        # Test invalid post
        invalid_post = {
            "uri": "",
            "cid": "",
            "author": {},
            "record": {"text": "", "createdAt": "invalid-date"},
        }
        response = httpx.post(f"{BASE_URL}/post", json=invalid_post)
        assert response.status_code in [422, 500]

        print("✓ Error handling test passed")
    
    
    def verify_post_in_feed(self, post_data, max_attempts=5, delay=1.0):
        """Helper function to verify post appears in feed"""
        for attempt in range(max_attempts):
            print(f"\n{'='*50}")
            print(f"Attempt {attempt + 1} to find post in feed...")
            
            # Database verification
            conn = sqlite3.connect("app/data/saskedchat.db")
            cursor = conn.cursor()
            
            # Debug print query results
            print("\nExecuting query:")
            cursor.execute("""
                SELECT p.*, s.handle 
                FROM posts p
                INNER JOIN subscribers s ON p.author = s.did
                WHERE p.text LIKE '%#SaskEdChat%'
                ORDER BY p.timestamp DESC
            """)
            
            db_results = cursor.fetchall()
            print(f"Found {len(db_results)} posts in database:")
            for result in db_results:
                print(f"URI: {result[0]}")
                print(f"Text: {result[3]}")
                print("---")
            
            # Check feed endpoint with cache bypass
            feed_response = httpx.get(f"{BASE_URL}/feed?limit=100&bypass_cache=true")
            assert feed_response.status_code == 200
            
            feed_data = feed_response.json()
            print(f"\nFeed contains {len(feed_data['feed'])} posts")
            
            # Print feed contents
            print("\nPosts in feed:")
            for feed_post in feed_data["feed"]:
                print(f"URI: {feed_post['post']['uri']}")
                print(f"Text: {feed_post['post']['record']['text']}")
                print("---")
                
                if feed_post["post"]["uri"] == post_data["uri"]:
                    print(f"✓ Found matching post!")
                    conn.close()
                    return True
                    
            if attempt < max_attempts - 1:
                print(f"\nPost not found, waiting {delay} seconds...")
                time.sleep(delay)
            
            conn.close()
        
        print("\n❌ Post not found in feed after all attempts")
        # Final database check
        conn = sqlite3.connect("app/data/saskedchat.db")
        cursor = conn.cursor()
        
        # Check post
        cursor.execute("SELECT * FROM posts WHERE uri = ?", (post_data["uri"],))
        final_post = cursor.fetchone()
        print(f"Post in database: {final_post is not None}")
        if final_post:
            print(f"Post details: {final_post}")
        
        # Check subscriber
        cursor.execute("SELECT * FROM subscribers WHERE did = ?", (post_data["author"]["did"],))
        subscriber = cursor.fetchone()
        print(f"Subscriber in database: {subscriber is not None}")
        if subscriber:
            print(f"Subscriber details: {subscriber}")
        
        conn.close()
        return False


    def test_edge_cases(self):
        """Test various edge cases and boundary conditions"""
        print("\nTesting edge cases...")

        def verify_database_state(user_did):
            """Helper function to check database state"""
            conn = sqlite3.connect("app/data/saskedchat.db")
            cursor = conn.cursor()

            print("\nChecking database state:")

            # Check subscribers
            cursor.execute("SELECT * FROM subscribers WHERE did = ?", (user_did,))
            subscriber = cursor.fetchone()
            print(f"Subscriber in database: {subscriber}")

            # Check posts
            cursor.execute("SELECT * FROM posts WHERE author = ?", (user_did,))
            posts = cursor.fetchall()
            print(f"Posts in database for user: {posts}")

            conn.close()
            return subscriber is not None, posts

        def create_subscribed_user():
            user = self.generate_valid_user()
            subscription = {
                "service": {"handle": user["handle"]},
                "subject": {"did": user["did"]},
                "createdAt": datetime.now().isoformat(),
            }

            print("\nCreating subscriber:")
            print(f"User DID: {user['did']}")
            print(f"Handle: {user['handle']}")

            response = httpx.post(f"{BASE_URL}/subscription", json=subscription)
            assert (
                response.status_code == 200
            ), f"Failed to create subscriber: {response.text}"

            # Verify subscriber was created
            is_subscribed, _ = verify_database_state(user["did"])
            assert is_subscribed, "Subscriber not found in database"

            return user

        edge_case = {
            "name": "Test post",
            "post_data": lambda user: {
                "uri": f"at://{user['did']}/post/test",
                "cid": "cid_test",
                "author": {"did": user["did"]},
                "record": {
                    "text": "Test post #SaskEdChat",
                    "createdAt": datetime.now().isoformat(),
                },
            },
            "expected_status": 200,
            "check": lambda response: response.json()["status"] == "success",
        }

        # Create test user
        print("\nCreating test user...")
        user = create_subscribed_user()
        print(f"Created user with DID: {user['did']}")

        try:
            # Prepare test data
            post_data = edge_case["post_data"](user)
            print(f"Sending data: {json.dumps(post_data, indent=2)}")

            # Submit post
            response = httpx.post(f"{BASE_URL}/post", json=post_data)
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")

            # Debug feed query
            print("\nDebug feed query:")
            self.debug_feed_query(user["did"])

            # Verify response
            assert (
                response.status_code == edge_case["expected_status"]
            ), f"Expected status {edge_case['expected_status']}, got {response.status_code}"

            assert edge_case["check"](
                response
            ), f"Response check failed for {edge_case['name']}"

            # Wait a moment before checking feed
            print("Waiting for post to be available in feed...")
            time.sleep(1)

            # Verify post appears in feed - Updated call
            if response.status_code == 200 and response.json()["status"] == "success":
                assert self.verify_post_in_feed(
                    post_data, max_attempts=5, delay=1.0
                ), f"Failed to find post {post_data['uri']} in feed"

            print(f"✓ {edge_case['name']} test passed")

        except Exception as e:
            print(f"❌ {edge_case['name']} test failed: {str(e)}")
            print(f"Error details: {str(e.__class__.__name__)}: {str(e)}")
            print("\nFinal database state:")
            verify_database_state(user["did"])
            raise

        print("\n✓ Edge case test completed")

    def test_advanced_edge_cases(self):
        """Test additional edge cases and complex scenarios"""
        print("\nTesting advanced edge cases...")

        # Clean database before test
        self.setup_class()

        def create_subscribed_user():
            user = self.generate_valid_user()
            subscription = {
                "service": {"handle": user["handle"]},
                "subject": {"did": user["did"]},
                "createdAt": datetime.now().isoformat()
            }
            response = httpx.post(f"{BASE_URL}/subscription", json=subscription)
            assert response.status_code == 200, "Failed to create subscriber"
            time.sleep(0.5)  # Give time for subscription to be processed
            return user

        advanced_cases = [
            {
                "name": "Standard hashtag",
                "post_data": lambda user: {
                    "uri": f"at://{user['did']}/post/standard",
                    "cid": "cid_standard",
                    "author": {"did": user["did"]},
                    "record": {
                        "text": "Regular post with #SaskEdChat hashtag",
                        "createdAt": datetime.now().isoformat()
                    }
                },
                "expected_status": 200,
                "check": lambda response: response.json()["status"] == "success"
            }
        ]

        # Run test cases
        user = create_subscribed_user()
        print(f"\nCreated test user: {user['did']}")

        for case in advanced_cases:
            print(f"\nTesting: {case['name']}")
            try:
                # Create post
                post_data = case["post_data"](user)
                print(f"Sending post: {json.dumps(post_data, indent=2)}")
                
                response = httpx.post(f"{BASE_URL}/post", json=post_data)
                print(f"Response status: {response.status_code}")
                print(f"Response body: {response.text}")
                
                assert response.status_code == case["expected_status"]
                assert case["check"](response)

                # Wait a moment for processing
                time.sleep(1.0)

                # Verify post in feed
                assert self.verify_post_in_feed(post_data), \
                    f"Failed to find post {post_data['uri']} in feed"

                print(f"✓ {case['name']} passed")
                
            except Exception as e:
                print(f"❌ {case['name']} failed: {str(e)}")
                # Print final database state for debugging
                conn = sqlite3.connect("app/data/saskedchat.db")
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM posts")
                posts = cursor.fetchall()
                cursor.execute("SELECT * FROM subscribers")
                subs = cursor.fetchall()
                conn.close()
                print("\nFinal database state:")
                print(f"Posts: {posts}")
                print(f"Subscribers: {subs}")
                raise

        print("\n✓ All advanced edge cases completed")



if __name__ == "__main__":
    print("Running SaskEdChat Feed tests...")
    test = TestSaskEdChatFeed()

    try:
        test.setup_class()
        test.test_healthcheck()
        test.test_cache_behavior()
        test.test_cursor_management()
        test.test_hashtag_monitoring()
        test.test_feed_pagination()
        test.test_error_handling()
        test.test_edge_cases()
        test.test_advanced_edge_cases()
        print("\n✓ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        raise
