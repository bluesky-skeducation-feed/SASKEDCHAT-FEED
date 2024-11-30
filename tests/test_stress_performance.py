import httpx
import asyncio
import time
from datetime import datetime, timedelta
import random
import statistics
import concurrent.futures
import json

class PerformanceTest:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.metrics = {}
        self.test_data = {
            "users": [],
            "posts": []
        }

    async def generate_large_dataset(self, num_users: int = 100, posts_per_user: int = 50):
        """Generate a large test dataset"""
        print(f"\nGenerating test dataset with {num_users} users and {posts_per_user} posts each...")
        
        # Generate test users
        for i in range(num_users):
            user = {
                "handle": f"test_user_{i}.bsky.social",
                "did": f"did:plc:test{i}",
                "subscription": {
                    "service": {"handle": f"test_user_{i}.bsky.social"},
                    "subject": {"did": f"did:plc:test{i}"},
                    "createdAt": datetime.now().isoformat()
                }
            }
            self.test_data["users"].append(user)

        # Generate test posts
        hashtags = ["#SaskEdChat", "#SKEd", "#SaskEd", "#EdChat"]
        topics = ["education", "technology", "assessment", "curriculum", "pedagogy"]
        
        for user in self.test_data["users"]:
            for i in range(posts_per_user):
                hashtag_count = random.randint(1, 3)
                selected_hashtags = random.sample(hashtags, hashtag_count)
                topic = random.choice(topics)
                post_time = datetime.now() - timedelta(days=random.randint(0, 30))
                
                post = {
                    "uri": f"at://{user['did']}/post/{i}",
                    "cid": f"cid{random.randint(10000, 99999)}",
                    "author": {"did": user["did"]},
                    "record": {
                        "text": f"Post about {topic} with {' '.join(selected_hashtags)}",
                        "createdAt": post_time.isoformat()
                    }
                }
                self.test_data["posts"].append(post)

        print(f"Generated {len(self.test_data['users'])} users and {len(self.test_data['posts'])} posts")

    async def test_data_ingestion(self):
        """Test performance of data ingestion"""
        print("\nTesting data ingestion performance...")
        
        # Add subscribers
        start_time = time.time()
        subscription_times = []
        
        for user in self.test_data["users"]:
            sub_start = time.time()
            response = httpx.post(f"{self.base_url}/subscription", json=user["subscription"])
            subscription_times.append(time.time() - sub_start)
            assert response.status_code == 200
        
        total_subscription_time = time.time() - start_time
        
        # Add posts in batches
        post_times = []
        batch_size = 50
        total_posts = len(self.test_data["posts"])
        
        start_time = time.time()
        for i in range(0, total_posts, batch_size):
            batch = self.test_data["posts"][i:i+batch_size]
            batch_start = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(
                        httpx.post, 
                        f"{self.base_url}/post", 
                        json=post
                    ) for post in batch
                ]
                for future in concurrent.futures.as_completed(futures):
                    response = future.result()
                    post_times.append(time.time() - batch_start)
                    assert response.status_code == 200
        
        total_post_time = time.time() - start_time
        
        self.metrics["ingestion"] = {
            "subscription_stats": {
                "total_time": total_subscription_time,
                "average_time": statistics.mean(subscription_times),
                "median_time": statistics.median(subscription_times),
                "p95_time": sorted(subscription_times)[int(len(subscription_times) * 0.95)],
                "count": len(subscription_times)
            },
            "post_stats": {
                "total_time": total_post_time,
                "average_time": statistics.mean(post_times),
                "median_time": statistics.median(post_times),
                "p95_time": sorted(post_times)[int(len(post_times) * 0.95)],
                "count": len(post_times)
            }
        }

    async def test_concurrent_reads(self, num_requests: int = 100):
        """Test concurrent read performance"""
        print("\nTesting concurrent read performance...")
        
        read_times = []
        errors = 0
        
        async def make_request():
            start_time = time.time()
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.base_url}/feed?limit=50&bypass_cache=true")
                    assert response.status_code == 200
                    return time.time() - start_time
            except Exception as e:
                nonlocal errors
                errors += 1
                return None

        tasks = [make_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        read_times = [t for t in results if t is not None]
        
        self.metrics["concurrent_reads"] = {
            "average_time": statistics.mean(read_times),
            "median_time": statistics.median(read_times),
            "p95_time": sorted(read_times)[int(len(read_times) * 0.95)],
            "errors": errors,
            "successful_requests": len(read_times),
            "total_requests": num_requests
        }

    async def test_complex_edge_cases(self):
        """Test complex edge cases"""
        print("\nTesting complex edge cases...")
        
        edge_cases = [
            # Unicode and special characters
            {
                "name": "Unicode post",
                "text": "Testing with emoji 🌟✨ and unicode üñîçødé #SaskEdChat"
            },
            # Very long post
            {
                "name": "Long post",
                "text": f"{'a' * 280}#SaskEdChat"  # Max length post
            },
            # Multiple hashtags
            {
                "name": "Multiple hashtags",
                "text": "#SaskEdChat #SKEd #EdChat #Teaching #Learning #Education"
            },
            # Special characters
            {
                "name": "Special characters",
                "text": "Test!@#$%^&*()_+ #SaskEdChat"
            },
            # HTML-like content
            {
                "name": "HTML content",
                "text": "<script>alert('test')</script> #SaskEdChat <b>Bold</b>"
            }
        ]
        
        edge_case_results = []
        user = random.choice(self.test_data["users"])
        
        for case in edge_cases:
            start_time = time.time()
            post = {
                "uri": f"at://{user['did']}/post/edge_{case['name']}",
                "cid": f"cid_edge_{case['name']}",
                "author": {"did": user["did"]},
                "record": {
                    "text": case["text"],
                    "createdAt": datetime.now().isoformat()
                }
            }
            
            response = httpx.post(f"{self.base_url}/post", json=post)
            processing_time = time.time() - start_time
            
            edge_case_results.append({
                "name": case["name"],
                "success": response.status_code == 200,
                "processing_time": processing_time,
                "response_code": response.status_code,
                "response_body": response.json()
            })
        
        self.metrics["edge_cases"] = edge_case_results

    def save_metrics(self, filename: str = None):
        """Save metrics to file"""
        if filename is None:
            filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics
            }, f, indent=2)
            
        print(f"\nMetrics saved to {filename}")

async def run_performance_tests():
    """Run all performance tests"""
    test = PerformanceTest()
    
    try:
        # Generate test data
        await test.generate_large_dataset(num_users=50, posts_per_user=20)
        
        # Run tests
        await test.test_data_ingestion()
        await test.test_concurrent_reads(num_requests=100)
        await test.test_complex_edge_cases()
        
        # Save results
        test.save_metrics()
        
        print("\nAll performance tests completed successfully!")
        
    except Exception as e:
        print(f"Error during performance testing: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(run_performance_tests())