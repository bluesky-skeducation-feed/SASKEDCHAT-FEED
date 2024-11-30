import httpx
import asyncio
import time
from datetime import datetime, timedelta
import random
import statistics
from typing import List, Dict
import concurrent.futures
import json
from tqdm import tqdm  # For progress bars

class PerformanceTest:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: Dict[str, List[float]] = {}
        self.users = []
        self.posts = []

    async def generate_test_data(self, num_users: int = 100, posts_per_user: int = 50):
        """Generate large dataset of users and posts"""
        print(f"\nGenerating test data: {num_users} users with {posts_per_user} posts each...")
        
        # Generate users
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
            self.users.append(user)

        # Generate posts
        topics = ["education", "technology", "assessment", "curriculum", "pedagogy"]
        additional_hashtags = ["#EdChat", "#EdTech", "#Teaching", "#Learning"]
        
        for user in self.users:
            for i in range(posts_per_user):
                topic = random.choice(topics)
                extra_hashtag = random.choice(additional_hashtags)
                post_time = datetime.now() - timedelta(days=random.randint(0, 30))
                
                post = {
                    "uri": f"at://{user['did']}/post/{i}",
                    "cid": f"cid{random.randint(10000, 99999)}",
                    "author": {"did": user["did"]},
                    "record": {
                        "text": f"Post about {topic} #{i} #SaskEdChat {extra_hashtag}",
                        "createdAt": post_time.isoformat()
                    }
                }
                self.posts.append(post)

        print(f"Generated {len(self.users)} users and {len(self.posts)} posts")

    async def measure_request_time(self, func, *args) -> float:
        """Measure time taken for a request"""
        start_time = time.time()
        await func(*args)
        return time.time() - start_time

    async def add_subscriber(self, subscription: dict) -> float:
        """Add a subscriber and measure response time"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/subscription",
                json=subscription
            )
            return response.status_code == 200

    async def add_post(self, post: dict) -> float:
        """Add a post and measure response time"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/post",
                json=post
            )
            return response.status_code == 200

    async def test_write_performance(self):
        """Test write performance with concurrent requests"""
        print("\nTesting write performance...")
        
        # Test subscriber additions
        print("\nTesting concurrent subscriber additions...")
        subscriber_times = []
        async with httpx.AsyncClient() as client:
            tasks = []
            for user in tqdm(self.users):
                start_time = time.time()
                response = await client.post(
                    f"{self.base_url}/subscription",
                    json=user['subscription']
                )
                subscriber_times.append(time.time() - start_time)
                
        self.results['subscriber_addition'] = subscriber_times
        
        # Test post additions
        print("\nTesting concurrent post additions...")
        post_times = []
        async with httpx.AsyncClient() as client:
            batch_size = 100  # Process posts in batches to avoid overwhelming the server
            for i in range(0, len(self.posts), batch_size):
                batch = self.posts[i:i+batch_size]
                batch_start = time.time()
                
                tasks = []
                for post in batch:
                    start_time = time.time()
                    response = await client.post(
                        f"{self.base_url}/post",
                        json=post
                    )
                    post_times.append(time.time() - start_time)
                    
                print(f"Batch {i//batch_size + 1} completed in {time.time() - batch_start:.2f}s")
                
        self.results['post_addition'] = post_times

    async def test_read_performance(self):
        """Test read performance with concurrent requests"""
        print("\nTesting read performance...")
        
        # Test feed retrieval with different limits
        limits = [10, 50, 100]
        for limit in limits:
            print(f"\nTesting feed retrieval with limit={limit}...")
            feed_times = []
            async with httpx.AsyncClient() as client:
                for _ in tqdm(range(100)):  # Make 100 requests for each limit
                    start_time = time.time()
                    response = await client.get(f"{self.base_url}/feed?limit={limit}")
                    feed_times.append(time.time() - start_time)
                    
            self.results[f'feed_retrieval_limit_{limit}'] = feed_times

    def print_results(self):
        """Print performance test results"""
        print("\nPerformance Test Results:")
        print("=" * 50)
        
        for test_name, times in self.results.items():
            print(f"\n{test_name}:")
            print(f"  Average response time: {statistics.mean(times):.4f}s")
            print(f"  Median response time: {statistics.median(times):.4f}s")
            print(f"  95th percentile: {sorted(times)[int(len(times)*0.95)]:.4f}s")
            print(f"  Min time: {min(times):.4f}s")
            print(f"  Max time: {max(times):.4f}s")
            print(f"  Total requests: {len(times)}")

        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': {k: {
                    'mean': statistics.mean(v),
                    'median': statistics.median(v),
                    'p95': sorted(v)[int(len(v)*0.95)],
                    'min': min(v),
                    'max': max(v),
                    'count': len(v)
                } for k, v in self.results.items()}
            }, f, indent=2)
        print(f"\nDetailed results saved to {filename}")

async def run_performance_tests():
    """Run all performance tests"""
    test = PerformanceTest()
    
    try:
        # Generate test data
        await test.generate_test_data(num_users=100, posts_per_user=50)
        
        # Run write performance tests
        await test.test_write_performance()
        
        # Run read performance tests
        await test.test_read_performance()
        
        # Print results
        test.print_results()
        
    except Exception as e:
        print(f"Error during performance testing: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting performance tests...")
    asyncio.run(run_performance_tests())