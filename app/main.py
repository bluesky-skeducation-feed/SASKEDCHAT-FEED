from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from atproto import Client
import logging
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager  # For @contextmanager
import psycopg2
from psycopg2.extras import DictCursor

MONITORED_HASHTAGS = [
    "#SaskEdChat",
    "#sked",
    "#SaskTEachers",
    "#SKTeachers",
    "#SaskEAs",
    "#SaskEd",
    "#SKEd",
]
# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get the root directory and env path with Railway consideration
root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env" if not os.getenv("RAILWAY_ENVIRONMENT") else None

# Load environment variables from the correct path
if env_path:
    load_dotenv(env_path)


def get_db_path():
    """Get database path from environment or use default"""
    if os.getenv("RAILWAY_ENVIRONMENT"):
        return ":memory:"
    return "app/data/saskedchat.db"


# Cache Implementation
class FeedCache:
    def __init__(self, ttl_seconds: int = 60, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache = {}
        self._timestamps = {}

    def get(self, key: str) -> Optional[dict]:
        if key in self._cache:
            timestamp = self._timestamps[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                return self._cache[key]
            else:
                del self._cache[key]
                del self._timestamps[key]
        return None

    def set(self, key: str, value: dict):
        if len(self._cache) >= self.max_size:
            # Remove oldest item
            oldest_key = min(self._timestamps.items(), key=lambda x: x[1])[0]
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]
        self._cache[key] = value
        self._timestamps[key] = datetime.now()

    def clear(self):
        """Clear all cached items"""
        self._cache.clear()
        self._timestamps.clear()


# Optimized Database Class
class Database:
    def __init__(self):
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL environment variable is not set")
        self.connection = psycopg2.connect(db_url, sslmode="require")
        self._init_db()

        def __del__(self):
            if hasattr(self, "connection") and self.connection:
                self.connection.close()

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor"""
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    def _init_db(self):
        with self.get_cursor() as cursor:
            # Create tables
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS subscribers (
                    did TEXT PRIMARY KEY,
                    handle TEXT,
                    timestamp BIGINT
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS posts (
                    uri TEXT PRIMARY KEY,
                    cid TEXT,
                    author TEXT,
                    text TEXT,
                    timestamp BIGINT
                )
            """
            )

            # Create indexes
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_posts_timestamp 
                ON posts(timestamp DESC)
            """
            )


# Pydantic models
class Post(BaseModel):
    uri: str
    cid: str
    author: Dict
    record: Dict


class ServiceInfo(BaseModel):
    handle: str

    @field_validator("handle")
    @classmethod
    def validate_handle(cls, v):
        if not v or not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError("Handle cannot be empty")
        return v.strip()


class SubjectInfo(BaseModel):
    did: str

    @field_validator("did")
    @classmethod
    def validate_did(cls, v):
        if not v or not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError("DID cannot be empty")
        if not v.startswith("did:"):
            raise ValueError('DID must start with "did:"')
        return v.strip()


class Subscription(BaseModel):
    service: ServiceInfo
    subject: SubjectInfo
    createdAt: str

    @field_validator("createdAt")
    @classmethod
    def validate_created_at(cls, v):
        try:
            datetime.fromisoformat(v)
            return v
        except (TypeError, ValueError):
            raise ValueError("Invalid ISO format datetime")


class FeedPost(BaseModel):
    post: Dict


class FeedResponse(BaseModel):
    cursor: Optional[str]  # This is still called 'cursor' in the response
    feed: List[FeedPost]

    def model_dump(self):
        return {
            "cursor": self.cursor,
            "feed": [post.model_dump() for post in self.feed],
        }


class FeedGeneratorView(BaseModel):
    feed: List[dict]
    cursor: Optional[str] = None


# Initialize database and cache
db = Database()
feed_cache = FeedCache(ttl_seconds=60)


# Initialize Bluesky client with error handling
def init_bluesky_client():
    """Initialize the Bluesky client with error handling and logging."""
    # Add debug logging for environment variables
    logger.info(f"Loading environment variables from: {env_path}")

    handle = os.getenv("BSKY_HANDLE")
    app_password = os.getenv("BSKY_APP_PASSWORD")

    logger.info(f"Loaded handle: {handle}")

    if not handle:
        logger.error("BSKY_HANDLE environment variable not found")
        raise ValueError("Missing BSKY_HANDLE in environment variables")

    if not app_password:
        logger.error("BSKY_APP_PASSWORD environment variable not found")
        raise ValueError("Missing BSKY_APP_PASSWORD in environment variables")

    try:
        client = Client()
        logger.info("Attempting to login to Bluesky...")
        logger.info(f"Using handle: {handle}")

        client.login(handle, app_password)
        logger.info("Successfully logged in to Bluesky")
        return client
    except Exception as e:
        logger.error(f"Failed to login to Bluesky: {str(e)}")
        logger.exception("Detailed login error:")
        raise ConnectionError(f"Bluesky login failed: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown: cleanup connections
    if hasattr(db, "connection") and db.connection:
        db.connection.close()
    feed_cache.clear()


# Initialize FastAPI app with lifespan
app = FastAPI(title="SaskEdChat Feed", lifespan=lifespan)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize Bluesky client
try:
    app.state.bsky_client = init_bluesky_client()
except (ValueError, ConnectionError) as e:
    logger.error(f"Could not initialize Bluesky client: {str(e)}")
    app.state.bsky_client = None


@app.get("/healthcheck")
async def healthcheck(request: Request):
    """Check if the service is running and database is accessible"""
    try:
        with db.get_cursor() as cursor:
            cursor.execute("SELECT 1")

        client_status = "connected" if request.app.state.bsky_client else "disconnected"
        return {
            "status": "healthy",
            "database": "connected",
            "bluesky_client": client_status,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/subscription")
async def handle_subscription(subscription: Subscription):
    """Handle new subscription requests"""
    try:
        created_at = datetime.fromisoformat(subscription.createdAt)

        with db.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO subscribers (did, handle, timestamp)
                VALUES (?, ?, ?)
                """,
                (
                    subscription.subject.did,
                    subscription.service.handle,
                    int(created_at.timestamp() * 1000),
                ),
            )
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Subscription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/post")
async def handle_post(post: Post, response: Response):
    try:
        logger.info("=== Starting Post Processing ===")
        logger.info(f"Post URI: {post.uri}")
        logger.info(f"Post Text: {post.record.get('text', '')}")
        logger.info(f"Author DID: {post.author['did']}")

        # Check if author is a subscriber first
        with db.get_cursor() as cursor:
            logger.info("Checking subscriber status...")
            cursor.execute(
                "SELECT did FROM subscribers WHERE did = ?", (post.author["did"],)
            )
            subscriber = cursor.fetchone()
            logger.info(f"Subscriber found: {subscriber}")

            # Only check hashtags if user is a subscriber
            post_text = post.record.get("text", "").lower()
            logger.info(f"Checking text for hashtags: {post_text}")

            found_hashtags = [
                tag for tag in MONITORED_HASHTAGS if tag.lower() in post_text
            ]
            logger.info(f"Found hashtags: {found_hashtags}")

            if found_hashtags:
                logger.info("Attempting to store post...")
                try:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO posts (uri, cid, author, text, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            post.uri,
                            post.cid,
                            post.author["did"],
                            post_text,
                            int(
                                datetime.fromisoformat(
                                    post.record["createdAt"]
                                ).timestamp()
                                * 1000
                            ),
                        ),
                    )
                    logger.info("Post stored successfully")
                except Exception as e:
                    logger.error(f"Database error while storing post: {str(e)}")
                    raise

        return {
            "status": "success",
            "matched_hashtags": found_hashtags,
            "user": post.author["did"],
        }

    except Exception as e:
        logger.error(f"Post processing error: {str(e)}")
        logger.exception("Detailed error:")
        return {"status": "error", "detail": str(e)}


@app.get("/feed", response_model=FeedResponse)
async def get_feed(
    limit: Optional[int] = 50,
    page_cursor: Optional[str] = None,
    bypass_cache: bool = False,
):
    """Get the feed of SaskEdChat posts with caching"""
    logger.debug(f"Feed request received - limit: {limit}, page_cursor: {page_cursor}")

    cache_key = f"feed:limit={limit}:cursor={page_cursor}"

    # Only check cache if not bypassing
    if not bypass_cache:
        cached_response = feed_cache.get(cache_key)
        if cached_response:
            logger.debug("Returning cached response")
            return cached_response

    try:
        with db.get_cursor() as cursor:
            query = """
                SELECT DISTINCT p.* 
                FROM posts p
                INNER JOIN subscribers s ON p.author = s.did
                WHERE p.text LIKE '%#SaskEdChat%'
            """

            params = []
            if page_cursor:
                query += " AND p.timestamp < ?"
                params.append(int(page_cursor))

            query += " ORDER BY p.timestamp DESC LIMIT ?"
            params.append(limit)

            logger.debug(f"Executing query: {query}")

            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                return FeedResponse(cursor=None, feed=[])

            feed = []
            for row in rows:
                feed.append(
                    FeedPost(
                        post={
                            "uri": row[0],
                            "cid": row[1],
                            "author": row[2],
                            "record": {
                                "text": row[3],
                                "createdAt": datetime.fromtimestamp(
                                    row[4] / 1000
                                ).isoformat(),
                            },
                        }
                    )
                )

            next_cursor = str(rows[-1][4]) if rows else None
            response = FeedResponse(cursor=next_cursor, feed=feed)

            # Only cache if not bypassing
            if not bypass_cache:
                feed_cache.set(cache_key, response.model_dump())

            return response

    except Exception as e:
        logger.error(f"Feed error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/.well-known/did.json")
async def did_json():
    return {
        "@context": ["https://www.w3.org/ns/did/v1"],
        "id": "did:web:web-production-6afef.up.railway.app",
        "service": [
            {
                "id": "#bsky_fg",
                "type": "BskyFeedGenerator",
                "serviceEndpoint": "https://web-production-6afef.up.railway.app",
            }
        ],
    }


@app.get("/xrpc/app.bsky.feed.getFeedSkeleton")
async def get_feed_skeleton(
    feed: str,
    cursor: Optional[str] = None,
    limit: Optional[int] = 50,
):
    try:
        with db.get_cursor() as cursor_db:
            params = []
            query = """
                SELECT DISTINCT p.uri, p.cid, p.timestamp 
                FROM posts p
                INNER JOIN subscribers s ON p.author = s.did
                WHERE p.text LIKE '%#SaskEdChat%'
            """
            
            if cursor:
                query += " AND p.timestamp < ?"
                params.append(int(cursor))
            
            query += " ORDER BY p.timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(int(limit))
            
            cursor_db.execute(query, params)
            rows = cursor_db.fetchall()
            
            # Handle empty results
            if not rows:
                return {
                    "cursor": None,
                    "feed": []
                }
            
            feed_items = [
                {
                    "post": row[0],  # uri
                }
                for row in rows
            ]
            
            # Only set next_cursor if we have results
            next_cursor = str(rows[-1][2]) if rows else None
            
            return {
                "cursor": next_cursor,
                "feed": feed_items
            }
            
    except Exception as e:
        logger.error(f"Feed error: {str(e)}")
        logger.exception("Detailed feed error:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/xrpc/app.bsky.feed.describeFeedGenerator")
async def describe_feed_generator():
    return {
        "did": "did:web:web-production-6afef.up.railway.app",
        "feeds": [
            {
                "uri": "at://did:web:web-production-6afef.up.railway.app/app.bsky.feed.generator/saskedchat",
                "name": "saskedchat",
                "displayName": "SaskEdChat Feed",
                "description": "A feed aggregating posts with Saskatchewan education-related hashtags",
            }
        ],
    }


@app.get("/debug/posts")
async def debug_posts():
    """Debug endpoint to check posts in database"""
    try:
        with db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM posts 
                ORDER BY timestamp DESC 
                LIMIT 10
            """
            )
            posts = cursor.fetchall()

            cursor.execute("SELECT COUNT(*) FROM posts")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM subscribers")
            sub_count = cursor.fetchone()[0]

            return {
                "total_posts": total,
                "total_subscribers": sub_count,
                "recent_posts": posts,
            }
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug/subscribers")
async def debug_subscribers():
    """Debug endpoint to check subscribers"""
    try:
        with db.get_cursor() as cursor:
            cursor.execute("SELECT * FROM subscribers")
            subs = cursor.fetchall()
            return {"subscribers": subs}
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug/subscribe")
async def debug_subscribe():
    """Debug endpoint to add initial subscriber"""
    try:
        with db.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO subscribers (did, handle, timestamp)
                VALUES (?, ?, ?)
                """,
                (
                    "did:web:web-production-96221.up.railway.app",
                    "sask-ed-feed.bsky.social",
                    int(datetime.now().timestamp() * 1000),
                ),
            )
            return {"status": "success", "message": "Added initial subscriber"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug/add-test-post")
async def add_test_post():
    """Debug endpoint to add a test post"""
    try:
        with db.get_cursor() as cursor:
            test_post = {
                "uri": "test_uri",
                "cid": "test_cid",
                "author": "did:web:web-production-96221.up.railway.app",
                "text": "This is a test post #SaskEdChat",
                "timestamp": int(datetime.now().timestamp() * 1000),
            }

            cursor.execute(
                """
                INSERT OR REPLACE INTO posts (uri, cid, author, text, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    test_post["uri"],
                    test_post["cid"],
                    test_post["author"],
                    test_post["text"],
                    test_post["timestamp"],
                ),
            )

            return {"status": "success", "post": test_post}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
