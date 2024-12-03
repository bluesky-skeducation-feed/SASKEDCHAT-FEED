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
from psycopg2 import pool, errors, extensions
from psycopg2.extras import DictCursor


extensions.set_wait_callback(None)

MONITORED_HASHTAGS = [
    "#SaskEdChat",
    "#sked",
    "#SaskTeachers",
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


class Database:
    def __init__(self):
        try:
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL environment variable is not set")
            self.pool = pool.SimpleConnectionPool(
                minconn=1, maxconn=10, dsn=db_url, sslmode="require"
            )
            self._init_db()  # Call _init_db after initializing pool
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {str(e)}")
            raise

    def __del__(self):
        if hasattr(self, "pool"):
            self.pool.closeall()

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor with error handling"""
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=DictCursor)
            yield cursor
            conn.commit()
        except errors.UniqueViolation as e:
            conn.rollback()
            logger.error(f"Unique constraint violation: {str(e)}")
            raise HTTPException(status_code=409, detail="Resource already exists")
        except errors.ForeignKeyViolation as e:
            conn.rollback()
            logger.error(f"Foreign key violation: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid reference")
        except errors.OperationalError as e:
            conn.rollback()
            logger.error(f"Database operational error: {str(e)}")
            raise HTTPException(status_code=503, detail="Database unavailable")
        except Exception as e:
            conn.rollback()
            logger.error(f"Unexpected database error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            cursor.close()
            self.pool.putconn(conn)

    def _init_db(self):
        """Initialize database tables with error handling"""
        with self.get_cursor() as cursor:
            try:
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

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_posts_author 
                    ON posts(author)
                    """
                )
            except Exception as e:
                logger.error(f"Failed to initialize database tables: {str(e)}")
                raise


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
    try:
        with db.get_cursor() as cursor:
            cursor.execute("SELECT 1")
        logger.info("Database connection verified at startup")
    except Exception as e:
        logger.error(f"Failed to verify database connection: {str(e)}")
        raise
    yield
    # Shutdown: cleanup connections
    if hasattr(db, "pool"):
        db.pool.closeall()
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


# Core Service
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


@app.get("/.well-known/did.json")
async def did_json():
    return {
        "@context": ["https://www.w3.org/ns/did/v1"],
        "id": "did:plc:yhebq6pwmyhlhdyhosu7jpmi",
        "alsoKnownAs": [],
        "authentication": [],
        "verificationMethod": [],
        "service": [
            {
                "id": "#bsky_fg",
                "type": "BskyFeedGenerator",
                "serviceEndpoint": "https://web-production-6afef.up.railway.app",
            }
        ],
    }


# Feed Generator Endpoints
@app.get("/xrpc/app.bsky.feed.describeFeedGenerator")
async def describe_feed_generator():
    return {
        "did": "did:plc:yhebq6pwmyhlhdyhosu7jpmi",  # Updated to match your assigned DID
        "feeds": [
            {
                "uri": "at://did:plc:yhebq6pwmyhlhdyhosu7jpmi/app.bsky.feed.generator/saskedchat",
                "name": "saskedchat",
                "displayName": "SaskEdChat Feed",
                "description": "A feed aggregating posts with Saskatchewan education-related hashtags",
            }
        ],
    }


@app.get("/xrpc/app.bsky.feed.getFeedSkeleton")
async def get_feed_skeleton(
    feed: str,
    cursor: Optional[str] = None,
    limit: Optional[int] = 30,
):
    try:
        logger.info(f"Feed request received - cursor: {cursor}, limit: {limit}")

        # Build ILIKE conditions for all hashtags
        hashtag_conditions = " OR ".join(
            [f"p.text ILIKE %(hashtag{i})s" for i, _ in enumerate(MONITORED_HASHTAGS)]
        )

        # Start with base query and parameters
        query = f"""
            SELECT DISTINCT p.uri, p.cid, p.timestamp 
            FROM posts p
            INNER JOIN subscribers s ON p.author = s.did
            WHERE ({hashtag_conditions})
            AND p.cid != 'test_cid'  -- Exclude test posts with invalid CIDs
        """

        # Create parameters dict with all hashtags
        params = {f"hashtag{i}": f"%{tag}%" for i, tag in enumerate(MONITORED_HASHTAGS)}

        # Add cursor condition if present
        if cursor:
            query += " AND p.timestamp < %(cursor)s"
            params["cursor"] = int(cursor)

        # Add ordering and limit
        query += " ORDER BY p.timestamp DESC LIMIT %(limit)s"
        params["limit"] = limit if limit is not None else 30

        with db.get_cursor() as cursor_db:
            try:
                cursor_db.execute(query, params)
                rows = cursor_db.fetchall()
                logger.info(f"Retrieved {len(rows)} rows from database")
            except Exception as db_error:
                logger.error(f"Database error: {str(db_error)}")
                raise HTTPException(status_code=500, detail=str(db_error))

            if not rows:
                logger.info("No posts found")
                response = {"feed": [], "cursor": None}
                logger.info(f"Empty response: {response}")
                return response

            feed_items = []
            for row in rows:
                post_uri = row[0]
                # Log the full row data for debugging
                logger.info(f"Processing row: {row}")

                if not post_uri.startswith("at://"):
                    post_uri = f"at://{post_uri}"

                # Simpler post structure
                feed_items.append({"post": post_uri})

            response = {
                "feed": feed_items,
                "cursor": str(rows[-1][2]) if rows else None,
            }

            logger.info(f"Final response structure: {response}")
            return response

    except Exception as e:
        logger.error(f"Feed error: {str(e)}")
        logger.exception("Detailed feed error:")
        raise HTTPException(status_code=500, detail=str(e))


# Main Functionality
@app.post("/subscription")
async def handle_subscription(subscription: Subscription):
    try:
        created_at = datetime.fromisoformat(subscription.createdAt)

        with db.get_cursor() as cursor:
            try:
                cursor.execute(
                    """
                    INSERT INTO subscribers (did, handle, timestamp)
                    VALUES (%(did)s, %(handle)s, %(timestamp)s)
                    ON CONFLICT (did) DO UPDATE SET
                        handle = EXCLUDED.handle,
                        timestamp = EXCLUDED.timestamp
                    """,
                    {
                        "did": subscription.subject.did,
                        "handle": subscription.service.handle,
                        "timestamp": int(created_at.timestamp() * 1000),
                    },
                )
                return {"status": "success"}
            except errors.UniqueViolation:
                # Handle duplicate subscription
                return {"status": "success", "message": "Subscription already exists"}
            except errors.OperationalError as e:
                logger.error(f"Database operation failed: {str(e)}")
                raise HTTPException(
                    status_code=503, detail="Service temporarily unavailable"
                )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Subscription error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/post")
async def handle_post(post: Post, response: Response):
    try:
        logger.info("=== Starting Post Processing ===")

        with db.get_cursor() as cursor:
            try:
                # Check subscriber status
                cursor.execute(
                    "SELECT did FROM subscribers WHERE did = %(did)s",
                    {"did": post.author["did"]},
                )
                subscriber = cursor.fetchone()

                if not subscriber:
                    return {"status": "ignored", "reason": "Not a subscriber"}

                post_text = post.record.get("text", "").lower()
                found_hashtags = [
                    tag for tag in MONITORED_HASHTAGS if tag.lower() in post_text
                ]

                if found_hashtags:
                    cursor.execute(
                        """
                        INSERT INTO posts (uri, cid, author, text, timestamp)
                        VALUES (%(uri)s, %(cid)s, %(author)s, %(text)s, %(timestamp)s)
                        ON CONFLICT (uri) DO UPDATE SET
                            cid = EXCLUDED.cid,
                            author = EXCLUDED.author,
                            text = EXCLUDED.text,
                            timestamp = EXCLUDED.timestamp
                        """,
                        {
                            "uri": post.uri,
                            "cid": post.cid,
                            "author": post.author["did"],
                            "text": post_text,
                            "timestamp": int(
                                datetime.fromisoformat(
                                    post.record["createdAt"]
                                ).timestamp()
                                * 1000
                            ),
                        },
                    )
                    return {
                        "status": "success",
                        "matched_hashtags": found_hashtags,
                        "user": post.author["did"],
                    }
                return {"status": "ignored", "reason": "No matching hashtags"}

            except errors.UniqueViolation:
                # This is not necessarily an error for posts
                return {"status": "success", "message": "Post already exists"}
            except errors.OperationalError as e:
                logger.error(f"Database operation failed: {str(e)}")
                raise HTTPException(
                    status_code=503, detail="Service temporarily unavailable"
                )
    except Exception as e:
        logger.error(f"Post processing error: {str(e)}")
        logger.exception("Detailed error:")
        raise HTTPException(status_code=500, detail=str(e))


# Debug Endpoints
@app.get("/debug/posts")
async def debug_posts():
    try:
        with db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM posts 
                ORDER BY timestamp DESC 
                LIMIT %(limit)s
                """,
                {"limit": 10},
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
                INSERT INTO subscribers (did, handle, timestamp)
                VALUES (%(did)s, %(handle)s, %(timestamp)s)
                ON CONFLICT (did) DO UPDATE SET
                    handle = EXCLUDED.handle,
                    timestamp = EXCLUDED.timestamp
                """,
                {
                    "did": "did:plc:yhebq6pwmyhlhdyhosu7jpmi",  # Updated DID
                    "handle": "sask-ed-feed.bsky.social",
                    "timestamp": int(datetime.now().timestamp() * 1000),
                },
            )
            return {"status": "success", "message": "Added initial subscriber"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug/add-test-post")
async def add_test_post():
    """Debug endpoint to add a test post"""
    try:
        with db.get_cursor() as cursor:
            # Generate a proper Bluesky post URI
            timestamp = int(datetime.now().timestamp() * 1000)
            rkey = f"{timestamp:x}"  # Convert timestamp to hex for rkey

            # Use a valid CID format (example format)
            test_post = {
                "uri": f"at://did:plc:yhebq6pwmyhlhdyhosu7jpmi/app.bsky.feed.post/{rkey}",
                "cid": "bafyreid27zk7lbis4zw5fz4podbvhs4rrhdzw2fv47x4jweqbwixr2urm4",  # Valid CID format
                "author": "did:plc:yhebq6pwmyhlhdyhosu7jpmi",
                "text": "This is a test post #SaskEdChat",
                "timestamp": timestamp,
            }

            cursor.execute(
                """
                INSERT INTO posts (uri, cid, author, text, timestamp)
                VALUES (%(uri)s, %(cid)s, %(author)s, %(text)s, %(timestamp)s)
                ON CONFLICT (uri) DO UPDATE SET
                    cid = EXCLUDED.cid,
                    author = EXCLUDED.author,
                    text = EXCLUDED.text,
                    timestamp = EXCLUDED.timestamp
                """,
                test_post,
            )

            logger.info(
                f"Created test post with URI: {test_post['uri']} and CID: {test_post['cid']}"
            )
            return {"status": "success", "post": test_post}
    except Exception as e:
        logger.error(f"Error creating test post: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
