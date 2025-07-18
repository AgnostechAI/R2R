"""Abstractions for semantic cache functionality."""

from datetime import datetime
from typing import Any, Optional, Union
from uuid import UUID

from pydantic import Field

from .base import R2RSerializable
from .search import AggregateSearchResult
from .vector import Vector


class CacheSettings(R2RSerializable):
    """Settings for semantic cache behavior."""

    enabled: bool = Field(
        default=True,
        description="Whether semantic caching is enabled"
    )
    
    similarity_threshold: float = Field(
        default=0.85,
        description="Minimum similarity score to consider a cache hit",
        ge=0.0,
        le=1.0
    )
    
    ttl_seconds: int = Field(
        default=86400,  # 24 hours
        description="Time-to-live for cache entries in seconds. Set to 0 for indefinite storage.",
        ge=0
    )
    
    max_cache_size: int = Field(
        default=1000,
        description="Maximum number of cache entries per collection",
        ge=0
    )
    
    bypass_cache: bool = Field(
        default=False,
        description="Bypass cache lookup for this request"
    )
    
    # NEW: Cache-specific collection targeting (separate from search filters)
    cache_scope_collection_ids: Optional[list[Union[str, UUID]]] = Field(
        default=None,
        description="Specific collection IDs to use for cache scoping. If None, will extract from search filters for backward compatibility."
    )
    
    use_search_filters_for_cache_scope: bool = Field(
        default=True,
        description="Whether to extract collection IDs from search filters for cache scoping. Set to False to use only cache_scope_collection_ids."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "similarity_threshold": 0.85,
                "ttl_seconds": 0,  # 0 = never expires, 86400 = 24 hours
                "max_cache_size": 1000,
                "bypass_cache": False,
                "cache_scope_collection_ids": ["3e157b3a-8469-51db-90d9-52e7d896b49b"],
                "use_search_filters_for_cache_scope": False
            }
        }


class CacheEntry(R2RSerializable):
    """A semantic cache entry containing a query-response pair."""

    id: UUID = Field(..., description="Unique identifier for the cache entry")
    
    query: str = Field(..., description="The original query text")
    
    query_embedding: Vector = Field(..., description="Embedding of the query")
    
    generated_answer: str = Field(..., description="The cached response")
    
    search_results: AggregateSearchResult = Field(
        ..., description="The search results used to generate the response"
    )
    
    citations: Optional[list[dict]] = Field(
        None, description="Citations from the original response"
    )
    
    collection_id: UUID = Field(
        ..., description="ID of the original collection this cache entry relates to"
    )
    
    cached_at: datetime = Field(
        default_factory=datetime.now,
        description="When this entry was cached"
    )
    
    ttl_seconds: int = Field(
        default=86400,
        description="Time-to-live for this cache entry in seconds. Set to 0 for indefinite storage."
    )
    
    hit_count: int = Field(
        default=0,
        description="Number of times this cache entry has been retrieved"
    )
    
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the cache entry"
    )

    def is_expired(self) -> bool:
        """Check if the cache entry has expired.
        
        Returns:
            bool: True if expired, False if still valid or set to never expire
        """
        if self.ttl_seconds <= 0:  # TTL of 0 or negative means never expires
            return False
        
        elapsed = (datetime.now() - self.cached_at).total_seconds()
        return elapsed > self.ttl_seconds

    def increment_hit_count(self) -> None:
        """Increment the hit count for this cache entry."""
        self.hit_count += 1

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "query": "What is machine learning?",
                "query_embedding": {
                    "data": [0.1, 0.2, 0.3],
                    "type": "FIXED",
                    "length": 3
                },
                "generated_answer": "Machine learning is a subset of artificial intelligence...",
                "search_results": {},
                "citations": [],
                "collection_id": "3e157b3a-8469-51db-90d9-52e7d896b49b",
                "cached_at": "2025-01-11T10:30:00Z",
                "ttl_seconds": 86400,
                "hit_count": 5,
                "metadata": {
                    "model_used": "gpt-4",
                    "response_tokens": 150
                }
            }
        }


class CacheSearchResult(R2RSerializable):
    """Result of a semantic cache search operation."""

    cache_entry: CacheEntry = Field(..., description="The matching cache entry")
    
    similarity_score: float = Field(
        ..., description="Similarity score between query and cached query",
        ge=0.0,
        le=1.0
    )
    
    cache_hit: bool = Field(
        default=True,
        description="Whether this represents a cache hit"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "cache_entry": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "query": "What is machine learning?",
                    "generated_answer": "Machine learning is...",
                    "collection_id": "3e157b3a-8469-51db-90d9-52e7d896b49b",
                    "cached_at": "2025-01-11T10:30:00Z"
                },
                "similarity_score": 0.92,
                "cache_hit": True
            }
        }


class CacheMetrics(R2RSerializable):
    """Metrics about cache performance."""

    total_requests: int = Field(default=0, description="Total cache lookup requests")
    
    cache_hits: int = Field(default=0, description="Number of cache hits")
    
    cache_misses: int = Field(default=0, description="Number of cache misses")
    
    hit_rate: float = Field(default=0.0, description="Cache hit rate percentage")
    
    total_entries: int = Field(default=0, description="Total cache entries")
    
    expired_entries: int = Field(default=0, description="Number of expired entries")
    
    average_similarity: float = Field(
        default=0.0, description="Average similarity score for cache hits"
    )

    def calculate_hit_rate(self) -> None:
        """Calculate and update the hit rate."""
        if self.total_requests > 0:
            self.hit_rate = (self.cache_hits / self.total_requests) * 100
        else:
            self.hit_rate = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "total_requests": 1000,
                "cache_hits": 850,
                "cache_misses": 150,
                "hit_rate": 85.0,
                "total_entries": 500,
                "expired_entries": 25,
                "average_similarity": 0.91
            }
        }


# Phase 2: Cache Entry Management Models

class CacheEntryUpdate(R2RSerializable):
    """Model for updating cache entry content"""
    entry_id: str = Field(..., description="ID of the cache entry to update")
    generated_answer: Optional[str] = Field(None, description="New answer content")
    search_results: Optional[dict] = Field(None, description="Updated search results")
    citations: Optional[list] = Field(None, description="Updated citations")
    ttl_seconds: Optional[int] = Field(None, description="New TTL (None=no change, 0=never expire, >0=TTL in seconds)")

class CacheEntryBulkUpdate(R2RSerializable):
    """Model for bulk updating cache entries"""
    updates: list[CacheEntryUpdate] = Field(..., description="List of updates to apply")

class CacheEntryDetail(R2RSerializable):
    """Detailed view of a cache entry"""
    entry_id: str = Field(..., description="Unique ID of the cache entry")
    query: str = Field(..., description="Original query text")
    answer: str = Field(..., description="Cached answer")
    search_results: dict = Field(..., description="Associated search results")
    citations: list = Field(..., description="Associated citations")
    collection_id: str = Field(..., description="ID of the original collection")
    cached_at: datetime = Field(..., description="When entry was cached")
    ttl_seconds: int = Field(..., description="Time-to-live in seconds (0=never expires)")
    hit_count: int = Field(..., description="Number of times accessed")
    last_accessed: Optional[datetime] = Field(None, description="Last access time")
    is_expired: bool = Field(..., description="Whether entry has expired")
    expires_at: Optional[datetime] = Field(None, description="Expiration time if TTL is set")

class CacheEntriesResponse(R2RSerializable):
    """Response for listing cache entries"""
    entries: Union[list[str], list[CacheEntryDetail]] = Field(..., description="Cache entries in requested format")
    total_count: int = Field(..., description="Total number of entries")
    format: str = Field(..., description="Format of entries (plain or detailed)")

class CacheTTLUpdate(R2RSerializable):
    """Model for updating cache entry TTLs"""
    ttl_updates: dict[str, Optional[int]] = Field(..., description="Map of entry_id to new TTL")

class CacheEntryUpdateRequest(R2RSerializable):
    """Request model for updating a cache entry"""
    generated_answer: Optional[str] = Field(None, description="New answer content")
    search_results: Optional[dict] = Field(None, description="Updated search results")
    citations: Optional[list] = Field(None, description="Updated citations")
    ttl_seconds: Optional[int] = Field(None, description="New TTL")

class CacheDeleteRequest(R2RSerializable):
    """Request model for deleting cache entries"""
    entry_ids: list[str] = Field(..., description="List of entry IDs to delete") 