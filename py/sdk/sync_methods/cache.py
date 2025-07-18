from typing import Optional, Union
from uuid import UUID

from shared.api.models import (
    WrappedCacheEntriesResponse,
    WrappedCacheEntryDetailResponse,
    WrappedBooleanResponse,
)
from shared.abstractions.cache import (
    CacheEntryUpdateRequest,
    CacheEntryBulkUpdate,
    CacheDeleteRequest,
    CacheTTLUpdate,
)


class CacheSDK:
    def __init__(self, client):
        self.client = client

    def get_cache_entries(
        self,
        collection_id: Union[str, UUID],
        format: str = "plain",
        include_expired: bool = False,
        offset: int = 0,
        limit: int = 100
    ) -> WrappedCacheEntriesResponse:
        """Get cache entries for a collection.

        Args:
            collection_id: The collection ID
            format: Output format - 'plain' or 'detailed'
            include_expired: Whether to include expired entries
            offset: Pagination offset
            limit: Number of entries to return (max 1000)

        Returns:
            WrappedCacheEntriesResponse: Cache entries in requested format
        """
        params = {
            "format": format,
            "include_expired": include_expired,
            "offset": offset,
            "limit": limit
        }
        
        response_dict = self.client._make_request(
            "GET",
            f"system/cache/entries/{collection_id}",
            params=params,
            version="v3"
        )
        
        return WrappedCacheEntriesResponse(**response_dict)

    def get_cache_entry_details(
        self,
        collection_id: Union[str, UUID],
        entry_id: str
    ) -> WrappedCacheEntryDetailResponse:
        """Get detailed information for a specific cache entry.

        Args:
            collection_id: The collection ID
            entry_id: The cache entry ID

        Returns:
            WrappedCacheEntryDetailResponse: Detailed cache entry information
        """
        response_dict = self.client._make_request(
            "GET",
            f"system/cache/entries/{collection_id}/{entry_id}",
            version="v3"
        )
        
        return WrappedCacheEntryDetailResponse(**response_dict)

    def update_cache_entry(
        self,
        collection_id: Union[str, UUID],
        entry_id: str,
        answer: Optional[str] = None,
        search_results: Optional[dict] = None,
        citations: Optional[list] = None,
        ttl_seconds: Optional[int] = None
    ) -> WrappedBooleanResponse:
        """Update an existing cache entry's content.

        Args:
            collection_id: The collection ID
            entry_id: The cache entry ID
            answer: New answer content
            search_results: Updated search results
            citations: Updated citations
            ttl_seconds: New TTL (None=no change, 0=never expire, >0=TTL)

        Returns:
            WrappedBooleanResponse: Success status
        """
        request = CacheEntryUpdateRequest(
            generated_answer=answer,
            search_results=search_results,
            citations=citations,
            ttl_seconds=ttl_seconds
        )
        
        response_dict = self.client._make_request(
            "PUT",
            f"system/cache/entries/{collection_id}/{entry_id}",
            json=request.model_dump(exclude_none=True),
            version="v3"
        )
        
        return WrappedBooleanResponse(**response_dict)

    def bulk_update_cache_entries(
        self,
        collection_id: Union[str, UUID],
        updates: list[dict]
    ) -> dict:
        """Bulk update multiple cache entries.

        Args:
            collection_id: The collection ID
            updates: List of updates, each containing:
                - entry_id: Cache entry ID
                - generated_answer: Optional new answer
                - search_results: Optional new search results
                - citations: Optional new citations
                - ttl_seconds: Optional new TTL

        Returns:
            dict: Summary of results {"succeeded": count, "failed": count, "errors": [...]}
        """
        request = CacheEntryBulkUpdate(updates=updates)
        
        response_dict = self.client._make_request(
            "PUT",
            f"system/cache/entries/{collection_id}/bulk",
            json=request.model_dump(),
            version="v3"
        )
        
        return response_dict["results"]

    def delete_cache_entries(
        self,
        collection_id: Union[str, UUID],
        entry_ids: list[str]
    ) -> dict:
        """Delete specific cache entries.

        Args:
            collection_id: The collection ID
            entry_ids: List of entry IDs to delete

        Returns:
            dict: Summary {"deleted": count, "failed": count, "errors": [...]}
        """
        request = CacheDeleteRequest(entry_ids=entry_ids)
        
        response_dict = self.client._make_request(
            "DELETE",
            f"system/cache/entries/{collection_id}",
            json=request.model_dump(),
            version="v3"
        )
        
        return response_dict["results"]

    def update_cache_ttls(
        self,
        collection_id: Union[str, UUID],
        ttl_updates: dict[str, Optional[int]]
    ) -> dict:
        """Update TTLs for multiple cache entries.

        Args:
            collection_id: The collection ID
            ttl_updates: Map of entry_id to new TTL (None or 0 = never expire)

        Returns:
            dict: Summary {"updated": count, "failed": count, "errors": [...]}
        """
        request = CacheTTLUpdate(ttl_updates=ttl_updates)
        
        response_dict = self.client._make_request(
            "PUT",
            f"system/cache/entries/{collection_id}/ttl",
            json=request.model_dump(),
            version="v3"
        )
        
        return response_dict["results"]