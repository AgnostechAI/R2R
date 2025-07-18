import logging
import textwrap
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

import psutil
from fastapi import Body, Depends, Path, Query

from core.base import R2RException
from core.base.api.models import (
    GenericMessageResponse,
    WrappedGenericMessageResponse,
    WrappedServerStatsResponse,
    WrappedSettingsResponse,
)
from shared.abstractions.cache import (
    CacheEntryUpdateRequest,
    CacheEntryBulkUpdate,
    CacheDeleteRequest,
    CacheTTLUpdate,
)

from ...abstractions import R2RProviders, R2RServices
from ...config import R2RConfig
from .base_router import BaseRouterV3


class SystemRouter(BaseRouterV3):
    def __init__(
        self,
        providers: R2RProviders,
        services: R2RServices,
        config: R2RConfig,
    ):
        logging.info("Initializing SystemRouter")
        super().__init__(providers, services, config)
        self.start_time = datetime.now(timezone.utc)

    def _setup_routes(self):
        @self.router.get(
            "/health",
            openapi_extra={
                "x-codeSamples": [
                    {
                        "lang": "Python",
                        "source": textwrap.dedent("""
                            from r2r import R2RClient

                            client = R2RClient()
                            # when using auth, do client.login(...)

                            result = client.system.health()
                        """),
                    },
                    {
                        "lang": "JavaScript",
                        "source": textwrap.dedent("""
                            const { r2rClient } = require("r2r-js");

                            const client = new r2rClient();

                            function main() {
                                const response = await client.system.health();
                            }

                            main();
                            """),
                    },
                    {
                        "lang": "cURL",
                        "source": textwrap.dedent("""
                            curl -X POST "https://api.example.com/v3/health"\\
                                 -H "Content-Type: application/json" \\
                                 -H "Authorization: Bearer YOUR_API_KEY" \\
                        """),
                    },
                ]
            },
        )
        @self.base_endpoint
        async def health_check() -> WrappedGenericMessageResponse:
            return GenericMessageResponse(message="ok")  # type: ignore

        @self.router.get(
            "/system/settings",
            dependencies=[Depends(self.rate_limit_dependency)],
            openapi_extra={
                "x-codeSamples": [
                    {
                        "lang": "Python",
                        "source": textwrap.dedent("""
                            from r2r import R2RClient

                            client = R2RClient()
                            # when using auth, do client.login(...)

                            result = client.system.settings()
                        """),
                    },
                    {
                        "lang": "JavaScript",
                        "source": textwrap.dedent("""
                            const { r2rClient } = require("r2r-js");

                            const client = new r2rClient();

                            function main() {
                                const response = await client.system.settings();
                            }

                            main();
                            """),
                    },
                    {
                        "lang": "cURL",
                        "source": textwrap.dedent("""
                            curl -X POST "https://api.example.com/v3/system/settings" \\
                                 -H "Content-Type: application/json" \\
                                 -H "Authorization: Bearer YOUR_API_KEY" \\
                        """),
                    },
                ]
            },
        )
        @self.base_endpoint
        async def app_settings(
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ) -> WrappedSettingsResponse:
            if not auth_user.is_superuser:
                raise R2RException(
                    "Only a superuser can call the `system/settings` endpoint.",
                    403,
                )
            return await self.services.management.app_settings()

        @self.router.get(
            "/system/status",
            dependencies=[Depends(self.rate_limit_dependency)],
            openapi_extra={
                "x-codeSamples": [
                    {
                        "lang": "Python",
                        "source": textwrap.dedent("""
                            from r2r import R2RClient

                            client = R2RClient()
                            # when using auth, do client.login(...)

                            result = client.system.status()
                        """),
                    },
                    {
                        "lang": "JavaScript",
                        "source": textwrap.dedent("""
                            const { r2rClient } = require("r2r-js");

                            const client = new r2rClient();

                            function main() {
                                const response = await client.system.status();
                            }

                            main();
                            """),
                    },
                    {
                        "lang": "cURL",
                        "source": textwrap.dedent("""
                            curl -X POST "https://api.example.com/v3/system/status" \\
                                 -H "Content-Type: application/json" \\
                                 -H "Authorization: Bearer YOUR_API_KEY" \\
                            """),
                    },
                ]
            },
        )
        @self.base_endpoint
        async def server_stats(
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ) -> WrappedServerStatsResponse:
            if not auth_user.is_superuser:
                raise R2RException(
                    "Only an authorized user can call the `system/status` endpoint.",
                    403,
                )
            return {  # type: ignore
                "start_time": self.start_time.isoformat(),
                "uptime_seconds": (
                    datetime.now(timezone.utc) - self.start_time
                ).total_seconds(),
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
            }

        @self.router.get(
            "/system/cache/analytics/{collection_id}",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="Get Cache Analytics",
        )
        @self.base_endpoint
        async def get_cache_analytics(
            collection_id: UUID,
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ):
            """Get analytics and performance metrics for semantic cache.
            
            Returns cache statistics including:
            - Total entries and hit counts
            - Most popular queries
            - Cache size and distribution
            - Performance metrics
            """
            if not auth_user.is_superuser:
                raise R2RException(
                    "Only a superuser can access cache analytics.",
                    403,
                )
            
            analytics = await self.services.ingestion.get_cache_analytics(collection_id)
            return {"analytics": analytics}

        @self.router.post(
            "/system/cache/invalidate/{collection_id}",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="Invalidate Cache Entries",
        )
        @self.base_endpoint
        async def invalidate_cache_entries(
            collection_id: UUID,
            query_patterns: Optional[list[str]] = Body(
                None,
                description="List of regex patterns to match queries for deletion"
            ),
            older_than_hours: Optional[int] = Body(
                None,
                description="Delete entries older than X hours"
            ),
            hit_count_threshold: Optional[int] = Body(
                None,
                description="Delete entries with hit count below threshold"
            ),
            delete_all: bool = Body(
                False,
                description="Delete all cache entries for this collection"
            ),
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ):
            """Invalidate cache entries based on various criteria.
            
            Supports deletion by:
            - Query patterns (regex matching)
            - Age (older than X hours)
            - Hit count threshold (unused entries)
            - Complete cache clearing
            """
            if not auth_user.is_superuser:
                raise R2RException(
                    "Only a superuser can invalidate cache entries.",
                    403,
                )
            
            result = await self.services.ingestion.invalidate_cache_entries(
                collection_id=collection_id,
                query_patterns=query_patterns,
                older_than_hours=older_than_hours,
                hit_count_threshold=hit_count_threshold,
                delete_all=delete_all
            )
            return {"invalidation_result": result}

        @self.router.post(
            "/system/cache/cleanup",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="Cleanup Expired Cache Entries",
        )
        @self.base_endpoint
        async def cleanup_expired_cache(
            collection_id: Optional[UUID] = Body(
                None,
                description="Specific collection to clean, or None for all collections"
            ),
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ):
            """Clean up expired cache entries based on TTL.
            
            Removes entries that have exceeded their time-to-live (TTL).
            If no collection_id provided, cleans all collections.
            """
            if not auth_user.is_superuser:
                raise R2RException(
                    "Only a superuser can cleanup cache entries.",
                    403,
                )
            
            result = await self.services.ingestion.cleanup_expired_cache_entries(
                collection_id=collection_id
            )
            return {"cleanup_result": result}

        # Phase 2: Cache Entry Management Endpoints

        @self.router.get(
            "/system/cache/entries/{collection_id}",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="Get Cache Entries",
        )
        @self.base_endpoint
        async def get_cache_entries(
            collection_id: UUID = Path(..., description="The collection ID"),
            format: str = Query("plain", description="Output format: 'plain' or 'detailed'"),
            include_expired: bool = Query(False, description="Include expired entries"),
            offset: int = Query(0, ge=0, description="Pagination offset"),
            limit: int = Query(100, ge=1, le=1000, description="Number of entries to return"),
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ):
            """
            Get cache entries for a collection in plain text or detailed format.
            
            Only superusers can access cache entries.
            """
            if not auth_user.is_superuser:
                raise R2RException(
                    "Only a superuser can access cache entries.",
                    403,
                )
            
            result = await self.services.ingestion.get_cache_entries(
                collection_id=collection_id,
                include_expired=include_expired,
                format=format,
                offset=offset,
                limit=limit
            )
            
            return {"results": result}

        @self.router.get(
            "/system/cache/entries/{collection_id}/{entry_id}",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="Get Cache Entry Details",
        )
        @self.base_endpoint
        async def get_cache_entry_details(
            collection_id: UUID = Path(..., description="The collection ID"),
            entry_id: str = Path(..., description="The cache entry ID"),
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ):
            """
            Get detailed information for a specific cache entry.
            
            Only superusers can access cache entry details.
            """
            if not auth_user.is_superuser:
                raise R2RException(
                    "Only a superuser can access cache entry details.",
                    403,
                )
            
            entry = await self.services.ingestion.get_cache_entry_details(
                collection_id=collection_id,
                entry_id=entry_id
            )
            
            return {"results": entry}

        @self.router.put(
            "/system/cache/entries/{collection_id}/{entry_id}",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="Update Cache Entry",
        )
        @self.base_endpoint
        async def update_cache_entry(
            collection_id: UUID = Path(..., description="The collection ID"),
            entry_id: str = Path(..., description="The cache entry ID"),
            request: CacheEntryUpdateRequest = Body(...),
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ):
            """
            Update an existing cache entry's content.
            
            Note: The query itself cannot be modified.
            Only superusers can update cache entries.
            """
            if not auth_user.is_superuser:
                raise R2RException(
                    "Only a superuser can update cache entries.",
                    403,
                )
            
            success = await self.services.ingestion.update_cache_entry(
                collection_id=collection_id,
                entry_id=entry_id,
                answer=request.generated_answer,
                search_results=request.search_results,
                citations=request.citations,
                ttl_seconds=request.ttl_seconds
            )
            
            return {"results": {"success": success}}

        @self.router.put(
            "/system/cache/entries/{collection_id}/bulk",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="Bulk Update Cache Entries",
        )
        @self.base_endpoint
        async def bulk_update_cache_entries(
            collection_id: UUID = Path(..., description="The collection ID"),
            request: CacheEntryBulkUpdate = Body(...),
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ):
            """
            Bulk update multiple cache entries.
            
            Only superusers can update cache entries.
            """
            if not auth_user.is_superuser:
                raise R2RException(
                    "Only a superuser can update cache entries.",
                    403,
                )
            
            result = await self.services.ingestion.bulk_update_cache_entries(
                collection_id=collection_id,
                updates=request.updates
            )
            
            return {"results": result}

        @self.router.delete(
            "/system/cache/entries/{collection_id}",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="Delete Cache Entries",
        )
        @self.base_endpoint
        async def delete_cache_entries(
            collection_id: UUID = Path(..., description="The collection ID"),
            request: CacheDeleteRequest = Body(...),
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ):
            """
            Delete specific cache entries.
            
            Only superusers can delete cache entries.
            """
            if not auth_user.is_superuser:
                raise R2RException(
                    "Only a superuser can delete cache entries.",
                    403,
                )
            
            result = await self.services.ingestion.delete_cache_entries(
                collection_id=collection_id,
                entry_ids=request.entry_ids
            )
            
            return {"results": result}

        @self.router.put(
            "/system/cache/entries/{collection_id}/ttl",
            dependencies=[Depends(self.rate_limit_dependency)],
            summary="Update Cache TTLs",
        )
        @self.base_endpoint
        async def update_cache_ttls(
            collection_id: UUID = Path(..., description="The collection ID"),
            request: CacheTTLUpdate = Body(...),
            auth_user=Depends(self.providers.auth.auth_wrapper()),
        ):
            """
            Update TTLs for multiple cache entries.
            
            Set TTL to 0 or None to make entries never expire.
            Only superusers can update cache TTLs.
            """
            if not auth_user.is_superuser:
                raise R2RException(
                    "Only a superuser can update cache TTLs.",
                    403,
                )
            
            result = await self.services.ingestion.update_cache_ttls(
                collection_id=collection_id,
                ttl_updates=request.ttl_updates
            )
            
            return {"results": result}
