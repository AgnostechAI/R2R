import asyncio
import json
import logging
from datetime import datetime, timedelta
import re
from typing import Any, AsyncGenerator, Optional, Sequence
from uuid import UUID

from fastapi import HTTPException

from core.base import (
    Document,
    DocumentChunk,
    DocumentResponse,
    DocumentType,
    GenerationConfig,
    IngestionStatus,
    R2RException,
    RawChunk,
    UnprocessedChunk,
    Vector,
    VectorEntry,
    VectorType,
    generate_id,
    SearchSettings,
)
from core.base.abstractions import (
    ChunkEnrichmentSettings,
    IndexMeasure,
    IndexMethod,
    R2RDocumentProcessingError,
    VectorTableName,
)
from core.base.api.models import User
from shared.abstractions import PDFParsingError, PopplerNotFoundError
from shared.abstractions.cache import (
    CacheEntryDetail,
    CacheEntriesResponse,
    CacheEntryUpdate,
)

from ..abstractions import R2RProviders
from ..config import R2RConfig

logger = logging.getLogger()
STARTING_VERSION = "v0"


class IngestionService:
    """A refactored IngestionService that inlines all pipe logic for parsing,
    embedding, and vector storage directly in its methods."""

    def __init__(
        self,
        config: R2RConfig,
        providers: R2RProviders,
    ) -> None:
        self.config = config
        self.providers = providers

    async def ingest_file_ingress(
        self,
        file_data: dict,
        user: User,
        document_id: UUID,
        size_in_bytes,
        metadata: Optional[dict] = None,
        version: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        """Pre-ingests a file by creating or validating the DocumentResponse
        entry.

        Does not actually parse/ingest the content. (See parse_file() for that
        step.)
        """
        try:
            if not file_data:
                raise R2RException(
                    status_code=400, message="No files provided for ingestion."
                )
            if not file_data.get("filename"):
                raise R2RException(
                    status_code=400, message="File name not provided."
                )

            metadata = metadata or {}
            version = version or STARTING_VERSION

            document_info = self.create_document_info_from_file(
                document_id,
                user,
                file_data["filename"],
                metadata,
                version,
                size_in_bytes,
            )

            existing_document_info = (
                await self.providers.database.documents_handler.get_documents_overview(
                    offset=0,
                    limit=100,
                    filter_user_ids=[user.id],
                    filter_document_ids=[document_id],
                )
            )["results"]

            # Validate ingestion status for re-ingestion
            if len(existing_document_info) > 0:
                existing_doc = existing_document_info[0]
                if existing_doc.ingestion_status == IngestionStatus.SUCCESS:
                    raise R2RException(
                        status_code=409,
                        message=(
                            f"Document {document_id} already exists. "
                            "Submit a DELETE request to `/documents/{document_id}` "
                            "to delete this document and allow for re-ingestion."
                        ),
                    )
                elif existing_doc.ingestion_status != IngestionStatus.FAILED:
                    raise R2RException(
                        status_code=409,
                        message=(
                            f"Document {document_id} is currently ingesting "
                            f"with status {existing_doc.ingestion_status}."
                        ),
                    )

            # Set to PARSING until we actually parse
            document_info.ingestion_status = IngestionStatus.PARSING
            await self.providers.database.documents_handler.upsert_documents_overview(
                document_info
            )

            return {
                "info": document_info,
            }
        except R2RException as e:
            logger.error(f"R2RException in ingest_file_ingress: {str(e)}")
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error during ingestion: {str(e)}"
            ) from e

    def create_document_info_from_file(
        self,
        document_id: UUID,
        user: User,
        file_name: str,
        metadata: dict,
        version: str,
        size_in_bytes: int,
    ) -> DocumentResponse:
        file_extension = (
            file_name.split(".")[-1].lower() if file_name != "N/A" else "txt"
        )
        if file_extension.upper() not in DocumentType.__members__:
            raise R2RException(
                status_code=415,
                message=f"'{file_extension}' is not a valid DocumentType.",
            )

        metadata = metadata or {}
        metadata["version"] = version

        return DocumentResponse(
            id=document_id,
            owner_id=user.id,
            collection_ids=metadata.get("collection_ids", []),
            document_type=DocumentType[file_extension.upper()],
            title=(
                metadata.get("title", file_name.split("/")[-1])
                if file_name != "N/A"
                else "N/A"
            ),
            metadata=metadata,
            version=version,
            size_in_bytes=size_in_bytes,
            ingestion_status=IngestionStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    def _create_document_info_from_chunks(
        self,
        document_id: UUID,
        user: User,
        chunks: list[RawChunk],
        metadata: dict,
        version: str,
    ) -> DocumentResponse:
        metadata = metadata or {}
        metadata["version"] = version

        return DocumentResponse(
            id=document_id,
            owner_id=user.id,
            collection_ids=metadata.get("collection_ids", []),
            document_type=DocumentType.TXT,
            title=metadata.get("title", f"Ingested Chunks - {document_id}"),
            metadata=metadata,
            version=version,
            size_in_bytes=sum(
                len(chunk.text.encode("utf-8")) for chunk in chunks
            ),
            ingestion_status=IngestionStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    async def parse_file(
        self,
        document_info: DocumentResponse,
        ingestion_config: dict | None,
    ) -> AsyncGenerator[DocumentChunk, None]:
        """Reads the file content from the DB, calls the ingestion
        provider to parse, and yields DocumentChunk objects."""
        version = document_info.version or "v0"
        ingestion_config_override = ingestion_config or {}

        # The ingestion config might specify a different provider, etc.
        override_provider = ingestion_config_override.pop("provider", None)
        if (
            override_provider
            and override_provider != self.providers.ingestion.config.provider
        ):
            raise ValueError(
                f"Provider '{override_provider}' does not match ingestion provider "
                f"'{self.providers.ingestion.config.provider}'."
            )

        try:
            # Pull file from DB
            retrieved = (
                await self.providers.database.files_handler.retrieve_file(
                    document_info.id
                )
            )
            if not retrieved:
                # No file found in the DB, can't parse
                raise R2RDocumentProcessingError(
                    document_id=document_info.id,
                    error_message="No file content found in DB for this document.",
                )

            file_name, file_wrapper, file_size = retrieved

            # Read the content
            with file_wrapper as file_content_stream:
                file_content = file_content_stream.read()

            # Build a barebones Document object
            doc = Document(
                id=document_info.id,
                collection_ids=document_info.collection_ids,
                owner_id=document_info.owner_id,
                metadata={
                    "document_type": document_info.document_type.value,
                    **document_info.metadata,
                },
                document_type=document_info.document_type,
            )

            # Delegate to the ingestion provider to parse
            async for extraction in self.providers.ingestion.parse(
                file_content,  # raw bytes
                doc,
                ingestion_config_override,
            ):
                # Adjust chunk ID to incorporate version
                # or any other needed transformations
                extraction.id = generate_id(f"{extraction.id}_{version}")
                extraction.metadata["version"] = version
                yield extraction

        except (PopplerNotFoundError, PDFParsingError) as e:
            raise R2RDocumentProcessingError(
                error_message=e.message,
                document_id=document_info.id,
                status_code=e.status_code,
            ) from None
        except Exception as e:
            if isinstance(e, R2RException):
                raise
            raise R2RDocumentProcessingError(
                document_id=document_info.id,
                error_message=f"Error parsing document: {str(e)}",
            ) from e

    async def augment_document_info(
        self,
        document_info: DocumentResponse,
        chunked_documents: list[dict],
    ) -> None:
        if not self.config.ingestion.skip_document_summary:
            document = f"Document Title: {document_info.title}\n"
            if document_info.metadata != {}:
                document += f"Document Metadata: {json.dumps(document_info.metadata)}\n"

            document += "Document Text:\n"
            for chunk in chunked_documents[
                : self.config.ingestion.chunks_for_document_summary
            ]:
                document += chunk["data"]

            messages = await self.providers.database.prompts_handler.get_message_payload(
                system_prompt_name=self.config.ingestion.document_summary_system_prompt,
                task_prompt_name=self.config.ingestion.document_summary_task_prompt,
                task_inputs={
                    "document": document[
                        : self.config.ingestion.document_summary_max_length
                    ]
                },
            )

            response = await self.providers.llm.aget_completion(
                messages=messages,
                generation_config=GenerationConfig(
                    model=self.config.ingestion.document_summary_model
                    or self.config.app.fast_llm
                ),
            )

            document_info.summary = response.choices[0].message.content  # type: ignore

            if not document_info.summary:
                raise ValueError("Expected a generated response.")

            embedding = await self.providers.embedding.async_get_embedding(
                text=document_info.summary,
            )
            document_info.summary_embedding = embedding
        return

    async def embed_document(
        self,
        chunked_documents: list[dict],
        embedding_batch_size: int = 8,
    ) -> AsyncGenerator[VectorEntry, None]:
        """Inline replacement for the old embedding_pipe.run(...).

        Batches the embedding calls and yields VectorEntry objects.
        """
        if not chunked_documents:
            return

        concurrency_limit = (
            self.providers.embedding.config.concurrent_request_limit or 5
        )
        extraction_batch: list[DocumentChunk] = []
        tasks: set[asyncio.Task] = set()

        async def process_batch(
            batch: list[DocumentChunk],
        ) -> list[VectorEntry]:
            # All text from the batch
            texts = [
                (
                    ex.data.decode("utf-8")
                    if isinstance(ex.data, bytes)
                    else ex.data
                )
                for ex in batch
            ]
            # Retrieve embeddings in bulk
            vectors = await self.providers.embedding.async_get_embeddings(
                texts,  # list of strings
            )
            # Zip them back together
            results = []
            for raw_vector, extraction in zip(vectors, batch, strict=False):
                results.append(
                    VectorEntry(
                        id=extraction.id,
                        document_id=extraction.document_id,
                        owner_id=extraction.owner_id,
                        collection_ids=extraction.collection_ids,
                        vector=Vector(data=raw_vector, type=VectorType.FIXED),
                        text=(
                            extraction.data.decode("utf-8")
                            if isinstance(extraction.data, bytes)
                            else str(extraction.data)
                        ),
                        metadata={**extraction.metadata},
                    )
                )
            return results

        async def run_process_batch(batch: list[DocumentChunk]):
            return await process_batch(batch)

        # Convert each chunk dict to a DocumentChunk
        for chunk_dict in chunked_documents:
            extraction = DocumentChunk.from_dict(chunk_dict)
            extraction_batch.append(extraction)

            # If we hit a batch threshold, spawn a task
            if len(extraction_batch) >= embedding_batch_size:
                tasks.add(
                    asyncio.create_task(run_process_batch(extraction_batch))
                )
                extraction_batch = []

            # If tasks are at concurrency limit, wait for the first to finish
            while len(tasks) >= concurrency_limit:
                done, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    for vector_entry in await t:
                        yield vector_entry

        # Handle any leftover items
        if extraction_batch:
            tasks.add(asyncio.create_task(run_process_batch(extraction_batch)))

        # Gather remaining tasks
        for future_task in asyncio.as_completed(tasks):
            for vector_entry in await future_task:
                yield vector_entry

    async def store_embeddings(
        self,
        embeddings: Sequence[dict | VectorEntry],
        storage_batch_size: int = 128,
    ) -> AsyncGenerator[str, None]:
        """Inline replacement for the old vector_storage_pipe.run(...).

        Batches up the vector entries, enforces usage limits, stores them, and
        yields a success/error string (or you could yield a StorageResult).
        """
        if not embeddings:
            return

        vector_entries: list[VectorEntry] = []
        for item in embeddings:
            if isinstance(item, VectorEntry):
                vector_entries.append(item)
            else:
                vector_entries.append(VectorEntry.from_dict(item))

        vector_batch: list[VectorEntry] = []
        document_counts: dict[UUID, int] = {}

        # We'll track usage from the first user we see; if your scenario allows
        # multiple user owners in a single ingestion, you'd need to refine usage checks.
        current_usage = None
        user_id_for_usage_check: UUID | None = None

        count = 0

        for msg in vector_entries:
            # If we haven't set usage yet, do so on the first chunk
            if current_usage is None:
                user_id_for_usage_check = msg.owner_id
                usage_data = (
                    await self.providers.database.chunks_handler.list_chunks(
                        limit=1,
                        offset=0,
                        filters={"owner_id": msg.owner_id},
                    )
                )
                current_usage = usage_data["total_entries"]

            # Figure out the user's limit
            user = await self.providers.database.users_handler.get_user_by_id(
                msg.owner_id
            )
            max_chunks = (
                self.providers.database.config.app.default_max_chunks_per_user
                if self.providers.database.config.app
                else 1e10
            )
            if user.limits_overrides and "max_chunks" in user.limits_overrides:
                max_chunks = user.limits_overrides["max_chunks"]

            # Add to our local batch
            vector_batch.append(msg)
            document_counts[msg.document_id] = (
                document_counts.get(msg.document_id, 0) + 1
            )
            count += 1

            # Check usage
            if (
                current_usage is not None
                and (current_usage + len(vector_batch) + count) > max_chunks
            ):
                error_message = f"User {msg.owner_id} has exceeded the maximum number of allowed chunks: {max_chunks}"
                logger.error(error_message)
                yield error_message
                continue

            # Once we hit our batch size, store them
            if len(vector_batch) >= storage_batch_size:
                try:
                    await (
                        self.providers.database.chunks_handler.upsert_entries(
                            vector_batch
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to store vector batch: {e}")
                    yield f"Error: {e}"
                vector_batch.clear()

        # Store any leftover items
        if vector_batch:
            try:
                await self.providers.database.chunks_handler.upsert_entries(
                    vector_batch
                )
            except Exception as e:
                logger.error(f"Failed to store final vector batch: {e}")
                yield f"Error: {e}"

        # Summaries
        for doc_id, cnt in document_counts.items():
            info_msg = f"Successful ingestion for document_id: {doc_id}, with vector count: {cnt}"
            logger.info(info_msg)
            yield info_msg

    async def finalize_ingestion(
        self, document_info: DocumentResponse
    ) -> None:
        """Called at the end of a successful ingestion pipeline to set the
        document status to SUCCESS or similar final steps."""

        async def empty_generator():
            yield document_info

        await self.update_document_status(
            document_info, IngestionStatus.SUCCESS
        )
        return empty_generator()

    async def update_document_status(
        self,
        document_info: DocumentResponse,
        status: IngestionStatus,
        metadata: Optional[dict] = None,
    ) -> None:
        document_info.ingestion_status = status
        if metadata:
            document_info.metadata = {**document_info.metadata, **metadata}
        await self._update_document_status_in_db(document_info)

    async def _update_document_status_in_db(
        self, document_info: DocumentResponse
    ):
        try:
            await self.providers.database.documents_handler.upsert_documents_overview(
                document_info
            )
        except Exception as e:
            logger.error(
                f"Failed to update document status: {document_info.id}. Error: {str(e)}"
            )

    async def ingest_chunks_ingress(
        self,
        document_id: UUID,
        metadata: Optional[dict],
        chunks: list[RawChunk],
        user: User,
        *args: Any,
        **kwargs: Any,
    ) -> DocumentResponse:
        """Directly ingest user-provided text chunks (rather than from a
        file)."""
        if not chunks:
            raise R2RException(
                status_code=400, message="No chunks provided for ingestion."
            )
        metadata = metadata or {}
        version = STARTING_VERSION

        document_info = self._create_document_info_from_chunks(
            document_id,
            user,
            chunks,
            metadata,
            version,
        )

        existing_document_info = (
            await self.providers.database.documents_handler.get_documents_overview(
                offset=0,
                limit=100,
                filter_user_ids=[user.id],
                filter_document_ids=[document_id],
            )
        )["results"]
        if len(existing_document_info) > 0:
            existing_doc = existing_document_info[0]
            if existing_doc.ingestion_status != IngestionStatus.FAILED:
                raise R2RException(
                    status_code=409,
                    message=(
                        f"Document {document_id} was already ingested "
                        "and is not in a failed state."
                    ),
                )

        await self.providers.database.documents_handler.upsert_documents_overview(
            document_info
        )
        return document_info

    async def update_chunk_ingress(
        self,
        document_id: UUID,
        chunk_id: UUID,
        text: str,
        user: User,
        metadata: Optional[dict] = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        """Update an individual chunk's text and metadata, re-embed, and re-
        store it."""
        # Verify chunk exists and user has access
        existing_chunks = (
            await self.providers.database.chunks_handler.list_document_chunks(
                document_id=document_id,
                offset=0,
                limit=1,
            )
        )
        if not existing_chunks["results"]:
            raise R2RException(
                status_code=404,
                message=f"Chunk with chunk_id {chunk_id} not found.",
            )

        existing_chunk = (
            await self.providers.database.chunks_handler.get_chunk(chunk_id)
        )
        if not existing_chunk:
            raise R2RException(
                status_code=404,
                message=f"Chunk with id {chunk_id} not found",
            )

        if (
            str(existing_chunk["owner_id"]) != str(user.id)
            and not user.is_superuser
        ):
            raise R2RException(
                status_code=403,
                message="You don't have permission to modify this chunk.",
            )

        # Merge metadata
        merged_metadata = {**existing_chunk["metadata"]}
        if metadata is not None:
            merged_metadata |= metadata

        # Create updated chunk
        extraction_data = {
            "id": chunk_id,
            "document_id": document_id,
            "collection_ids": kwargs.get(
                "collection_ids", existing_chunk["collection_ids"]
            ),
            "owner_id": existing_chunk["owner_id"],
            "data": text or existing_chunk["text"],
            "metadata": merged_metadata,
        }
        extraction = DocumentChunk(**extraction_data).model_dump()

        # Re-embed
        embeddings_generator = self.embed_document(
            [extraction], embedding_batch_size=1
        )
        embeddings = []
        async for embedding in embeddings_generator:
            embeddings.append(embedding)

        # Re-store
        store_gen = self.store_embeddings(embeddings, storage_batch_size=1)
        async for _ in store_gen:
            pass

        return extraction

    async def _get_enriched_chunk_text(
        self,
        chunk_idx: int,
        chunk: dict,
        document_id: UUID,
        document_summary: str | None,
        chunk_enrichment_settings: ChunkEnrichmentSettings,
        list_document_chunks: list[dict],
    ) -> VectorEntry:
        """Helper for chunk_enrichment.

        Leverages an LLM to rewrite or expand chunk text, then re-embeds it.
        """
        preceding_chunks = [
            list_document_chunks[idx]["text"]
            for idx in range(
                max(0, chunk_idx - chunk_enrichment_settings.n_chunks),
                chunk_idx,
            )
        ]
        succeeding_chunks = [
            list_document_chunks[idx]["text"]
            for idx in range(
                chunk_idx + 1,
                min(
                    len(list_document_chunks),
                    chunk_idx + chunk_enrichment_settings.n_chunks + 1,
                ),
            )
        ]
        try:
            # Obtain the updated text from the LLM
            updated_chunk_text = (
                (
                    await self.providers.llm.aget_completion(
                        messages=await self.providers.database.prompts_handler.get_message_payload(
                            task_prompt_name=chunk_enrichment_settings.chunk_enrichment_prompt,
                            task_inputs={
                                "document_summary": document_summary or "None",
                                "chunk": chunk["text"],
                                "preceding_chunks": (
                                    "\n".join(preceding_chunks)
                                    if preceding_chunks
                                    else "None"
                                ),
                                "succeeding_chunks": (
                                    "\n".join(succeeding_chunks)
                                    if succeeding_chunks
                                    else "None"
                                ),
                                "chunk_size": self.config.ingestion.chunk_size
                                or 1024,
                            },
                        ),
                        generation_config=chunk_enrichment_settings.generation_config
                        or GenerationConfig(model=self.config.app.fast_llm),
                    )
                )
                .choices[0]
                .message.content
            )
        except Exception:
            updated_chunk_text = chunk["text"]
            chunk["metadata"]["chunk_enrichment_status"] = "failed"
        else:
            chunk["metadata"]["chunk_enrichment_status"] = (
                "success" if updated_chunk_text else "failed"
            )

        if not updated_chunk_text or not isinstance(updated_chunk_text, str):
            updated_chunk_text = str(chunk["text"])
            chunk["metadata"]["chunk_enrichment_status"] = "failed"

        # Re-embed
        data = await self.providers.embedding.async_get_embedding(
            updated_chunk_text
        )
        chunk["metadata"]["original_text"] = chunk["text"]

        return VectorEntry(
            id=generate_id(str(chunk["id"])),
            vector=Vector(data=data, type=VectorType.FIXED, length=len(data)),
            document_id=document_id,
            owner_id=chunk["owner_id"],
            collection_ids=chunk["collection_ids"],
            text=updated_chunk_text,
            metadata=chunk["metadata"],
        )

    async def chunk_enrichment(
        self,
        document_id: UUID,
        document_summary: str | None,
        chunk_enrichment_settings: ChunkEnrichmentSettings,
    ) -> int:
        """Example function that modifies chunk text via an LLM then re-embeds
        and re-stores all chunks for the given document."""
        list_document_chunks = (
            await self.providers.database.chunks_handler.list_document_chunks(
                document_id=document_id,
                offset=0,
                limit=-1,
            )
        )["results"]

        new_vector_entries: list[VectorEntry] = []
        tasks = []
        total_completed = 0

        for chunk_idx, chunk in enumerate(list_document_chunks):
            tasks.append(
                self._get_enriched_chunk_text(
                    chunk_idx=chunk_idx,
                    chunk=chunk,
                    document_id=document_id,
                    document_summary=document_summary,
                    chunk_enrichment_settings=chunk_enrichment_settings,
                    list_document_chunks=list_document_chunks,
                )
            )

            # Process in batches of e.g. 128 concurrency
            if len(tasks) == 128:
                new_vector_entries.extend(await asyncio.gather(*tasks))
                total_completed += 128
                logger.info(
                    f"Completed {total_completed} out of {len(list_document_chunks)} chunks for document {document_id}"
                )
                tasks = []

        # Finish any remaining tasks
        new_vector_entries.extend(await asyncio.gather(*tasks))
        logger.info(
            f"Completed enrichment of {len(list_document_chunks)} chunks for document {document_id}"
        )

        # Delete old chunks from vector db
        await self.providers.database.chunks_handler.delete(
            filters={"document_id": document_id}
        )

        # Insert the newly enriched entries
        await self.providers.database.chunks_handler.upsert_entries(
            new_vector_entries
        )
        return len(new_vector_entries)

    async def store_cache_entry(
        self,
        query: str,
        response: dict,
        collection_id: UUID,
        owner_id: UUID,
        cache_settings: Optional[dict] = None,
    ) -> str:
        """Store a RAG query-response pair in the semantic cache.
        
        Args:
            query: The original query text
            response: The RAG response containing generated_answer, search_results, etc.
            collection_id: ID of the original collection
            owner_id: ID of the user who made the query
            cache_settings: Optional cache configuration settings
            
        Returns:
            str: Success message with cache entry ID
        """
        
        try:
            # Get cache collection ID (original collection ID + "_cache")
            cache_collection_id = await self._get_cache_collection_id(collection_id)
            
            # Generate embedding for the query
            query_embedding = await self.providers.embedding.async_get_embedding(query)
            
            # Create cache entry ID
            cache_entry_id = generate_id(f"cache_{query}_{collection_id}")
            
            # Create cache document ID  
            cache_document_id = generate_id(f"cache_doc_{query}_{collection_id}")
            
            # Extract cache settings
            settings = cache_settings or {}
            ttl_seconds = settings.get("ttl_seconds", 0)  # 0 = indefinite storage by default
            
            # Create cache metadata
            cache_metadata = {
                "type": "semantic_cache_entry",
                "original_query": query,
                "generated_answer": response.get("generated_answer", ""),
                "search_results": json.dumps(response.get("search_results", {})),
                "citations": json.dumps(response.get("citations", [])),
                "cached_at": datetime.now().isoformat(),
                "cache_ttl": ttl_seconds,
                "collection_id": str(collection_id),
                "hit_count": 0,
                "model_used": response.get("metadata", {}).get("model", "unknown"),
                "response_tokens": len(response.get("generated_answer", "").split()),
            }
            
            # Create vector entry for the cache
            cache_vector_entry = VectorEntry(
                id=cache_entry_id,
                document_id=cache_document_id,
                owner_id=owner_id,
                collection_ids=[cache_collection_id],
                vector=Vector(
                    data=query_embedding,
                    type=VectorType.FIXED,
                    length=len(query_embedding)
                ),
                text=query,  # Store the query as the searchable text
                metadata=cache_metadata,
            )
            
            # Store the cache entry
            await self.providers.database.chunks_handler.upsert_entries([cache_vector_entry])
            
            logger.info(f"Stored cache entry {cache_entry_id} for collection {collection_id}")
            return f"Successfully cached entry {cache_entry_id}"
            
        except Exception as e:
            logger.error(f"Error storing cache entry: {e}")
            return f"Error storing cache entry: {e}"

    async def _get_cache_collection_id(self, collection_id: UUID) -> UUID:
        """Get the cache collection ID for a given collection.
        
        This method handles both regular collection IDs and cache collection IDs:
        - If given a regular collection ID, finds the associated cache collection
        - If given a cache collection ID (name ends with "_cache"), returns it directly
        
        Args:
            collection_id: Either a regular collection ID or a cache collection ID
            
        Returns:
            UUID: The cache collection ID
        """
        try:
            # Get collection info
            collections_overview = await self.providers.database.collections_handler.get_collections_overview(
                offset=0,
                limit=100,
                filter_collection_ids=[collection_id]
            )
            
            if not collections_overview["results"]:
                raise ValueError(f"Collection {collection_id} not found")
                
            collection = collections_overview["results"][0]
            collection_name = collection.name
            
            # Check if this is already a cache collection
            if collection_name.endswith("_cache"):
                logger.info(f"Collection {collection_id} is already a cache collection")
                return collection_id
            
            # Look for the cache collection (name + "_cache")
            cache_name = f"{collection_name}_cache"
            
            # Search for cache collection by name
            all_collections = await self.providers.database.collections_handler.get_collections_overview(
                offset=0,
                limit=1000,  # Get all collections to search by name
                filter_user_ids=[collection.owner_id]
            )
            
            for coll in all_collections["results"]:
                if coll.name == cache_name:
                    return coll.id
                    
            raise ValueError(f"Cache collection '{cache_name}' not found for collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error getting cache collection ID: {e}")
            raise

    async def _is_cache_collection(self, collection_id: UUID) -> bool:
        """Check if a collection is a cache collection.
        
        Args:
            collection_id: The collection ID to check
            
        Returns:
            bool: True if it's a cache collection, False otherwise
        """
        try:
            collections_overview = await self.providers.database.collections_handler.get_collections_overview(
                offset=0,
                limit=1,
                filter_collection_ids=[collection_id]
            )
            
            if collections_overview["results"]:
                return collections_overview["results"][0].name.endswith("_cache")
            return False
        except Exception:
            return False

    async def search_cache_entries(
        self,
        query: str,
        collection_id: UUID,
        similarity_threshold: float = 0.85,
        limit: int = 5,
    ) -> list[dict]:
        """Search for similar cached entries for a given query.
        
        Args:
            query: The query to search for
            collection_id: ID of the original collection
            similarity_threshold: Minimum similarity score for cache hits
            limit: Maximum number of results to return
            
        Returns:
            list[dict]: List of matching cache entries with similarity scores
        """
        
        try:
            # Get cache collection ID
            cache_collection_id = await self._get_cache_collection_id(collection_id)
            
            # Generate embedding for the query
            query_embedding = await self.providers.embedding.async_get_embedding(query)
            
            # Search cache collection for similar queries
            search_settings = SearchSettings(
                use_semantic_search=True,
                limit=limit,
                include_scores=True,
                include_metadatas=True,
                filters={"collection_ids": {"$overlap": [str(cache_collection_id)]}}
            )
            
            # Perform semantic search on cache collection
            results = await self.providers.database.chunks_handler.semantic_search(
                query_vector=query_embedding,
                search_settings=search_settings
            )
            
            # Filter by similarity threshold and check for expired entries
            cache_hits = []
            for result in results:
                if result.score and result.score >= similarity_threshold:
                    # Check if entry is expired (only if TTL > 0)
                    cached_at_str = result.metadata.get("cached_at")
                    cache_ttl = result.metadata.get("cache_ttl", 0)
                    
                    if cached_at_str and cache_ttl > 0:  # Only check expiration if TTL is set
                        cached_at = datetime.fromisoformat(cached_at_str)
                        elapsed = (datetime.now() - cached_at).total_seconds()
                        
                        if elapsed > cache_ttl:
                            logger.info(f"Cache entry {result.id} expired, skipping")
                            continue
                    
                    cache_hits.append({
                        "id": result.id,
                        "original_query": result.text,
                        "generated_answer": result.metadata.get("generated_answer", ""),
                        "search_results": json.loads(result.metadata.get("search_results", "{}")),
                        "citations": json.loads(result.metadata.get("citations", "[]")),
                        "similarity_score": result.score,
                        "hit_count": result.metadata.get("hit_count", 0),
                        "cached_at": result.metadata.get("cached_at"),
                        "metadata": result.metadata
                    })
            
            return cache_hits
            
        except Exception as e:
            logger.error(f"Error searching cache entries: {e}")
            return []

    async def increment_cache_hit_count(self, cache_entry_id: UUID) -> None:
        """Increment the hit count for a cache entry."""
        try:
            # Get the current cache entry
            cache_entry = await self.providers.database.chunks_handler.get_chunk(cache_entry_id)
            if not cache_entry:
                logger.warning(f"Cache entry {cache_entry_id} not found")
                return
                
            # Update hit count
            metadata = json.loads(cache_entry["metadata"]) if isinstance(cache_entry["metadata"], str) else cache_entry["metadata"]
            metadata["hit_count"] = metadata.get("hit_count", 0) + 1
            metadata["last_accessed"] = datetime.now().isoformat()
            
            # Create updated vector entry
            updated_entry = VectorEntry(
                id=cache_entry_id,
                document_id=UUID(cache_entry["document_id"]),
                owner_id=UUID(cache_entry["owner_id"]),
                collection_ids=cache_entry["collection_ids"],
                vector=Vector(
                    data=json.loads(cache_entry["vec"]) if isinstance(cache_entry["vec"], str) else cache_entry["vec"],
                    type=VectorType.FIXED
                ),
                text=cache_entry["text"],
                metadata=metadata,
            )
            
            # Update the entry
            await self.providers.database.chunks_handler.upsert_entries([updated_entry])
            
        except Exception as e:
            logger.error(f"Error incrementing cache hit count: {e}")

    async def get_cache_analytics(self, collection_id: UUID) -> dict:
        """Get analytics and metrics for cache performance.
        
        Args:
            collection_id: ID of the original collection
            
        Returns:
            dict: Cache analytics including hit rates, entry counts, etc.
        """
        try:
            # Get cache collection ID
            cache_collection_id = await self._get_cache_collection_id(collection_id)
            
            # Get all cache entries for this collection
            cache_chunks = await self.providers.database.chunks_handler.list_chunks(
                offset=0,
                limit=10000,  # Large limit to get all entries
                filters={
                    "collection_ids": {"$overlap": [str(cache_collection_id)]}
                }
            )
            
            if not cache_chunks["results"]:
                return {
                    "collection_id": str(collection_id),
                    "cache_collection_id": str(cache_collection_id),
                    "total_entries": 0,
                    "total_hits": 0,
                    "average_hits_per_entry": 0,
                    "most_popular_queries": [],
                    "cache_size_mb": 0,
                    "oldest_entry": None,
                    "newest_entry": None
                }
            
            total_entries = len(cache_chunks["results"])
            total_hits = 0
            entry_details = []
            total_size_bytes = 0
            
            for chunk in cache_chunks["results"]:
                metadata = chunk.get("metadata", {})
                hit_count = metadata.get("hit_count", 0)
                total_hits += hit_count
                
                entry_details.append({
                    "query": chunk.get("text", ""),
                    "hit_count": hit_count,
                    "cached_at": metadata.get("cached_at"),
                    "last_accessed": metadata.get("last_accessed"),
                    "ttl": metadata.get("cache_ttl", 0)
                })
                
                # Estimate size (text + metadata)
                text_size = len(chunk.get("text", "").encode("utf-8"))
                metadata_size = len(str(metadata).encode("utf-8"))
                total_size_bytes += text_size + metadata_size
            
            # Sort by hit count for most popular queries
            entry_details.sort(key=lambda x: x["hit_count"], reverse=True)
            most_popular = entry_details[:10]  # Top 10
            
            # Find oldest and newest entries
            entries_with_dates = [e for e in entry_details if e["cached_at"]]
            oldest_entry = min(entries_with_dates, key=lambda x: x["cached_at"])["cached_at"] if entries_with_dates else None
            newest_entry = max(entries_with_dates, key=lambda x: x["cached_at"])["cached_at"] if entries_with_dates else None
            
            return {
                "collection_id": str(collection_id),
                "cache_collection_id": str(cache_collection_id),
                "total_entries": total_entries,
                "total_hits": total_hits,
                "average_hits_per_entry": round(total_hits / total_entries, 2) if total_entries > 0 else 0,
                "most_popular_queries": [
                    {"query": q["query"][:100], "hit_count": q["hit_count"]} 
                    for q in most_popular[:5]  # Top 5 for brevity
                ],
                "cache_size_mb": round(total_size_bytes / (1024 * 1024), 2),
                "oldest_entry": oldest_entry,
                "newest_entry": newest_entry,
                "entries_by_hit_count": {
                    "0_hits": len([e for e in entry_details if e["hit_count"] == 0]),
                    "1-5_hits": len([e for e in entry_details if 1 <= e["hit_count"] <= 5]),
                    "6-20_hits": len([e for e in entry_details if 6 <= e["hit_count"] <= 20]),
                    "20+_hits": len([e for e in entry_details if e["hit_count"] > 20])
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cache analytics: {e}")
            return {
                "collection_id": str(collection_id),
                "error": str(e),
                "total_entries": 0,
                "total_hits": 0
            }

    async def invalidate_cache_entries(
        self,
        collection_id: UUID,
        query_patterns: Optional[list[str]] = None,
        older_than_hours: Optional[int] = None,
        hit_count_threshold: Optional[int] = None,
        delete_all: bool = False
    ) -> dict:
        """Invalidate (delete) cache entries based on various criteria.
        
        Args:
            collection_id: ID of the original collection
            query_patterns: List of query patterns to match for deletion
            older_than_hours: Delete entries older than X hours
            hit_count_threshold: Delete entries with hit count below threshold
            delete_all: Delete all cache entries for this collection
            
        Returns:
            dict: Summary of deletion operation
        """
        try:
            # Get cache collection ID
            cache_collection_id = await self._get_cache_collection_id(collection_id)
            
            # Get all cache entries for this collection
            cache_chunks = await self.providers.database.chunks_handler.list_chunks(
                offset=0,
                limit=10000,
                filters={
                    "collection_ids": {"$overlap": [str(cache_collection_id)]},
                    "metadata": {"$contains": {"type": "semantic_cache_entry"}}
                }
            )
            
            if not cache_chunks["results"]:
                return {
                    "collection_id": str(collection_id),
                    "deleted_count": 0,
                    "message": "No cache entries found"
                }
            
            entries_to_delete = []
            total_entries = len(cache_chunks["results"])
            
            for chunk in cache_chunks["results"]:
                should_delete = delete_all
                
                if not should_delete and query_patterns:
                    query_text = chunk.get("text", "").lower()
                    for pattern in query_patterns:
                        if re.search(pattern.lower(), query_text):
                            should_delete = True
                            break
                
                if not should_delete and older_than_hours:
                    cached_at_str = chunk.get("metadata", {}).get("cached_at")
                    if cached_at_str:
                        cached_at = datetime.fromisoformat(cached_at_str)
                        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
                        if cached_at < cutoff_time:
                            should_delete = True
                
                if not should_delete and hit_count_threshold is not None:
                    hit_count = chunk.get("metadata", {}).get("hit_count", 0)
                    if hit_count < hit_count_threshold:
                        should_delete = True
                
                if should_delete:
                    entries_to_delete.append(chunk["id"])
            
            # Delete the selected entries
            if entries_to_delete:
                for entry_id in entries_to_delete:
                    await self.providers.database.chunks_handler.delete(
                        filters={"id": {"$eq": entry_id}}
                    )
            
            return {
                "collection_id": str(collection_id),
                "cache_collection_id": str(cache_collection_id),
                "total_entries": total_entries,
                "deleted_count": len(entries_to_delete),
                "remaining_count": total_entries - len(entries_to_delete),
                "deletion_criteria": {
                    "query_patterns": query_patterns,
                    "older_than_hours": older_than_hours,
                    "hit_count_threshold": hit_count_threshold,
                    "delete_all": delete_all
                }
            }
            
        except Exception as e:
            logger.error(f"Error invalidating cache entries: {e}")
            return {
                "collection_id": str(collection_id),
                "error": str(e),
                "deleted_count": 0
            }

    async def cleanup_expired_cache_entries(self, collection_id: Optional[UUID] = None) -> dict:
        """Clean up expired cache entries based on TTL.
        
        Args:
            collection_id: Optional specific collection to clean, or None for all collections
            
        Returns:
            dict: Summary of cleanup operation
        """
        
        try:
            deleted_count = 0
            processed_collections = []
            
            if collection_id:
                # Clean specific collection
                cache_collection_id = await self._get_cache_collection_id(collection_id)
                collection_ids_to_process = [cache_collection_id]
                processed_collections.append(str(collection_id))
            else:
                # Get all collections and find their cache collections
                collections_overview = await self.providers.database.collections_handler.get_collections_overview(
                    offset=0,
                    limit=1000
                )
                
                collection_ids_to_process = []
                for collection in collections_overview["results"]:
                    if collection.name.endswith("_cache"):
                        collection_ids_to_process.append(collection.id)
                        # Extract original collection name
                        original_name = collection.name[:-6]  # Remove "_cache" suffix
                        processed_collections.append(original_name)
            
            # Process each cache collection
            for cache_coll_id in collection_ids_to_process:
                cache_chunks = await self.providers.database.chunks_handler.list_chunks(
                    offset=0,
                    limit=10000,
                    filters={
                        "collection_ids": {"$overlap": [str(cache_coll_id)]}
                    }
                )
                
                for chunk in cache_chunks["results"]:
                    metadata = chunk.get("metadata", {})
                    cache_ttl = metadata.get("cache_ttl", 0)
                    cached_at_str = metadata.get("cached_at")
                    
                    # Skip entries with indefinite storage (TTL = 0)
                    if cache_ttl == 0 or not cached_at_str:
                        continue
                    
                    # Check if entry has expired
                    cached_at = datetime.fromisoformat(cached_at_str)
                    elapsed = (datetime.now() - cached_at).total_seconds()
                    
                    if elapsed > cache_ttl:
                        # Delete expired entry
                        await self.providers.database.chunks_handler.delete(
                            filters={"id": {"$eq": chunk["id"]}}
                        )
                        deleted_count += 1
            
            return {
                "deleted_expired_entries": deleted_count,
                "processed_collections": processed_collections,
                "cleanup_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up expired cache entries: {e}")
            return {
                "error": str(e),
                "deleted_expired_entries": 0
            }

    async def list_chunks(
        self,
        offset: int,
        limit: int,
        filters: Optional[dict[str, Any]] = None,
        include_vectors: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        return await self.providers.database.chunks_handler.list_chunks(
            offset=offset,
            limit=limit,
            filters=filters,
            include_vectors=include_vectors,
        )

    async def get_chunk(
        self,
        chunk_id: UUID,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        return await self.providers.database.chunks_handler.get_chunk(chunk_id)

    # Phase 2: Cache Entry Management Methods

    async def get_cache_entries(
        self,
        collection_id: UUID,
        include_expired: bool = False,
        format: str = "plain",  # "plain" or "detailed"
        offset: int = 0,
        limit: int = 100
    ) -> CacheEntriesResponse:
        """
        Get all cache entries for a collection in plain text or detailed format.
        
        Args:
            collection_id: The collection ID
            include_expired: Whether to include expired entries
            format: Output format - "plain" for text, "detailed" for full objects
            offset: Pagination offset
            limit: Number of entries to return (max 1000)
            
        Returns:
            CacheEntriesResponse with entries in requested format
        """
        try:
            # Validate input parameters
            if format not in ["plain", "detailed"]:
                raise ValueError("Format must be 'plain' or 'detailed'")
            
            if limit > 1000:
                limit = 1000  # Cap at reasonable maximum
            
            # Get cache collection ID
            cache_collection_id = await self._get_cache_collection_id(collection_id)
            if not cache_collection_id:
                raise ValueError(f"No cache collection found for {collection_id}")
            
            # Fetch entries from cache collection with pagination
            # Note: We only filter by collection_id, similar to how semantic_search works
            # The metadata filter was causing issues with retrieving cache entries
            filters = {
                "collection_ids": {"$overlap": [str(cache_collection_id)]}
            }
            
            results = await self.providers.database.chunks_handler.list_chunks(
                filters=filters,
                offset=offset,
                limit=limit,
                include_vectors=True  # Include vectors to ensure we get all data
            )
            
            entries = []
            for chunk in results["results"]:
                metadata = chunk["metadata"]
                
                # Check if expired
                cached_at = datetime.fromisoformat(metadata["cached_at"])
                ttl_seconds = metadata.get("cache_ttl", 0)
                is_expired = False
                expires_at = None
                
                if ttl_seconds > 0:
                    expires_at = cached_at + timedelta(seconds=ttl_seconds)
                    is_expired = datetime.now() > expires_at
                
                # Skip expired entries if not requested
                if is_expired and not include_expired:
                    continue
                
                if format == "plain":
                    # Format as plain text
                    last_accessed = metadata.get("last_accessed", "Never")
                    ttl_display = "Never expires" if ttl_seconds == 0 else f"{ttl_seconds} seconds"
                    
                    entry_text = f"""Query: {metadata['original_query']}
Answer: {metadata['generated_answer']}
TTL: {ttl_display}
Hit Count: {metadata.get('hit_count', 0)}
Last Accessed: {last_accessed}
Cached At: {metadata['cached_at']}
Entry ID: {chunk['id']}
---"""
                    entries.append(entry_text)
                else:
                    # Format as detailed object
                    # Parse JSON data with error handling
                    try:
                        search_results = json.loads(metadata.get("search_results", "{}"))
                    except json.JSONDecodeError:
                        search_results = {}
                        logger.warning(f"Failed to parse search_results for cache entry {chunk['id']}")
                    
                    try:
                        citations = json.loads(metadata.get("citations", "[]"))
                    except json.JSONDecodeError:
                        citations = []
                        logger.warning(f"Failed to parse citations for cache entry {chunk['id']}")
                    
                    entry = CacheEntryDetail(
                        entry_id=chunk["id"],
                        query=metadata["original_query"],
                        answer=metadata["generated_answer"],
                        search_results=search_results,
                        citations=citations,
                        collection_id=str(collection_id),
                        cached_at=cached_at,
                        ttl_seconds=ttl_seconds,
                        hit_count=metadata.get("hit_count", 0),
                        last_accessed=metadata.get("last_accessed"),
                        is_expired=is_expired,
                        expires_at=expires_at
                    )
                    entries.append(entry)
            
            return CacheEntriesResponse(
                entries=entries,
                total_count=results.get("total_entries", len(entries)),
                format=format
            )
            
        except Exception as e:
            logger.error(f"Error getting cache entries: {e}")
            raise

    async def get_cache_entry_details(
        self,
        collection_id: UUID,
        entry_id: str
    ) -> CacheEntryDetail:
        """
        Get detailed information for a specific cache entry.
        
        Args:
            collection_id: The collection ID
            entry_id: The cache entry ID
            
        Returns:
            CacheEntryDetail with full information
        """
        try:
            # Validate entry_id is valid UUID format
            try:
                entry_uuid = UUID(entry_id)
            except ValueError:
                raise ValueError(f"Invalid entry ID format: {entry_id}")
            # Get cache collection ID
            cache_collection_id = await self._get_cache_collection_id(collection_id)
            if not cache_collection_id:
                raise ValueError(f"No cache collection found for {collection_id}")
            
            # Fetch the specific entry
            chunk = await self.providers.database.chunks_handler.get_chunk(
                chunk_id=entry_uuid
            )
            
            if not chunk:
                raise ValueError(f"Cache entry {entry_id} not found")
                
            metadata = chunk["metadata"]
            
            # Verify it belongs to the correct cache collection
            if str(cache_collection_id) not in chunk["collection_ids"]:
                raise ValueError(f"Cache entry {entry_id} does not belong to collection {collection_id}")
            
            # Calculate expiration
            cached_at = datetime.fromisoformat(metadata["cached_at"])
            ttl_seconds = metadata.get("cache_ttl", 0)
            is_expired = False
            expires_at = None
            
            if ttl_seconds > 0:
                expires_at = cached_at + timedelta(seconds=ttl_seconds)
                is_expired = datetime.now() > expires_at
            
            # Parse JSON data with error handling
            try:
                search_results = json.loads(metadata.get("search_results", "{}"))
            except json.JSONDecodeError:
                search_results = {}
                logger.warning(f"Failed to parse search_results for cache entry {entry_id}")
            
            try:
                citations = json.loads(metadata.get("citations", "[]"))
            except json.JSONDecodeError:
                citations = []
                logger.warning(f"Failed to parse citations for cache entry {entry_id}")
            
            return CacheEntryDetail(
                entry_id=entry_id,
                query=metadata["original_query"],
                answer=metadata["generated_answer"],
                search_results=search_results,
                citations=citations,
                collection_id=str(collection_id),
                cached_at=cached_at,
                ttl_seconds=ttl_seconds,
                hit_count=metadata.get("hit_count", 0),
                last_accessed=metadata.get("last_accessed"),
                is_expired=is_expired,
                expires_at=expires_at
            )
            
        except Exception as e:
            logger.error(f"Error getting cache entry details: {e}")
            raise

    async def update_cache_entry(
        self,
        collection_id: UUID,
        entry_id: str,
        answer: Optional[str] = None,
        search_results: Optional[dict] = None,
        citations: Optional[list] = None,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Update content of a cache entry. Note: Query remains immutable.
        
        Args:
            collection_id: The collection ID
            entry_id: The cache entry ID
            answer: New answer content
            search_results: Updated search results
            citations: Updated citations
            ttl_seconds: New TTL (None=no change, 0=never expire, >0=TTL)
            
        Returns:
            bool: True if successful
        """
        try:
            # Validate entry_id is valid UUID format
            try:
                entry_uuid = UUID(entry_id)
            except ValueError:
                raise ValueError(f"Invalid entry ID format: {entry_id}")
            # Get cache collection ID
            cache_collection_id = await self._get_cache_collection_id(collection_id)
            if not cache_collection_id:
                raise ValueError(f"No cache collection found for {collection_id}")
            
            # Fetch existing entry with vectors for potential updates
            existing = await self.providers.database.chunks_handler.get_chunk(
                chunk_id=entry_uuid,
                include_vectors=True
            )
            if not existing:
                raise ValueError(f"Cache entry {entry_id} not found")
                
            # Verify it belongs to the correct cache collection
            if str(cache_collection_id) not in existing["collection_ids"]:
                raise ValueError(f"Cache entry {entry_id} does not belong to collection {collection_id}")
            
            # Update metadata
            metadata = existing["metadata"].copy()
            
            # Update only provided fields
            if answer is not None:
                metadata["generated_answer"] = answer
            if search_results is not None:
                try:
                    metadata["search_results"] = json.dumps(search_results)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid search_results format: {e}")
            if citations is not None:
                try:
                    metadata["citations"] = json.dumps(citations)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid citations format: {e}")
            if ttl_seconds is not None:
                metadata["cache_ttl"] = ttl_seconds
                
            # Update timestamp but preserve hit count
            metadata["updated_at"] = datetime.now().isoformat()
            
            # Update the entry (keep query and embedding unchanged)
            # Note: We need to update through the vector entry mechanism
            # since chunks_handler doesn't have update_entry method
            vector_data = existing.get("vector")
            if not vector_data:
                # If no vector data, we need to handle this case
                raise ValueError(f"No vector data found for cache entry {entry_id}")
            
            updated_entry = VectorEntry(
                id=entry_uuid,
                document_id=UUID(existing.get("document_id")),
                owner_id=UUID(existing.get("owner_id")) if existing.get("owner_id") else None,
                collection_ids=[UUID(cid) if isinstance(cid, str) else cid for cid in existing.get("collection_ids", [])],
                vector=Vector(
                    data=vector_data,
                    type=VectorType.FIXED,
                    length=len(vector_data)
                ),
                text=existing.get("text"),
                metadata=metadata
            )
            await self.providers.database.chunks_handler.upsert_entries([updated_entry])
            
            logger.info(f"Updated cache entry {entry_id} for collection {collection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating cache entry: {e}")
            raise

    async def bulk_update_cache_entries(
        self,
        collection_id: UUID,
        updates: list[CacheEntryUpdate]
    ) -> dict:
        """
        Update multiple cache entries at once.
        
        Args:
            collection_id: The collection ID
            updates: List of updates to apply
            
        Returns:
            dict: Summary of results {"succeeded": count, "failed": count, "errors": [...]}
        """
        try:
            # Get cache collection ID
            cache_collection_id = await self._get_cache_collection_id(collection_id)
            if not cache_collection_id:
                raise ValueError(f"No cache collection found for {collection_id}")
            
            results = {"succeeded": 0, "failed": 0, "errors": []}
            
            for update in updates:
                try:
                    await self.update_cache_entry(
                        collection_id=collection_id,
                        entry_id=update.entry_id,
                        answer=update.generated_answer,
                        search_results=update.search_results,
                        citations=update.citations,
                        ttl_seconds=update.ttl_seconds
                    )
                    results["succeeded"] += 1
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({
                        "entry_id": update.entry_id,
                        "error": str(e)
                    })
                    logger.error(f"Failed to update cache entry {update.entry_id}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk update: {e}")
            raise

    async def delete_cache_entries(
        self,
        collection_id: UUID,
        entry_ids: list[str]
    ) -> dict:
        """
        Delete specific cache entries.
        
        Args:
            collection_id: The collection ID
            entry_ids: List of entry IDs to delete
            
        Returns:
            dict: Summary {"deleted": count, "failed": count, "errors": [...]}
        """
        try:
            # Get cache collection ID
            cache_collection_id = await self._get_cache_collection_id(collection_id)
            if not cache_collection_id:
                raise ValueError(f"No cache collection found for {collection_id}")
            
            results = {"deleted": 0, "failed": 0, "errors": []}
            
            for entry_id in entry_ids:
                try:
                    # Validate entry_id is valid UUID format
                    try:
                        entry_uuid = UUID(entry_id)
                    except ValueError:
                        raise ValueError(f"Invalid entry ID format: {entry_id}")
                    
                    # Verify entry belongs to this collection before deleting
                    chunk = await self.providers.database.chunks_handler.get_chunk(
                        chunk_id=entry_uuid
                    )
                    
                    if chunk and str(cache_collection_id) in chunk["collection_ids"]:
                        # Use delete method with correct filter syntax
                        await self.providers.database.chunks_handler.delete(
                            filters={"id": {"$eq": entry_uuid}}
                        )
                        results["deleted"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append({
                            "entry_id": entry_id,
                            "error": "Entry not found or does not belong to collection"
                        })
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({
                        "entry_id": entry_id,
                        "error": str(e)
                    })
                    logger.error(f"Failed to delete cache entry {entry_id}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error deleting cache entries: {e}")
            raise

    async def update_cache_ttls(
        self,
        collection_id: UUID,
        ttl_updates: dict[str, Optional[int]]
    ) -> dict:
        """
        Update TTLs for cache entries.
        
        Args:
            collection_id: The collection ID
            ttl_updates: Map of entry_id to new TTL (None or 0 = never expire)
            
        Returns:
            dict: Summary {"updated": count, "failed": count, "errors": [...]}
        """
        try:
            results = {"updated": 0, "failed": 0, "errors": []}
            
            for entry_id, ttl_seconds in ttl_updates.items():
                try:
                    # Use None or 0 to mean "never expire"
                    ttl_value = 0 if ttl_seconds is None else ttl_seconds
                    
                    await self.update_cache_entry(
                        collection_id=collection_id,
                        entry_id=entry_id,
                        ttl_seconds=ttl_value
                    )
                    results["updated"] += 1
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({
                        "entry_id": entry_id,
                        "error": str(e)
                    })
                    logger.error(f"Failed to update TTL for entry {entry_id}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error updating cache TTLs: {e}")
            raise


class IngestionServiceAdapter:
    @staticmethod
    def _parse_user_data(user_data) -> User:
        if isinstance(user_data, str):
            try:
                user_data = json.loads(user_data)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid user data format: {user_data}"
                ) from e
        return User.from_dict(user_data)

    @staticmethod
    def parse_ingest_file_input(data: dict) -> dict:
        return {
            "user": IngestionServiceAdapter._parse_user_data(data["user"]),
            "metadata": data["metadata"],
            "document_id": (
                UUID(data["document_id"]) if data["document_id"] else None
            ),
            "version": data.get("version"),
            "ingestion_config": data["ingestion_config"] or {},
            "file_data": data["file_data"],
            "size_in_bytes": data["size_in_bytes"],
            "collection_ids": data.get("collection_ids", []),
        }

    @staticmethod
    def parse_ingest_chunks_input(data: dict) -> dict:
        return {
            "user": IngestionServiceAdapter._parse_user_data(data["user"]),
            "metadata": data["metadata"],
            "document_id": data["document_id"],
            "chunks": [
                UnprocessedChunk.from_dict(chunk) for chunk in data["chunks"]
            ],
            "id": data.get("id"),
        }

    @staticmethod
    def parse_update_chunk_input(data: dict) -> dict:
        return {
            "user": IngestionServiceAdapter._parse_user_data(data["user"]),
            "document_id": UUID(data["document_id"]),
            "id": UUID(data["id"]),
            "text": data["text"],
            "metadata": data.get("metadata"),
            "collection_ids": data.get("collection_ids", []),
        }

    @staticmethod
    def parse_update_files_input(data: dict) -> dict:
        return {
            "user": IngestionServiceAdapter._parse_user_data(data["user"]),
            "document_ids": [UUID(doc_id) for doc_id in data["document_ids"]],
            "metadatas": data["metadatas"],
            "ingestion_config": data["ingestion_config"],
            "file_sizes_in_bytes": data["file_sizes_in_bytes"],
            "file_datas": data["file_datas"],
        }

    @staticmethod
    def parse_create_vector_index_input(data: dict) -> dict:
        return {
            "table_name": VectorTableName(data["table_name"]),
            "index_method": IndexMethod(data["index_method"]),
            "index_measure": IndexMeasure(data["index_measure"]),
            "index_name": data["index_name"],
            "index_column": data["index_column"],
            "index_arguments": data["index_arguments"],
            "concurrently": data["concurrently"],
        }

    @staticmethod
    def parse_list_vector_indices_input(input_data: dict) -> dict:
        return {"table_name": input_data["table_name"]}

    @staticmethod
    def parse_delete_vector_index_input(input_data: dict) -> dict:
        return {
            "index_name": input_data["index_name"],
            "table_name": input_data.get("table_name"),
            "concurrently": input_data.get("concurrently", True),
        }

    @staticmethod
    def parse_select_vector_index_input(input_data: dict) -> dict:
        return {
            "index_name": input_data["index_name"],
            "table_name": input_data.get("table_name"),
        }
