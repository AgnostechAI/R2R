import uuid
import pytest

from core.base.api.models import CollectionResponse


@pytest.mark.asyncio
async def test_create_cache_collection(management_service):
    """Test creating a cache collection without a knowledge graph."""
    owner_id = uuid.uuid4()
    
    # Create a cache collection
    collection = await management_service.create_cache_collection(
        owner_id=owner_id,
        name="Test Cache Collection",
        description="A test cache collection without a graph"
    )
    
    assert isinstance(collection, CollectionResponse)
    assert collection.name == "Test Cache Collection_cache"  # Should have _cache appended
    assert collection.owner_id == owner_id
    assert collection.description == "A test cache collection without a graph"
    
    # Verify it's identified as a cache collection
    is_cache = await management_service.is_cache_collection(collection.id)
    assert is_cache is True


@pytest.mark.asyncio
async def test_create_regular_collection_vs_cache_collection(management_service):
    """Test that regular collections have graphs while cache collections don't."""
    owner_id = uuid.uuid4()
    
    # Create a regular collection
    regular_collection = await management_service.create_collection(
        owner_id=owner_id,
        name="Regular Collection",
        description="A regular collection with a graph"
    )
    
    # Create a cache collection  
    cache_collection = await management_service.create_cache_collection(
        owner_id=owner_id,
        name="Cache Collection",
        description="A cache collection without a graph"
    )
    
    # Verify naming conventions
    assert regular_collection.name == "Regular Collection"
    assert cache_collection.name == "Cache Collection_cache"
    
    # Verify the regular collection is NOT a cache collection
    is_regular_cache = await management_service.is_cache_collection(regular_collection.id)
    assert is_regular_cache is False
    
    # Verify the cache collection IS a cache collection
    is_cache_cache = await management_service.is_cache_collection(cache_collection.id)
    assert is_cache_cache is True


@pytest.mark.asyncio
async def test_delete_cache_collection(management_service):
    """Test deleting a cache collection."""
    owner_id = uuid.uuid4()
    
    # Create a cache collection
    collection = await management_service.create_cache_collection(
        owner_id=owner_id,
        name="Delete Test Cache Collection",
        description="A cache collection to be deleted"
    )
    
    # Verify it exists and is a cache collection
    is_cache = await management_service.is_cache_collection(collection.id)
    assert is_cache is True
    
    # Delete the cache collection
    result = await management_service.delete_cache_collection(collection.id)
    assert result is True


@pytest.mark.asyncio 
async def test_cache_collection_with_default_name(management_service):
    """Test creating a cache collection with default naming."""
    owner_id = uuid.uuid4()
    
    # Create cache collection without specifying name
    collection = await management_service.create_cache_collection(owner_id=owner_id)
    
    assert isinstance(collection, CollectionResponse)
    assert collection.owner_id == owner_id
    # Should have a default name from config
    assert collection.name is not None
    
    # Verify it's a cache collection
    is_cache = await management_service.is_cache_collection(collection.id)
    assert is_cache is True


@pytest.mark.asyncio
async def test_cache_collection_with_description_only(management_service):
    """Test creating a cache collection with only description."""
    owner_id = uuid.uuid4()
    
    collection = await management_service.create_cache_collection(
        owner_id=owner_id,
        description="Cache collection with only description"
    )
    
    assert isinstance(collection, CollectionResponse)
    assert collection.description == "Cache collection with only description"
    
    # Verify it's a cache collection
    is_cache = await management_service.is_cache_collection(collection.id)
    assert is_cache is True 