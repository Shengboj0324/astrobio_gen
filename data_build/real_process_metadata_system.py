#!/usr/bin/env python3
"""
Real Process Metadata Collection System
======================================

State-of-the-art process metadata collection system that:
- Collects ONLY real data from live APIs
- No fallback or mock data generation
- Proper SSL certificate handling
- Honest success/failure reporting
- Robust error handling without compromising data integrity

This system fails fast and reports accurately - no false positives.
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
import ssl
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

import aiohttp
import certifi
import requests

# Configure robust logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProcessMetadataType(Enum):
    """Types of process metadata we collect"""

    EXPERIMENTAL_PROVENANCE = "experimental_provenance"
    OBSERVATIONAL_CONTEXT = "observational_context"
    COMPUTATIONAL_LINEAGE = "computational_lineage"
    METHODOLOGICAL_EVOLUTION = "methodological_evolution"
    QUALITY_CONTROL_PROCESSES = "quality_control_processes"
    DECISION_TREES = "decision_trees"
    SYSTEMATIC_BIASES = "systematic_biases"
    FAILED_EXPERIMENTS = "failed_experiments"


class SourcePlatform(Enum):
    """Verified source platforms with working APIs"""

    ARXIV = "arxiv"
    PUBMED = "pubmed"
    CROSSREF = "crossref"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENALEX = "openalex"
    ZENODO = "zenodo"


@dataclass
class RealProcessMetadataSource:
    """A verified real process metadata source"""

    source_id: str
    platform: SourcePlatform
    metadata_type: ProcessMetadataType
    title: str
    description: str
    url: str
    doi: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[str] = None
    source_quality: float = 0.0
    verification_status: str = "verified"
    collection_timestamp: Optional[datetime] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


class RealDataCollector:
    """Collects real data from verified academic and scientific sources"""

    def __init__(self):
        # Create SSL context with proper certificate verification
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

        # API endpoints for real data sources
        self.api_endpoints = {
            SourcePlatform.ARXIV: "http://export.arxiv.org/api/query",
            SourcePlatform.CROSSREF: "https://api.crossref.org/works",
            SourcePlatform.SEMANTIC_SCHOLAR: "https://api.semanticscholar.org/graph/v1/paper/search",
            SourcePlatform.OPENALEX: "https://api.openalex.org/works",
            SourcePlatform.ZENODO: "https://zenodo.org/api/records",
        }

        # Search terms optimized for each process metadata type
        self.search_strategies = {
            ProcessMetadataType.EXPERIMENTAL_PROVENANCE: [
                "laboratory protocol astrobiology",
                "experimental methodology exoplanet",
                "sample preparation space science",
                "analytical methods planetary science",
                "laboratory standards astrobiology",
            ],
            ProcessMetadataType.OBSERVATIONAL_CONTEXT: [
                "telescope observations exoplanet",
                "astronomical instrumentation",
                "observational methodology astronomy",
                "calibration procedures telescope",
                "atmospheric observations planetary",
            ],
            ProcessMetadataType.COMPUTATIONAL_LINEAGE: [
                "data processing pipeline astronomy",
                "computational methods astrophysics",
                "algorithm development planetary",
                "software methodology astronomical",
                "numerical methods space science",
            ],
            ProcessMetadataType.METHODOLOGICAL_EVOLUTION: [
                "methodology development astrobiology",
                "technique evolution planetary science",
                "historical methods astronomy",
                "paradigm shift exoplanet research",
                "methodological advances space",
            ],
            ProcessMetadataType.QUALITY_CONTROL_PROCESSES: [
                "quality assurance astronomy",
                "validation procedures astrophysics",
                "uncertainty analysis planetary",
                "error assessment astronomical",
                "quality metrics space science",
            ],
            ProcessMetadataType.DECISION_TREES: [
                "decision criteria astronomical",
                "selection methodology exoplanet",
                "evaluation framework planetary",
                "reasoning processes astrobiology",
                "expert judgment astronomy",
            ],
            ProcessMetadataType.SYSTEMATIC_BIASES: [
                "systematic error astronomy",
                "bias assessment astrophysics",
                "measurement uncertainty planetary",
                "instrumental bias telescope",
                "systematic effects space",
            ],
            ProcessMetadataType.FAILED_EXPERIMENTS: [
                "negative results astrobiology",
                "null findings exoplanet",
                "unsuccessful observations astronomy",
                "failed detection planetary",
                "inconclusive results space",
            ],
        }

        # Request session with proper headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Astrobiology-Research-Process-Metadata-Collector/1.0 (academic-research)",
                "Accept": "application/json, application/xml",
            }
        )

        logger.info("RealDataCollector initialized with verified SSL context")

    async def collect_real_sources(
        self,
        metadata_type: ProcessMetadataType,
        target_count: int = 100,
        timeout_seconds: int = 300,
    ) -> List[RealProcessMetadataSource]:
        """Collect real sources for a specific metadata type with timeout"""

        logger.info(
            f"Starting REAL data collection for {metadata_type.value} (target: {target_count})"
        )
        start_time = time.time()

        real_sources = []
        search_terms = self.search_strategies[metadata_type]

        # Collect from multiple platforms in parallel with timeout
        collection_tasks = []

        # Only use platforms we can actually access
        working_platforms = [
            SourcePlatform.CROSSREF,
            SourcePlatform.OPENALEX,
            SourcePlatform.SEMANTIC_SCHOLAR,
        ]

        for platform in working_platforms:
            for search_term in search_terms[:3]:  # Limit search terms for efficiency
                task = self._collect_from_platform(
                    platform,
                    metadata_type,
                    search_term,
                    target_count // len(working_platforms) // 3,
                )
                collection_tasks.append(task)

        try:
            # Execute with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*collection_tasks, return_exceptions=True), timeout=timeout_seconds
            )

            # Process results and filter out exceptions
            for result in results:
                if isinstance(result, list):
                    real_sources.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Collection task failed: {result}")

            # Remove duplicates and validate
            real_sources = self._deduplicate_and_validate(real_sources)

            collection_time = time.time() - start_time

            if len(real_sources) < target_count * 0.5:  # Less than 50% of target
                logger.error(
                    f"INSUFFICIENT REAL DATA: Only {len(real_sources)}/{target_count} sources collected for {metadata_type.value}"
                )
                raise ValueError(
                    f"Failed to collect sufficient real data: {len(real_sources)}/{target_count}"
                )

            logger.info(
                f"Successfully collected {len(real_sources)} REAL sources for {metadata_type.value} in {collection_time:.2f}s"
            )
            return real_sources[:target_count]  # Return exactly target count

        except asyncio.TimeoutError:
            logger.error(f"Collection timeout after {timeout_seconds}s for {metadata_type.value}")
            raise TimeoutError(f"Real data collection timed out for {metadata_type.value}")
        except Exception as e:
            logger.error(f"Real data collection failed for {metadata_type.value}: {e}")
            raise

    async def _collect_from_platform(
        self,
        platform: SourcePlatform,
        metadata_type: ProcessMetadataType,
        search_term: str,
        count: int,
    ) -> List[RealProcessMetadataSource]:
        """Collect real sources from a specific platform"""

        sources = []

        try:
            if platform == SourcePlatform.CROSSREF:
                sources = await self._collect_from_crossref(metadata_type, search_term, count)
            elif platform == SourcePlatform.OPENALEX:
                sources = await self._collect_from_openalex(metadata_type, search_term, count)
            elif platform == SourcePlatform.SEMANTIC_SCHOLAR:
                sources = await self._collect_from_semantic_scholar(
                    metadata_type, search_term, count
                )

            logger.info(
                f"Collected {len(sources)} real sources from {platform.value} for '{search_term}'"
            )
            return sources

        except Exception as e:
            logger.warning(f"Failed to collect from {platform.value} for '{search_term}': {e}")
            return []

    async def _collect_from_crossref(
        self, metadata_type: ProcessMetadataType, search_term: str, count: int
    ) -> List[RealProcessMetadataSource]:
        """Collect real sources from CrossRef API"""

        sources = []

        try:
            # Build CrossRef query
            params = {
                "query": search_term,
                "rows": min(count, 50),
                "filter": "type:journal-article",
                "sort": "relevance",
                "order": "desc",
            }

            url = f"{self.api_endpoints[SourcePlatform.CROSSREF]}?{urlencode(params)}"

            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=self.ssl_context),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        items = data.get("message", {}).get("items", [])

                        for item in items[:count]:
                            # Extract real metadata
                            title = " ".join(item.get("title", ["No title"]))
                            abstract = item.get("abstract", "No abstract available")

                            # Get authors
                            authors = []
                            for author in item.get("author", []):
                                if "given" in author and "family" in author:
                                    authors.append(f"{author['given']} {author['family']}")

                            # Get DOI and URL
                            doi = item.get("DOI")
                            url = item.get("URL", f"https://doi.org/{doi}" if doi else "")

                            # Get publication date
                            pub_date = None
                            if "published-print" in item:
                                date_parts = item["published-print"].get("date-parts", [])
                                if date_parts and len(date_parts[0]) >= 3:
                                    pub_date = f"{date_parts[0][0]}-{date_parts[0][1]:02d}-{date_parts[0][2]:02d}"

                            source = RealProcessMetadataSource(
                                source_id=f"crossref_{doi.replace('/', '_') if doi else uuid.uuid4().hex[:8]}",
                                platform=SourcePlatform.CROSSREF,
                                metadata_type=metadata_type,
                                title=title,
                                description=abstract[:500] if abstract else "",
                                url=url,
                                doi=doi,
                                authors=authors,
                                publication_date=pub_date,
                                source_quality=self._calculate_crossref_quality(item),
                                collection_timestamp=datetime.now(timezone.utc),
                                raw_metadata=item,
                            )

                            sources.append(source)

                    else:
                        logger.warning(f"CrossRef API returned status {response.status}")

        except Exception as e:
            logger.error(f"CrossRef collection error: {e}")

        return sources

    async def _collect_from_openalex(
        self, metadata_type: ProcessMetadataType, search_term: str, count: int
    ) -> List[RealProcessMetadataSource]:
        """Collect real sources from OpenAlex API"""

        sources = []

        try:
            # Build OpenAlex query
            params = {
                "search": search_term,
                "per-page": min(count, 50),
                "filter": "type:article",
                "sort": "relevance_score:desc",
            }

            url = f"{self.api_endpoints[SourcePlatform.OPENALEX]}?{urlencode(params)}"

            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=self.ssl_context),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        results = data.get("results", [])

                        for item in results[:count]:
                            # Extract real metadata
                            title = item.get("title", "No title")
                            abstract = item.get("abstract", "No abstract available")

                            # Get authors
                            authors = []
                            for authorship in item.get("authorships", []):
                                author = authorship.get("author", {})
                                if "display_name" in author:
                                    authors.append(author["display_name"])

                            # Get DOI and URL
                            doi = item.get("doi")
                            if doi and doi.startswith("https://doi.org/"):
                                doi = doi.replace("https://doi.org/", "")

                            url = item.get("primary_location", {}).get("landing_page_url", "")
                            if not url and doi:
                                url = f"https://doi.org/{doi}"

                            # Get publication date
                            pub_date = item.get("publication_date")

                            source = RealProcessMetadataSource(
                                source_id=f"openalex_{item.get('id', uuid.uuid4().hex).split('/')[-1]}",
                                platform=SourcePlatform.OPENALEX,
                                metadata_type=metadata_type,
                                title=title,
                                description=abstract[:500] if abstract else "",
                                url=url,
                                doi=doi,
                                authors=authors,
                                publication_date=pub_date,
                                source_quality=self._calculate_openalex_quality(item),
                                collection_timestamp=datetime.now(timezone.utc),
                                raw_metadata=item,
                            )

                            sources.append(source)

                    else:
                        logger.warning(f"OpenAlex API returned status {response.status}")

        except Exception as e:
            logger.error(f"OpenAlex collection error: {e}")

        return sources

    async def _collect_from_semantic_scholar(
        self, metadata_type: ProcessMetadataType, search_term: str, count: int
    ) -> List[RealProcessMetadataSource]:
        """Collect real sources from Semantic Scholar API"""

        sources = []

        try:
            # Build Semantic Scholar query
            params = {
                "query": search_term,
                "limit": min(count, 50),
                "fields": "paperId,title,abstract,authors,year,url,externalIds,publicationDate",
            }

            url = f"{self.api_endpoints[SourcePlatform.SEMANTIC_SCHOLAR]}?{urlencode(params)}"

            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=self.ssl_context),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        papers = data.get("data", [])

                        for paper in papers[:count]:
                            # Extract real metadata
                            title = paper.get("title", "No title")
                            abstract = paper.get("abstract", "No abstract available")

                            # Get authors
                            authors = []
                            for author in paper.get("authors", []):
                                if "name" in author:
                                    authors.append(author["name"])

                            # Get DOI and URL
                            external_ids = paper.get("externalIds", {})
                            doi = external_ids.get("DOI")

                            url = paper.get("url", "")
                            if not url and doi:
                                url = f"https://doi.org/{doi}"

                            # Get publication date
                            pub_date = paper.get("publicationDate") or str(paper.get("year", ""))

                            source = RealProcessMetadataSource(
                                source_id=f"semantic_scholar_{paper.get('paperId', uuid.uuid4().hex)}",
                                platform=SourcePlatform.SEMANTIC_SCHOLAR,
                                metadata_type=metadata_type,
                                title=title,
                                description=abstract[:500] if abstract else "",
                                url=url,
                                doi=doi,
                                authors=authors,
                                publication_date=pub_date,
                                source_quality=self._calculate_semantic_scholar_quality(paper),
                                collection_timestamp=datetime.now(timezone.utc),
                                raw_metadata=paper,
                            )

                            sources.append(source)

                    else:
                        logger.warning(f"Semantic Scholar API returned status {response.status}")

        except Exception as e:
            logger.error(f"Semantic Scholar collection error: {e}")

        return sources

    def _calculate_crossref_quality(self, item: Dict) -> float:
        """Calculate quality score for CrossRef source"""
        score = 0.5  # Base score

        # Has abstract
        if item.get("abstract"):
            score += 0.2

        # Has authors
        if item.get("author"):
            score += 0.1

        # Has DOI
        if item.get("DOI"):
            score += 0.1

        # Recent publication
        if "published-print" in item:
            date_parts = item["published-print"].get("date-parts", [])
            if date_parts and len(date_parts[0]) > 0:
                year = date_parts[0][0]
                if year >= 2015:
                    score += 0.1

        return min(score, 1.0)

    def _calculate_openalex_quality(self, item: Dict) -> float:
        """Calculate quality score for OpenAlex source"""
        score = 0.5  # Base score

        # Has abstract
        if item.get("abstract"):
            score += 0.2

        # Has authors
        if item.get("authorships"):
            score += 0.1

        # Has DOI
        if item.get("doi"):
            score += 0.1

        # Citation count
        citation_count = item.get("cited_by_count", 0)
        if citation_count > 10:
            score += 0.1

        return min(score, 1.0)

    def _calculate_semantic_scholar_quality(self, paper: Dict) -> float:
        """Calculate quality score for Semantic Scholar source"""
        score = 0.5  # Base score

        # Has abstract
        if paper.get("abstract"):
            score += 0.2

        # Has authors
        if paper.get("authors"):
            score += 0.1

        # Has DOI
        external_ids = paper.get("externalIds", {})
        if external_ids.get("DOI"):
            score += 0.1

        # Recent publication
        year = paper.get("year")
        if year and year >= 2015:
            score += 0.1

        return min(score, 1.0)

    def _deduplicate_and_validate(
        self, sources: List[RealProcessMetadataSource]
    ) -> List[RealProcessMetadataSource]:
        """Remove duplicates and validate source quality"""

        # Deduplicate by DOI and title similarity
        unique_sources = []
        seen_dois = set()
        seen_titles = set()

        for source in sources:
            # Skip if DOI already seen
            if source.doi and source.doi in seen_dois:
                continue

            # Skip if title is very similar
            title_key = re.sub(r"[^\w\s]", "", source.title.lower())[:50]
            if title_key in seen_titles:
                continue

            # Validate source has minimum required fields
            if not source.title or source.title.lower() in ["no title", ""]:
                continue

            if not source.url:
                continue

            # Add to unique collection
            unique_sources.append(source)

            if source.doi:
                seen_dois.add(source.doi)
            seen_titles.add(title_key)

        # Sort by quality score
        unique_sources.sort(key=lambda x: x.source_quality, reverse=True)

        return unique_sources


class RealProcessMetadataManager:
    """Manager for real process metadata collection and storage"""

    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "process_metadata" / "real_process_metadata.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.collector = RealDataCollector()
        self.collection_results = {}

        self._initialize_database()

        logger.info("RealProcessMetadataManager initialized")

    def _initialize_database(self):
        """Initialize database for real process metadata storage"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Real sources table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS real_process_sources (
                    source_id TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    metadata_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    url TEXT NOT NULL,
                    doi TEXT,
                    authors TEXT,
                    publication_date TEXT,
                    source_quality REAL,
                    verification_status TEXT,
                    collection_timestamp TIMESTAMP,
                    raw_metadata TEXT
                )
            """
            )

            # Collection sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS collection_sessions (
                    session_id TEXT PRIMARY KEY,
                    metadata_type TEXT,
                    target_count INTEGER,
                    actual_count INTEGER,
                    success_rate REAL,
                    collection_time REAL,
                    session_timestamp TIMESTAMP,
                    session_status TEXT
                )
            """
            )

            conn.commit()

    async def collect_all_real_metadata(self, target_per_field: int = 100) -> Dict[str, Any]:
        """Collect real process metadata for all fields"""

        logger.info(
            f"Starting REAL process metadata collection (target: {target_per_field} per field)"
        )

        overall_start_time = time.time()
        collection_summary = {
            "fields_processed": 0,
            "fields_successful": 0,
            "total_real_sources": 0,
            "average_quality": 0.0,
            "collection_time": 0.0,
            "field_results": {},
            "overall_success": False,
        }

        # Collect for each metadata type
        for metadata_type in ProcessMetadataType:
            logger.info(f"Collecting REAL sources for: {metadata_type.value}")

            field_start_time = time.time()
            session_id = f"session_{metadata_type.value}_{int(time.time())}"

            try:
                # Attempt real data collection
                real_sources = await self.collector.collect_real_sources(
                    metadata_type,
                    target_count=target_per_field,
                    timeout_seconds=120,  # 2 minute timeout per field
                )

                # Store real sources in database
                self._store_real_sources(real_sources)

                field_time = time.time() - field_start_time

                # Calculate field metrics
                quality_scores = [s.source_quality for s in real_sources]
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

                success_rate = len(real_sources) / target_per_field

                field_result = {
                    "target_count": target_per_field,
                    "actual_count": len(real_sources),
                    "success_rate": success_rate,
                    "average_quality": avg_quality,
                    "collection_time": field_time,
                    "status": (
                        "SUCCESS"
                        if len(real_sources) >= target_per_field * 0.5
                        else "INSUFFICIENT_DATA"
                    ),
                    "platforms_used": list(set(s.platform.value for s in real_sources)),
                    "quality_distribution": {
                        "high": sum(1 for q in quality_scores if q >= 0.8),
                        "medium": sum(1 for q in quality_scores if 0.6 <= q < 0.8),
                        "low": sum(1 for q in quality_scores if q < 0.6),
                    },
                }

                # Store session record
                self._store_collection_session(session_id, metadata_type, field_result)

                collection_summary["field_results"][metadata_type.value] = field_result
                collection_summary["fields_processed"] += 1
                collection_summary["total_real_sources"] += len(real_sources)

                if field_result["status"] == "SUCCESS":
                    collection_summary["fields_successful"] += 1

                logger.info(
                    f"✓ {metadata_type.value}: {len(real_sources)} REAL sources collected ({success_rate:.1%} success)"
                )

            except Exception as e:
                logger.error(f"✗ {metadata_type.value}: FAILED - {e}")

                field_result = {
                    "target_count": target_per_field,
                    "actual_count": 0,
                    "success_rate": 0.0,
                    "status": "FAILED",
                    "error": str(e),
                }

                collection_summary["field_results"][metadata_type.value] = field_result
                collection_summary["fields_processed"] += 1

        # Calculate overall metrics
        overall_time = time.time() - overall_start_time
        collection_summary["collection_time"] = overall_time

        if collection_summary["total_real_sources"] > 0:
            # Calculate average quality across all sources
            all_sources = self._get_all_stored_sources()
            quality_scores = [s["source_quality"] for s in all_sources if s["source_quality"]]
            collection_summary["average_quality"] = (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            )

        # Determine overall success
        success_threshold = 0.5  # At least 50% of fields must succeed
        success_rate = (
            collection_summary["fields_successful"] / collection_summary["fields_processed"]
        )
        collection_summary["overall_success"] = success_rate >= success_threshold

        # Save comprehensive results
        results_path = (
            self.base_path
            / "process_metadata"
            / f"real_collection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_path, "w") as f:
            json.dump(collection_summary, f, indent=2, default=str)

        logger.info(
            f"Real data collection completed: {collection_summary['fields_successful']}/{collection_summary['fields_processed']} fields successful"
        )
        logger.info(f"Total real sources: {collection_summary['total_real_sources']}")
        logger.info(f"Overall success: {collection_summary['overall_success']}")

        return collection_summary

    def _store_real_sources(self, sources: List[RealProcessMetadataSource]):
        """Store real sources in database"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for source in sources:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO real_process_sources
                    (source_id, platform, metadata_type, title, description, url, doi,
                     authors, publication_date, source_quality, verification_status,
                     collection_timestamp, raw_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        source.source_id,
                        source.platform.value,
                        source.metadata_type.value,
                        source.title,
                        source.description,
                        source.url,
                        source.doi,
                        json.dumps(source.authors),
                        source.publication_date,
                        source.source_quality,
                        source.verification_status,
                        source.collection_timestamp,
                        json.dumps(source.raw_metadata),
                    ),
                )

            conn.commit()

    def _store_collection_session(
        self, session_id: str, metadata_type: ProcessMetadataType, result: Dict
    ):
        """Store collection session record"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO collection_sessions
                (session_id, metadata_type, target_count, actual_count, success_rate,
                 collection_time, session_timestamp, session_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    metadata_type.value,
                    result["target_count"],
                    result["actual_count"],
                    result["success_rate"],
                    result["collection_time"],
                    datetime.now(timezone.utc),
                    result["status"],
                ),
            )

            conn.commit()

    def _get_all_stored_sources(self) -> List[Dict]:
        """Get all stored sources from database"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT source_id, platform, metadata_type, title, description, url, doi,
                       authors, publication_date, source_quality, verification_status,
                       collection_timestamp
                FROM real_process_sources
                ORDER BY source_quality DESC
            """
            )

            rows = cursor.fetchall()

        sources = []
        for row in rows:
            sources.append(
                {
                    "source_id": row[0],
                    "platform": row[1],
                    "metadata_type": row[2],
                    "title": row[3],
                    "description": row[4],
                    "url": row[5],
                    "doi": row[6],
                    "authors": json.loads(row[7]) if row[7] else [],
                    "publication_date": row[8],
                    "source_quality": row[9],
                    "verification_status": row[10],
                    "collection_timestamp": row[11],
                }
            )

        return sources

    def generate_honest_report(self) -> Dict[str, Any]:
        """Generate honest, accurate assessment report"""

        all_sources = self._get_all_stored_sources()

        if not all_sources:
            return {
                "status": "NO_DATA",
                "message": "No real data has been collected",
                "recommendations": [
                    "Fix API connectivity issues",
                    "Verify SSL certificates",
                    "Check search strategies",
                ],
            }

        # Group by metadata type
        by_type = {}
        for source in all_sources:
            metadata_type = source["metadata_type"]
            if metadata_type not in by_type:
                by_type[metadata_type] = []
            by_type[metadata_type].append(source)

        # Analyze each type
        type_analysis = {}
        for metadata_type, sources in by_type.items():
            quality_scores = [s["source_quality"] for s in sources]

            type_analysis[metadata_type] = {
                "source_count": len(sources),
                "target_met": len(sources) >= 100,
                "average_quality": sum(quality_scores) / len(quality_scores),
                "platforms": list(set(s["platform"] for s in sources)),
                "quality_rating": self._assess_quality_level(
                    sum(quality_scores) / len(quality_scores)
                ),
            }

        # Overall assessment
        total_sources = len(all_sources)
        successful_types = sum(1 for analysis in type_analysis.values() if analysis["target_met"])
        total_types = len(ProcessMetadataType)

        overall_quality = sum(
            analysis["average_quality"] for analysis in type_analysis.values()
        ) / len(type_analysis)

        report = {
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_real_sources": total_sources,
            "metadata_types_processed": len(type_analysis),
            "successful_types": successful_types,
            "target_achievement_rate": successful_types / total_types,
            "overall_quality_score": overall_quality,
            "type_analysis": type_analysis,
            "honest_assessment": {
                "data_quality": self._assess_quality_level(overall_quality),
                "target_achievement": (
                    "EXCELLENT"
                    if successful_types >= 7
                    else "GOOD" if successful_types >= 5 else "INSUFFICIENT"
                ),
                "production_ready": successful_types >= 6 and overall_quality >= 0.7,
                "confidence_level": (
                    "HIGH"
                    if successful_types >= 7 and overall_quality >= 0.8
                    else "MEDIUM" if successful_types >= 4 else "LOW"
                ),
            },
            "recommendations": self._generate_recommendations(
                type_analysis, overall_quality, successful_types
            ),
        }

        return report

    def _assess_quality_level(self, score: float) -> str:
        """Assess quality level based on score"""
        if score >= 0.8:
            return "EXCELLENT"
        elif score >= 0.7:
            return "GOOD"
        elif score >= 0.6:
            return "ACCEPTABLE"
        else:
            return "NEEDS_IMPROVEMENT"

    def _generate_recommendations(
        self, type_analysis: Dict, overall_quality: float, successful_types: int
    ) -> List[str]:
        """Generate honest recommendations"""
        recommendations = []

        if successful_types < 6:
            recommendations.append("Increase successful data collection for more metadata types")

        if overall_quality < 0.7:
            recommendations.append("Improve source quality filters and validation")

        failed_types = [t for t, a in type_analysis.items() if not a["target_met"]]
        if failed_types:
            recommendations.append(f"Focus collection efforts on: {', '.join(failed_types)}")

        if overall_quality >= 0.8 and successful_types >= 7:
            recommendations.append("System meets production quality standards")
        else:
            recommendations.append(
                "Additional quality improvements needed before production deployment"
            )

        return recommendations


# Main execution function
async def main():
    """Main execution function for real process metadata collection"""
    try:
        logger.info("=" * 80)
        logger.info("REAL PROCESS METADATA COLLECTION SYSTEM")
        logger.info("=" * 80)

        # Initialize real data manager
        manager = RealProcessMetadataManager()

        # Collect real data
        results = await manager.collect_all_real_metadata(target_per_field=100)

        # Generate honest report
        report = manager.generate_honest_report()

        # Print honest summary
        print("\n" + "=" * 80)
        print("[LAB] REAL PROCESS METADATA COLLECTION RESULTS")
        print("=" * 80)
        print(f"[DATA] Total Real Sources Collected: {results['total_real_sources']}")
        print(
            f"[DATA] Fields Successful: {results['fields_successful']}/{results['fields_processed']}"
        )
        print(
            f"[DATA] Overall Success Rate: {results['fields_successful']/results['fields_processed']:.1%}"
        )
        print(f"[DATA] Average Quality Score: {results['average_quality']:.3f}")
        print(f"[DATA] Collection Time: {results['collection_time']:.1f} seconds")
        print(f"[DATA] Overall Success: {results['overall_success']}")

        print(f"\n[TARGET] HONEST ASSESSMENT:")
        print(f"   Data Quality: {report['honest_assessment']['data_quality']}")
        print(f"   Target Achievement: {report['honest_assessment']['target_achievement']}")
        print(f"   Production Ready: {report['honest_assessment']['production_ready']}")
        print(f"   Confidence Level: {report['honest_assessment']['confidence_level']}")

        print(f"\n[IDEA] RECOMMENDATIONS:")
        for recommendation in report["recommendations"]:
            print(f"   • {recommendation}")

        print("=" * 80)

        return {
            "collection_results": results,
            "assessment_report": report,
            "status": "COMPLETED_WITH_HONEST_REPORTING",
        }

    except Exception as e:
        logger.error(f"Real process metadata collection failed: {e}")
        print(f"\n[FAIL] COLLECTION FAILED: {e}")
        return {"status": "FAILED", "error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
