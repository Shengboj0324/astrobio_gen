#!/usr/bin/env python3
"""
Enterprise-Grade URL Management System
======================================

Comprehensive URL resilience system for scientific data acquisition with:
- Multi-tier failover and mirror support
- Geographic routing and VPN handling
- Health monitoring and validation
- Predictive URL discovery
- Institution partnership integration
- Community-maintained URL registry

This replaces all hardcoded URLs throughout the astrobiology research platform
with a centralized, resilient, and intelligent URL management system.
"""

import asyncio
import aiohttp
import yaml
import logging
import time
import random
import json
import hashlib
import ipaddress
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

# Optional GeoIP imports
try:
    import geoip2.database
    import geoip2.errors
    GEOIP_AVAILABLE = True
except ImportError:
    GEOIP_AVAILABLE = False
    geoip2 = None

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import sqlite3
import threading
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class URLStatus(Enum):
    """URL status enumeration"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"

class GeographicRegion(Enum):
    """Geographic regions for routing optimization"""
    US_EAST = "us_east"
    US_WEST = "us_west"
    EUROPE = "europe"
    ASIA = "asia"
    GLOBAL = "global"
    
@dataclass
class URLHealthCheck:
    """URL health check configuration and results"""
    url: str
    endpoint: str = "/"
    expected_status: int = 200
    timeout_seconds: int = 30
    retry_count: int = 3
    last_checked: Optional[datetime] = None
    status: URLStatus = URLStatus.UNKNOWN
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    success_rate: float = 0.0

@dataclass 
class DataSourceRegistry:
    """Data source configuration from registry"""
    name: str
    domain: str
    primary_url: str
    mirror_urls: List[str] = field(default_factory=list)
    endpoints: Dict[str, str] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_check: Dict[str, Any] = field(default_factory=dict)
    geographic_routing: Dict[str, str] = field(default_factory=dict)
    authentication: Dict[str, Any] = field(default_factory=dict)
    last_verified: Optional[str] = None
    status: str = "unknown"

class URLManager:
    """
    Enterprise-grade URL management system with intelligent failover
    """
    
    def __init__(self, config_path: str = "config/data_sources", cache_ttl: int = 3600):
        self.config_path = Path(config_path)
        self.cache_ttl = cache_ttl
        self.url_cache = {}
        self.health_status = {}
        self.geographic_cache = {}
        self.community_registry = {}
        self.predictive_patterns = {}
        
        # Initialize database for URL tracking
        self.db_path = Path("data/metadata/url_registry.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        
        # Load all registries
        self.registries = self._load_all_registries()
        
        # Start health monitoring
        self.health_monitor_interval = 900  # 15 minutes
        self.monitoring_active = False
        
        # Geographic detection
        self.current_region = self._detect_geographic_region()
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"URLManager initialized with {len(self.registries)} data sources")
        logger.info(f"Detected geographic region: {self.current_region}")
    
    def _initialize_database(self):
        """Initialize URL tracking database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS url_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    url TEXT NOT NULL,
                    url_type TEXT NOT NULL, -- primary, mirror, community
                    status TEXT NOT NULL,
                    response_time_ms REAL,
                    last_checked TIMESTAMP,
                    consecutive_failures INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_name, url, url_type)
                );
                
                CREATE TABLE IF NOT EXISTS url_discovery (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    discovered_url TEXT NOT NULL,
                    discovery_method TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS geographic_routing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    region TEXT NOT NULL,
                    preferred_url TEXT NOT NULL,
                    performance_score REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_name, region)
                );
                
                CREATE TABLE IF NOT EXISTS community_contributions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    contributor_id TEXT NOT NULL,
                    url_suggestion TEXT NOT NULL,
                    validation_status TEXT DEFAULT 'pending',
                    votes_up INTEGER DEFAULT 0,
                    votes_down INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_url_health_source ON url_health(source_name);
                CREATE INDEX IF NOT EXISTS idx_url_health_status ON url_health(status);
                CREATE INDEX IF NOT EXISTS idx_discovery_source ON url_discovery(source_name);
                CREATE INDEX IF NOT EXISTS idx_geographic_source ON geographic_routing(source_name);
            """)
    
    def _load_all_registries(self) -> Dict[str, DataSourceRegistry]:
        """Load all data source registries from YAML files"""
        registries = {}
        
        # Core registry directories
        registry_dirs = [
            "core_registries",
            "mirrors", 
            "community_sources",
            "institutional_partners"
        ]
        
        for registry_dir in registry_dirs:
            registry_path = self.config_path / registry_dir
            if registry_path.exists():
                for yaml_file in registry_path.glob("*.yaml"):
                    try:
                        with open(yaml_file, 'r') as f:
                            data = yaml.safe_load(f)
                            
                        # Convert YAML data to DataSourceRegistry objects
                        for source_name, config in data.items():
                            registries[source_name] = DataSourceRegistry(
                                name=config.get('name', source_name),
                                domain=config.get('domain', 'unknown'),
                                primary_url=config.get('primary_url', ''),
                                mirror_urls=config.get('mirror_urls', []),
                                endpoints=config.get('endpoints', {}),
                                performance=config.get('performance', {}),
                                metadata=config.get('metadata', {}),
                                health_check=config.get('health_check', {}),
                                geographic_routing=config.get('geographic_routing', {}),
                                authentication=config.get('authentication', {}),
                                last_verified=config.get('last_verified'),
                                status=config.get('status', 'unknown')
                            )
                            
                    except Exception as e:
                        logger.error(f"Error loading registry {yaml_file}: {e}")
        
        return registries
    
    def _detect_geographic_region(self) -> GeographicRegion:
        """Detect current geographic region for routing optimization"""
        try:
            # Get public IP
            response = requests.get('https://httpbin.org/ip', timeout=10)
            ip = response.json()['origin'].split(',')[0].strip()
            
            # Try to determine region based on IP geolocation
            try:
                ip_obj = ipaddress.ip_address(ip)
                if ip_obj.is_private:
                    # Behind NAT/VPN, try alternative detection
                    return self._detect_region_alternative()
                
                # Use GeoIP if available, otherwise use simple heuristic
                if GEOIP_AVAILABLE:
                    # Use maxmind GeoIP database if available in production
                    # For now, fall back to simple heuristic
                    pass
                
                # Use a simple heuristic based on common IP ranges
                # In production, use maxmind GeoIP database or similar
                if ip.startswith(('129.', '130.', '131.')):  # Common US ranges
                    return GeographicRegion.US_EAST
                elif ip.startswith(('192.', '193.', '194.')):  # Common EU ranges  
                    return GeographicRegion.EUROPE
                else:
                    return GeographicRegion.GLOBAL
                    
            except ValueError:
                return GeographicRegion.GLOBAL
                
        except Exception as e:
            logger.debug(f"Geographic region detection failed: {e}")  # ✅ IMPROVED - Better error reporting
            return GeographicRegion.GLOBAL
    
    def _detect_region_alternative(self) -> GeographicRegion:
        """Alternative region detection for VPN/proxy scenarios"""
        try:
            # Try timezone-based detection
            import time
            tz_offset = time.timezone
            
            if -8 <= tz_offset <= -5:  # US timezones
                return GeographicRegion.US_EAST
            elif 0 <= tz_offset <= 3:  # European timezones
                return GeographicRegion.EUROPE
            elif 5 <= tz_offset <= 10:  # Asian timezones
                return GeographicRegion.ASIA
            else:
                return GeographicRegion.GLOBAL
                
        except:
            return GeographicRegion.GLOBAL
    
    async def get_optimal_url(self, source_name: str, endpoint: str = "", 
                            prefer_region: Optional[GeographicRegion] = None) -> Optional[str]:
        """
        Get the optimal URL for a data source with intelligent routing
        
        Args:
            source_name: Name of the data source
            endpoint: Specific endpoint path
            prefer_region: Preferred geographic region (overrides auto-detection)
            
        Returns:
            Optimal URL with failover support, or None if all sources failed
        """
        if source_name not in self.registries:
            logger.error(f"Unknown data source: {source_name}")
            return None
        
        registry = self.registries[source_name]
        region = prefer_region or self.current_region
        
        # Build candidate URLs with priority
        candidates = self._build_candidate_urls(registry, region, endpoint)
        
        # Test candidates in priority order
        for priority, url in candidates:
            if await self._test_url_health(url, registry.health_check):
                # Cache successful URL
                self._cache_successful_url(source_name, url, region)
                return url
        
        # All primary candidates failed, try predictive discovery
        discovered_url = await self._discover_alternative_url(source_name, endpoint)
        if discovered_url:
            return discovered_url
        
        # Ultimate fallback: try community-contributed URLs
        community_url = await self._try_community_urls(source_name, endpoint)
        if community_url:
            return community_url
        
        logger.error(f"All URL candidates failed for {source_name}")
        return None
    
    def _build_candidate_urls(self, registry: DataSourceRegistry, 
                            region: GeographicRegion, endpoint: str) -> List[Tuple[int, str]]:
        """Build prioritized list of candidate URLs"""
        candidates = []
        
        # Priority 1: Region-specific preference
        if region.value in registry.geographic_routing:
            preferred_type = registry.geographic_routing[region.value]
            if preferred_type == "primary_url":
                url = self._build_full_url(registry.primary_url, endpoint)
                candidates.append((1, url))
            elif preferred_type.startswith("mirror_urls["):
                # Extract mirror index
                idx = int(preferred_type.split('[')[1].split(']')[0])
                if idx < len(registry.mirror_urls):
                    url = self._build_full_url(registry.mirror_urls[idx], endpoint)
                    candidates.append((1, url))
        
        # Priority 2: Primary URL (if not already added)
        primary_url = self._build_full_url(registry.primary_url, endpoint)
        if not any(url == primary_url for _, url in candidates):
            candidates.append((2, primary_url))
        
        # Priority 3: Mirror URLs
        for i, mirror_url in enumerate(registry.mirror_urls):
            url = self._build_full_url(mirror_url, endpoint)
            if not any(existing_url == url for _, existing_url in candidates):
                candidates.append((3 + i, url))
        
        # Priority N: Cached successful URLs for this region
        cached_url = self._get_cached_url(registry.name, region)
        if cached_url and not any(url == cached_url for _, url in candidates):
            candidates.append((10, cached_url))
        
        return sorted(candidates, key=lambda x: x[0])
    
    def _build_full_url(self, base_url: str, endpoint: str) -> str:
        """Build full URL from base URL and endpoint"""
        if not endpoint:
            return base_url
        
        # Handle absolute URLs in endpoint
        if endpoint.startswith('http'):
            return endpoint
            
        # Join base URL with endpoint
        return urljoin(base_url.rstrip('/') + '/', endpoint.lstrip('/'))
    
    async def _test_url_health(self, url: str, health_config: Dict[str, Any]) -> bool:
        """Test URL health with comprehensive validation"""
        try:
            timeout = health_config.get('timeout_seconds', 30)
            expected_status = health_config.get('expected_status', 200)
            
            start_time = time.time()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    # Basic status check
                    if response.status == expected_status:
                        # Additional content validation if specified
                        if 'content_validation' in health_config:
                            content = await response.text()
                            if not self._validate_content(content, health_config['content_validation']):
                                return False
                        
                        # Update health status
                        self._update_url_health(url, URLStatus.ACTIVE, response_time)
                        return True
                    else:
                        self._update_url_health(url, URLStatus.FAILED, response_time, 
                                              f"HTTP {response.status}")
                        return False
                        
        except asyncio.TimeoutError:
            self._update_url_health(url, URLStatus.FAILED, None, "Timeout")
            return False
        except Exception as e:
            self._update_url_health(url, URLStatus.FAILED, None, str(e))
            return False
    
    def _validate_content(self, content: str, validation_config: Dict[str, Any]) -> bool:
        """Validate URL content based on configuration"""
        try:
            # Check for required keywords
            if 'required_keywords' in validation_config:
                for keyword in validation_config['required_keywords']:
                    if keyword not in content:
                        return False
            
            # Check for forbidden keywords
            if 'forbidden_keywords' in validation_config:
                for keyword in validation_config['forbidden_keywords']:
                    if keyword in content:
                        return False
            
            # Check content length
            if 'min_length' in validation_config:
                if len(content) < validation_config['min_length']:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Content validation error: {e}")
            return False
    
    def _update_url_health(self, url: str, status: URLStatus, 
                          response_time: Optional[float], error_message: Optional[str] = None):
        """Update URL health status in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO url_health 
                    (source_name, url, url_type, status, response_time_ms, 
                     last_checked, error_message, consecutive_failures)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 
                           CASE WHEN ? = 'ACTIVE' THEN 0 
                                ELSE COALESCE((SELECT consecutive_failures FROM url_health 
                                             WHERE url = ? LIMIT 1), 0) + 1 END)
                """, (
                    self._extract_source_name_from_url(url),
                    url,
                    self._determine_url_type(url),
                    status.value,
                    response_time,
                    datetime.now(timezone.utc),
                    error_message,
                    status.value,
                    url
                ))
                
        except Exception as e:
            logger.error(f"Error updating URL health: {e}")
    
    def _extract_source_name_from_url(self, url: str) -> str:
        """Extract source name from URL by matching against registries"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        for source_name, registry in self.registries.items():
            # Check primary URL
            if domain in registry.primary_url.lower():
                return source_name
            
            # Check mirror URLs
            for mirror_url in registry.mirror_urls:
                if domain in mirror_url.lower():
                    return source_name
        
        return "unknown"
    
    def _determine_url_type(self, url: str) -> str:
        """Determine if URL is primary, mirror, or community"""
        for source_name, registry in self.registries.items():
            if url.startswith(registry.primary_url):
                return "primary"
            
            for mirror_url in registry.mirror_urls:
                if url.startswith(mirror_url):
                    return "mirror"
        
        return "community"
    
    def _cache_successful_url(self, source_name: str, url: str, region: GeographicRegion):
        """Cache successful URL for future use"""
        with self.lock:
            cache_key = f"{source_name}_{region.value}"
            self.url_cache[cache_key] = {
                'url': url,
                'timestamp': datetime.now(timezone.utc),
                'ttl': self.cache_ttl
            }
    
    def _get_cached_url(self, source_name: str, region: GeographicRegion) -> Optional[str]:
        """Get cached URL if still valid"""
        with self.lock:
            cache_key = f"{source_name}_{region.value}"
            if cache_key in self.url_cache:
                cached = self.url_cache[cache_key]
                age = (datetime.now(timezone.utc) - cached['timestamp']).total_seconds()
                if age < cached['ttl']:
                    return cached['url']
                else:
                    # Remove expired cache
                    del self.url_cache[cache_key]
        
        return None
    
    async def _discover_alternative_url(self, source_name: str, endpoint: str) -> Optional[str]:
        """Predictive URL discovery using pattern matching"""
        if source_name not in self.registries:
            return None
        
        registry = self.registries[source_name]
        base_domain = urlparse(registry.primary_url).netloc
        
        # Common URL patterns to try
        patterns = [
            f"https://archive.{base_domain}",
            f"https://data.{base_domain}",
            f"https://mirror.{base_domain}",
            f"https://backup.{base_domain}",
            f"https://www2.{base_domain}",
            f"https://{base_domain.replace('www.', '')}",
            f"https://api.{base_domain}",
        ]
        
        # Institution-specific patterns
        if 'nasa' in base_domain:
            patterns.extend([
                f"https://archive.stsci.edu",
                f"https://mast.stsci.edu",
                f"https://data.nasa.gov"
            ])
        elif 'ncbi' in base_domain:
            patterns.extend([
                f"https://ftp-private.ncbi.nlm.nih.gov",
                f"https://ftp.ncbi.nih.gov"
            ])
        
        # Test patterns
        for pattern in patterns:
            test_url = self._build_full_url(pattern, endpoint)
            if await self._test_url_health(test_url, registry.health_check):
                # Log discovery for community contribution
                self._log_url_discovery(source_name, test_url, "pattern_matching", 0.8)
                return test_url
        
        return None
    
    def _log_url_discovery(self, source_name: str, url: str, method: str, confidence: float):
        """Log discovered URL for community validation"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO url_discovery 
                    (source_name, discovered_url, discovery_method, confidence_score)
                    VALUES (?, ?, ?, ?)
                """, (source_name, url, method, confidence))
                
        except Exception as e:
            logger.error(f"Error logging URL discovery: {e}")
    
    async def _try_community_urls(self, source_name: str, endpoint: str) -> Optional[str]:
        """Try community-contributed URLs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT url_suggestion, votes_up, votes_down 
                    FROM community_contributions 
                    WHERE source_name = ? AND validation_status = 'approved'
                    ORDER BY (votes_up - votes_down) DESC
                    LIMIT 5
                """, (source_name,))
                
                for row in cursor:
                    community_url = self._build_full_url(row[0], endpoint)
                    if source_name in self.registries:
                        health_config = self.registries[source_name].health_check
                        if await self._test_url_health(community_url, health_config):
                            return community_url
        
        except Exception as e:
            logger.error(f"Error trying community URLs: {e}")
        
        return None
    
    async def start_health_monitoring(self):
        """Start continuous health monitoring of all URLs"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        async def monitor_loop():
            while self.monitoring_active:
                try:
                    await self._run_health_checks()
                    await asyncio.sleep(self.health_monitor_interval)
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute on error
        
        # Start monitoring in background
        asyncio.create_task(monitor_loop())
        logger.info("URL health monitoring started")
    
    async def _run_health_checks(self):
        """Run health checks on all registered URLs"""
        tasks = []
        
        for source_name, registry in self.registries.items():
            # Check primary URL
            primary_url = registry.primary_url
            health_config = registry.health_check
            tasks.append(self._test_url_health(primary_url, health_config))
            
            # Check mirror URLs
            for mirror_url in registry.mirror_urls:
                tasks.append(self._test_url_health(mirror_url, health_config))
        
        # Execute all health checks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log overall health status
        total_checks = len(tasks)
        successful_checks = sum(1 for result in results if result is True)
        success_rate = successful_checks / total_checks if total_checks > 0 else 0
        
        logger.info(f"Health check completed: {successful_checks}/{total_checks} "
                   f"URLs healthy ({success_rate:.1%})")
    
    def stop_health_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        logger.info("URL health monitoring stopped")
    
    def get_health_status(self, source_name: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for all sources or specific source"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if source_name:
                    cursor = conn.execute("""
                        SELECT url, url_type, status, response_time_ms, 
                               last_checked, consecutive_failures, error_message
                        FROM url_health 
                        WHERE source_name = ?
                        ORDER BY url_type, consecutive_failures
                    """, (source_name,))
                else:
                    cursor = conn.execute("""
                        SELECT source_name, url, url_type, status, response_time_ms,
                               last_checked, consecutive_failures, error_message
                        FROM url_health 
                        ORDER BY source_name, url_type, consecutive_failures
                    """)
                
                results = cursor.fetchall()
                
                if source_name:
                    return {
                        'source_name': source_name,
                        'urls': [
                            {
                                'url': row[0],
                                'type': row[1], 
                                'status': row[2],
                                'response_time_ms': row[3],
                                'last_checked': row[4],
                                'consecutive_failures': row[5],
                                'error_message': row[6]
                            }
                            for row in results
                        ]
                    }
                else:
                    # Group by source
                    grouped = {}
                    for row in results:
                        source = row[0]
                        if source not in grouped:
                            grouped[source] = []
                        grouped[source].append({
                            'url': row[1],
                            'type': row[2],
                            'status': row[3],
                            'response_time_ms': row[4],
                            'last_checked': row[5],
                            'consecutive_failures': row[6],
                            'error_message': row[7]
                        })
                    
                    return grouped
                    
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {}
    
    def add_community_url(self, source_name: str, url: str, contributor_id: str) -> bool:
        """Add community-contributed URL"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO community_contributions 
                    (source_name, contributor_id, url_suggestion)
                    VALUES (?, ?, ?)
                """, (source_name, contributor_id, url))
            
            logger.info(f"Community URL added for {source_name}: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding community URL: {e}")
            return False
    
    def get_registry_info(self, source_name: str) -> Optional[Dict[str, Any]]:
        """Get complete registry information for a data source"""
        if source_name not in self.registries:
            return None
        
        registry = self.registries[source_name]
        return {
            'name': registry.name,
            'domain': registry.domain,
            'primary_url': registry.primary_url,
            'mirror_urls': registry.mirror_urls,
            'endpoints': registry.endpoints,
            'performance': registry.performance,
            'metadata': registry.metadata,
            'health_check': registry.health_check,
            'geographic_routing': registry.geographic_routing,
            'authentication': registry.authentication,
            'last_verified': registry.last_verified,
            'status': registry.status
        }
    
    def list_available_sources(self) -> List[str]:
        """List all available data sources"""
        return list(self.registries.keys())
    
    async def validate_all_sources(self) -> Dict[str, Dict[str, Any]]:
        """Validate all data sources and return comprehensive report"""
        validation_results = {}
        
        for source_name in self.registries.keys():
            try:
                # Test primary URL
                primary_url = await self.get_optimal_url(source_name)
                primary_valid = primary_url is not None
                
                # Test mirrors
                registry = self.registries[source_name]
                mirror_results = []
                
                for mirror_url in registry.mirror_urls:
                    mirror_valid = await self._test_url_health(mirror_url, registry.health_check)
                    mirror_results.append({
                        'url': mirror_url,
                        'valid': mirror_valid
                    })
                
                validation_results[source_name] = {
                    'primary_url_valid': primary_valid,
                    'optimal_url': primary_url,
                    'mirror_results': mirror_results,
                    'total_mirrors': len(registry.mirror_urls),
                    'working_mirrors': sum(1 for m in mirror_results if m['valid']),
                    'overall_status': 'healthy' if primary_valid or any(m['valid'] for m in mirror_results) else 'failed'
                }
                
            except Exception as e:
                validation_results[source_name] = {
                    'error': str(e),
                    'overall_status': 'error'
                }
        
        return validation_results

# Global URL manager instance
url_manager = None

def get_url_manager() -> URLManager:
    """Get global URL manager instance"""
    global url_manager
    if url_manager is None:
        url_manager = URLManager()
    return url_manager

# Convenience functions for backward compatibility
async def get_data_source_url(source_name: str, endpoint: str = "") -> Optional[str]:
    """Get optimal URL for a data source (backward compatible)"""
    manager = get_url_manager()
    return await manager.get_optimal_url(source_name, endpoint)

def start_url_monitoring():
    """Start URL health monitoring"""
    manager = get_url_manager()
    asyncio.create_task(manager.start_health_monitoring())

if __name__ == "__main__":
    # Test the URL manager
    async def test_url_manager():
        manager = URLManager()
        await manager.start_health_monitoring()
        
        # Test some sources
        test_sources = ['nasa_exoplanet_archive', 'kegg_database', 'ncbi_databases']
        
        for source in test_sources:
            url = await manager.get_optimal_url(source)
            print(f"{source}: {url}")
        
        # Get health status
        health = manager.get_health_status()
        print(f"Health status: {len(health)} sources monitored")
        
        manager.stop_health_monitoring()
    
    asyncio.run(test_url_manager()) 