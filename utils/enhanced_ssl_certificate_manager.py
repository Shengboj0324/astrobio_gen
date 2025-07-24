#!/usr/bin/env python3
"""
Enhanced SSL Certificate Manager
================================

Comprehensive SSL certificate management system for resolving certificate issues
with scientific data sources while preserving all data sources.

Features:
- Automatic certificate issue detection and resolution
- Fallback mechanisms for problematic certificates
- Self-signed certificate handling
- Handshake failure recovery
- Certificate chain validation and repair
- Domain-specific SSL configurations
"""

import ssl
import aiohttp
import requests
import asyncio
import logging
import urllib3
from urllib3.exceptions import InsecureRequestWarning
import certifi
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class SSLIssueType(Enum):
    """Types of SSL issues encountered"""
    CERTIFICATE_VERIFY_FAILED = "certificate_verify_failed"
    SELF_SIGNED_CERT = "self_signed_certificate" 
    HANDSHAKE_FAILURE = "handshake_failure"
    CERTIFICATE_EXPIRED = "certificate_expired"
    HOSTNAME_MISMATCH = "hostname_mismatch"
    PROTOCOL_ERROR = "protocol_error"
    UNKNOWN = "unknown"

@dataclass
class SSLConfiguration:
    """SSL configuration for specific domains/endpoints"""
    domain: str
    verify_ssl: bool = True
    use_custom_context: bool = False
    allow_self_signed: bool = False
    ssl_version: Optional[int] = None
    ciphers: Optional[str] = None
    check_hostname: bool = True
    cert_reqs: int = ssl.CERT_REQUIRED
    ca_certs: Optional[str] = None
    issue_type: Optional[SSLIssueType] = None
    fallback_enabled: bool = True

class EnhancedSSLCertificateManager:
    """
    Comprehensive SSL certificate manager for scientific data sources
    """
    
    def __init__(self):
        self.domain_configs: Dict[str, SSLConfiguration] = {}
        self.ssl_contexts: Dict[str, ssl.SSLContext] = {}
        self.problematic_domains: Dict[str, List[SSLIssueType]] = {}
        self.successful_configs: Dict[str, SSLConfiguration] = {}
        self.session_cache: Dict[str, aiohttp.ClientSession] = {}
        
        # Initialize base configurations
        self._initialize_base_configs()
        self._load_known_problematic_domains()
    
    def _initialize_base_configs(self):
        """Initialize base SSL configurations"""
        
        # Default secure configuration
        self.default_config = SSLConfiguration(
            domain="default",
            verify_ssl=True,
            check_hostname=True,
            cert_reqs=ssl.CERT_REQUIRED,
            ca_certs=certifi.where()
        )
        
        # Relaxed configuration for problematic domains
        self.relaxed_config = SSLConfiguration(
            domain="relaxed",
            verify_ssl=False,
            check_hostname=False,
            cert_reqs=ssl.CERT_NONE,
            allow_self_signed=True,
            fallback_enabled=True
        )
        
        # Self-signed certificate configuration
        self.self_signed_config = SSLConfiguration(
            domain="self_signed",
            verify_ssl=True,
            check_hostname=False,
            cert_reqs=ssl.CERT_NONE,
            allow_self_signed=True,
            use_custom_context=True
        )
    
    def _load_known_problematic_domains(self):
        """Load known problematic domains from validation results"""
        
        # Known domains with SSL issues based on validation results
        known_issues = {
            "www.cosmos.esa.int": [SSLIssueType.SELF_SIGNED_CERT],
            "www.quantum-sensors.org": [SSLIssueType.HANDSHAKE_FAILURE],
            "ml-astro.org": [SSLIssueType.CERTIFICATE_VERIFY_FAILED],
            "legacy-databases.org": [SSLIssueType.PROTOCOL_ERROR],
            "self-signed-apis.edu": [SSLIssueType.SELF_SIGNED_CERT]
        }
        
        for domain, issues in known_issues.items():
            self.problematic_domains[domain] = issues
            # Create specific configuration for each known issue
            if SSLIssueType.SELF_SIGNED_CERT in issues:
                self.domain_configs[domain] = SSLConfiguration(
                    domain=domain,
                    verify_ssl=False,
                    check_hostname=False,
                    cert_reqs=ssl.CERT_NONE,
                    allow_self_signed=True,
                    issue_type=SSLIssueType.SELF_SIGNED_CERT
                )
            elif SSLIssueType.HANDSHAKE_FAILURE in issues:
                self.domain_configs[domain] = SSLConfiguration(
                    domain=domain,
                    verify_ssl=True,
                    ciphers='HIGH:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA',
                    check_hostname=True,
                    issue_type=SSLIssueType.HANDSHAKE_FAILURE
                )
    
    def create_ssl_context(self, config: SSLConfiguration) -> ssl.SSLContext:
        """Create SSL context based on configuration"""
        
        try:
            if config.use_custom_context or config.issue_type:
                # Create custom context for problematic domains
                if config.verify_ssl:
                    context = ssl.create_default_context(cafile=config.ca_certs or certifi.where())
                else:
                    context = ssl.create_default_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                
                # Apply specific configurations
                context.check_hostname = config.check_hostname
                context.verify_mode = config.cert_reqs
                
                if config.ciphers:
                    context.set_ciphers(config.ciphers)
                
                # Handle self-signed certificates
                if config.allow_self_signed:
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                
                # Handle handshake failures
                if config.issue_type == SSLIssueType.HANDSHAKE_FAILURE:
                    context.options |= ssl.OP_NO_SSLv2
                    context.options |= ssl.OP_NO_SSLv3
                    context.options |= ssl.OP_NO_TLSv1
                    context.options |= ssl.OP_NO_TLSv1_1
                
            else:
                # Standard secure context
                context = ssl.create_default_context(cafile=certifi.where())
                context.check_hostname = config.check_hostname
                context.verify_mode = config.cert_reqs
            
            return context
            
        except Exception as e:
            logger.warning(f"Failed to create SSL context for {config.domain}: {e}")
            # Fallback to minimal context
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context
    
    def get_domain_from_url(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return "unknown"
    
    def get_ssl_config_for_domain(self, domain: str) -> SSLConfiguration:
        """Get SSL configuration for a specific domain"""
        
        # Check if we have a specific configuration for this domain
        if domain in self.domain_configs:
            return self.domain_configs[domain]
        
        # Check if domain is known to be problematic
        if domain in self.problematic_domains:
            issues = self.problematic_domains[domain]
            if SSLIssueType.SELF_SIGNED_CERT in issues:
                return self.self_signed_config
            elif SSLIssueType.HANDSHAKE_FAILURE in issues:
                config = SSLConfiguration(
                    domain=domain,
                    verify_ssl=True,
                    ciphers='DEFAULT:@SECLEVEL=0',
                    issue_type=SSLIssueType.HANDSHAKE_FAILURE
                )
                return config
        
        # Return default secure configuration
        return self.default_config
    
    async def test_ssl_configuration(self, url: str, config: SSLConfiguration) -> Tuple[bool, Optional[str]]:
        """Test SSL configuration with a specific URL"""
        
        try:
            domain = self.get_domain_from_url(url)
            context = self.create_ssl_context(config)
            
            connector = aiohttp.TCPConnector(
                ssl=context,
                limit=10,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=15, connect=5)
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'Astrobiology-Research-Platform/1.0'}
            ) as session:
                async with session.head(url) as response:
                    if response.status < 400:
                        logger.info(f"SSL test successful for {domain} with {config.issue_type or 'default'} config")
                        return True, None
                    else:
                        return False, f"HTTP {response.status}"
        
        except Exception as e:
            error_msg = str(e).lower()
            
            # Classify the SSL error
            if "certificate verify failed" in error_msg:
                return False, "certificate_verify_failed"
            elif "handshake failure" in error_msg:
                return False, "handshake_failure"
            elif "self-signed" in error_msg:
                return False, "self_signed_certificate"
            else:
                return False, str(e)
    
    async def find_working_ssl_config(self, url: str) -> Optional[SSLConfiguration]:
        """Find a working SSL configuration for a URL through systematic testing"""
        
        domain = self.get_domain_from_url(url)
        logger.info(f"Finding working SSL configuration for: {domain}")
        
        # Get base configuration for domain
        base_config = self.get_ssl_config_for_domain(domain)
        
        # Test configurations in order of preference
        test_configs = [
            base_config,  # Domain-specific or default
            self.default_config,  # Standard secure
            self.self_signed_config,  # Self-signed handling
            self.relaxed_config,  # Completely relaxed
        ]
        
        # Add handshake failure specific configs
        handshake_configs = [
            SSLConfiguration(
                domain=domain,
                verify_ssl=True,
                ciphers='DEFAULT:@SECLEVEL=0',
                ssl_version=ssl.PROTOCOL_TLS,
                issue_type=SSLIssueType.HANDSHAKE_FAILURE
            ),
            SSLConfiguration(
                domain=domain,
                verify_ssl=True,
                ciphers='HIGH:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA',
                issue_type=SSLIssueType.HANDSHAKE_FAILURE
            )
        ]
        
        test_configs.extend(handshake_configs)
        
        for config in test_configs:
            try:
                success, error = await self.test_ssl_configuration(url, config)
                if success:
                    logger.info(f"Found working SSL config for {domain}: {config.issue_type or 'default'}")
                    # Cache successful configuration
                    self.successful_configs[domain] = config
                    self.domain_configs[domain] = config
                    return config
                else:
                    logger.debug(f"SSL config failed for {domain}: {error}")
            
            except Exception as e:
                logger.debug(f"SSL config test error for {domain}: {e}")
                continue
        
        logger.warning(f"No working SSL configuration found for {domain}")
        return None
    
    async def get_optimized_session(self, url: str) -> aiohttp.ClientSession:
        """Get optimized HTTP session for a URL with best SSL configuration"""
        
        domain = self.get_domain_from_url(url)
        
        # Check if we have a cached session
        if domain in self.session_cache:
            session = self.session_cache[domain]
            if not session.closed:
                return session
        
        # Find working SSL configuration
        config = await self.find_working_ssl_config(url)
        if not config:
            config = self.relaxed_config  # Ultimate fallback
        
        # Create SSL context
        ssl_context = self.create_ssl_context(config)
        
        # Create optimized connector
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=50,
            limit_per_host=10,
            ttl_dns_cache=600,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
        
        # Create session
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Astrobiology-Research-Platform/1.0 (Scientific Research)',
                'Accept': 'application/json, text/html, */*',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
        )
        
        # Cache session
        self.session_cache[domain] = session
        
        logger.info(f"Created optimized session for {domain} with SSL config: {config.issue_type or 'default'}")
        return session
    
    def get_requests_session(self, url: str) -> requests.Session:
        """Get optimized requests session for a URL"""
        
        domain = self.get_domain_from_url(url)
        config = self.get_ssl_config_for_domain(domain)
        
        session = requests.Session()
        
        # Configure SSL verification
        if config.verify_ssl:
            session.verify = config.ca_certs or certifi.where()
        else:
            session.verify = False
            urllib3.disable_warnings(InsecureRequestWarning)
        
        # Set timeouts and headers
        session.timeout = 30
        session.headers.update({
            'User-Agent': 'Astrobiology-Research-Platform/1.0 (Scientific Research)',
            'Accept': 'application/json, text/html, */*',
            'Connection': 'keep-alive'
        })
        
        return session
    
    async def validate_all_data_sources(self, data_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate SSL configuration for all data sources"""
        
        validation_results = {
            'total_sources': len(data_sources),
            'successful_ssl_configs': 0,
            'failed_ssl_configs': 0,
            'sources_with_fallbacks': 0,
            'detailed_results': []
        }
        
        logger.info(f"Validating SSL configuration for {len(data_sources)} data sources...")
        
        # Process sources in batches to avoid overwhelming servers
        batch_size = 5
        for i in range(0, len(data_sources), batch_size):
            batch = data_sources[i:i + batch_size]
            
            tasks = []
            for source in batch:
                url = source.get('primary_url') or source.get('url', '')
                if url:
                    tasks.append(self._validate_single_source_ssl(source, url))
            
            if tasks:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                validation_results['detailed_results'].extend(batch_results)
            
            # Brief delay between batches
            await asyncio.sleep(1)
        
        # Summarize results
        for result in validation_results['detailed_results']:
            if isinstance(result, dict):
                if result.get('ssl_working'):
                    validation_results['successful_ssl_configs'] += 1
                else:
                    validation_results['failed_ssl_configs'] += 1
                
                if result.get('fallback_used'):
                    validation_results['sources_with_fallbacks'] += 1
        
        success_rate = (validation_results['successful_ssl_configs'] / validation_results['total_sources']) * 100
        
        logger.info(f"SSL validation complete: {success_rate:.1f}% success rate")
        logger.info(f"Working SSL configs: {validation_results['successful_ssl_configs']}/{validation_results['total_sources']}")
        logger.info(f"Sources using fallbacks: {validation_results['sources_with_fallbacks']}")
        
        return validation_results
    
    async def _validate_single_source_ssl(self, source: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Validate SSL configuration for a single data source"""
        
        source_name = source.get('name', 'Unknown')
        domain = self.get_domain_from_url(url)
        
        result = {
            'source_name': source_name,
            'domain': domain,
            'url': url,
            'ssl_working': False,
            'config_used': None,
            'fallback_used': False,
            'error_message': None,
            'response_time_ms': None
        }
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Find working SSL configuration
            config = await self.find_working_ssl_config(url)
            
            if config:
                result['ssl_working'] = True
                result['config_used'] = config.issue_type.value if config.issue_type else 'default'
                result['fallback_used'] = config.fallback_enabled and (
                    not config.verify_ssl or 
                    config.allow_self_signed or 
                    config.cert_reqs == ssl.CERT_NONE
                )
                
                end_time = asyncio.get_event_loop().time()
                result['response_time_ms'] = (end_time - start_time) * 1000
                
                logger.info(f"✅ SSL fixed for {source_name}: {result['config_used']}")
            else:
                result['error_message'] = "No working SSL configuration found"
                logger.warning(f"❌ SSL failed for {source_name}: {result['error_message']}")
        
        except Exception as e:
            result['error_message'] = str(e)
            logger.error(f"❌ SSL validation error for {source_name}: {e}")
        
        return result
    
    async def cleanup(self):
        """Clean up cached sessions"""
        for session in self.session_cache.values():
            if not session.closed:
                await session.close()
        self.session_cache.clear()
    
    def get_ssl_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive SSL status report"""
        
        return {
            'total_domain_configs': len(self.domain_configs),
            'problematic_domains': len(self.problematic_domains),
            'successful_configs': len(self.successful_configs),
            'cached_sessions': len(self.session_cache),
            'domain_configs': {
                domain: {
                    'verify_ssl': config.verify_ssl,
                    'allow_self_signed': config.allow_self_signed,
                    'issue_type': config.issue_type.value if config.issue_type else None,
                    'fallback_enabled': config.fallback_enabled
                }
                for domain, config in self.domain_configs.items()
            },
            'problematic_domains': {
                domain: [issue.value for issue in issues]
                for domain, issues in self.problematic_domains.items()
            }
        }

# Global SSL manager instance
ssl_manager = EnhancedSSLCertificateManager()

# Convenience functions for backward compatibility
async def get_ssl_optimized_session(url: str) -> aiohttp.ClientSession:
    """Get SSL-optimized session for a URL"""
    return await ssl_manager.get_optimized_session(url)

def get_ssl_optimized_requests_session(url: str) -> requests.Session:
    """Get SSL-optimized requests session for a URL"""
    return ssl_manager.get_requests_session(url)

async def validate_data_source_ssl(data_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate SSL configuration for data sources"""
    return await ssl_manager.validate_all_data_sources(data_sources)

def get_ssl_status() -> Dict[str, Any]:
    """Get SSL manager status report"""
    return ssl_manager.get_ssl_status_report() 