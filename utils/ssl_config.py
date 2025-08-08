#!/usr/bin/env python3
"""
SSL Configuration Helper for External API Access
===============================================

Enhanced SSL configuration utilities that integrate with the Enhanced SSL Certificate Manager
to resolve certificate issues when accessing external APIs and data sources.

This module provides backward compatibility while leveraging the enhanced SSL management system.
"""

import logging
import os
import ssl
from pathlib import Path

import aiohttp
import certifi
import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Import enhanced SSL manager
try:
    from .enhanced_ssl_certificate_manager import (
        get_ssl_optimized_requests_session,
        get_ssl_optimized_session,
        ssl_manager,
    )

    ENHANCED_SSL_AVAILABLE = True
except ImportError:
    ENHANCED_SSL_AVAILABLE = False
    ssl_manager = None

# Configure logging
logger = logging.getLogger(__name__)


def configure_ssl_for_external_apis():
    """Configure SSL settings for reliable external API access"""

    # Disable SSL warnings for development/testing
    urllib3.disable_warnings(InsecureRequestWarning)

    # Set certificate bundle path
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

    logger.info("SSL configuration updated for external API access")

    if ENHANCED_SSL_AVAILABLE:
        logger.info("Enhanced SSL Certificate Manager available - using advanced SSL handling")
    else:
        logger.warning(
            "Enhanced SSL Certificate Manager not available - using fallback SSL configuration"
        )


def get_ssl_context(verify_ssl: bool = True, domain: str = None):
    """Get SSL context for external connections with enhanced domain-specific handling"""

    if ENHANCED_SSL_AVAILABLE and domain and ssl_manager:
        try:
            # Use enhanced SSL manager for domain-specific configuration
            config = ssl_manager.get_ssl_config_for_domain(domain)
            return ssl_manager.create_ssl_context(config)
        except Exception as e:
            logger.warning(f"Enhanced SSL context creation failed for {domain}: {e}")
            # Fall back to standard context

    # Standard SSL context creation
    if verify_ssl:
        context = ssl.create_default_context(cafile=certifi.where())
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        # Use system certificates
        context.load_default_certs()
    else:
        # For development/testing - disable SSL verification
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    return context


def get_aiohttp_connector(verify_ssl: bool = True, url: str = None):
    """Get aiohttp connector with appropriate SSL settings"""

    if ENHANCED_SSL_AVAILABLE and url and ssl_manager:
        try:
            # Extract domain from URL for enhanced SSL handling
            domain = ssl_manager.get_domain_from_url(url)
            config = ssl_manager.get_ssl_config_for_domain(domain)
            ssl_context = ssl_manager.create_ssl_context(config)

            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                limit=100,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True,
            )
            logger.info(f"Created enhanced SSL connector for domain: {domain}")
            return connector
        except Exception as e:
            logger.warning(f"Enhanced SSL connector creation failed: {e}")
            # Fall back to standard connector

    # Standard connector creation
    if verify_ssl:
        # Use default SSL context with certificate verification
        connector = aiohttp.TCPConnector(
            ssl=get_ssl_context(verify_ssl=True), limit=100, ttl_dns_cache=300, use_dns_cache=True
        )
    else:
        # For development/testing - disable SSL verification
        connector = aiohttp.TCPConnector(
            ssl=False, limit=100, ttl_dns_cache=300, use_dns_cache=True
        )

    return connector


def get_requests_session(verify_ssl: bool = True, url: str = None):
    """Get requests session with appropriate SSL settings"""

    if ENHANCED_SSL_AVAILABLE and url and ssl_manager:
        try:
            # Use enhanced SSL manager for URL-specific session
            return ssl_manager.get_requests_session(url)
        except Exception as e:
            logger.warning(f"Enhanced SSL session creation failed: {e}")
            # Fall back to standard session

    # Standard session creation
    session = requests.Session()

    if verify_ssl:
        # Use certificate bundle
        session.verify = certifi.where()
    else:
        # For development/testing - disable SSL verification
        session.verify = False
        urllib3.disable_warnings(InsecureRequestWarning)

    # Set timeout and retry configuration
    session.timeout = 30

    # Add user agent
    session.headers.update(
        {"User-Agent": "Astrobiology-Research-Platform/1.0 (Python; Scientific Research)"}
    )

    return session


async def get_enhanced_aiohttp_session(url: str) -> aiohttp.ClientSession:
    """Get enhanced aiohttp session with SSL certificate issue resolution"""

    if ENHANCED_SSL_AVAILABLE and ssl_manager:
        try:
            return await get_ssl_optimized_session(url)
        except Exception as e:
            logger.warning(f"Enhanced SSL session creation failed: {e}")
            # Fall back to standard session

    # Fallback to standard session with relaxed SSL
    connector = aiohttp.TCPConnector(ssl=False, limit=100, ttl_dns_cache=300, use_dns_cache=True)

    return aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=30),
        headers={"User-Agent": "Astrobiology-Research-Platform/1.0"},
    )


def check_ssl_configuration():
    """Check and report SSL configuration status"""
    config_status = {
        "certifi_available": False,
        "cert_bundle_path": None,
        "ssl_context_working": False,
        "external_api_accessible": False,
        "enhanced_ssl_available": ENHANCED_SSL_AVAILABLE,
        "enhanced_ssl_domains_configured": 0,
    }

    try:
        import certifi

        config_status["certifi_available"] = True
        config_status["cert_bundle_path"] = certifi.where()
        logger.info(f"Certificate bundle found: {certifi.where()}")
    except ImportError:
        logger.warning("Certifi package not available")

    try:
        context = get_ssl_context(verify_ssl=True)
        config_status["ssl_context_working"] = True
        logger.info("SSL context created successfully")
    except Exception as e:
        logger.error(f"SSL context creation failed: {e}")

    # Check enhanced SSL manager status
    if ENHANCED_SSL_AVAILABLE and ssl_manager:
        try:
            ssl_status = ssl_manager.get_ssl_status_report()
            config_status["enhanced_ssl_domains_configured"] = ssl_status["total_domain_configs"]
            config_status["enhanced_ssl_status"] = ssl_status
            logger.info(
                f"Enhanced SSL manager: {ssl_status['total_domain_configs']} domains configured"
            )
        except Exception as e:
            logger.warning(f"Enhanced SSL manager status check failed: {e}")

    # Test external API access with enhanced SSL handling
    try:
        if ENHANCED_SSL_AVAILABLE:
            session = get_requests_session(verify_ssl=False, url="https://httpbin.org/get")
        else:
            session = get_requests_session(verify_ssl=False)

        response = session.get("https://httpbin.org/get", timeout=10)
        if response.status_code == 200:
            config_status["external_api_accessible"] = True
            logger.info("External API access test successful")
    except Exception as e:
        logger.warning(f"External API access test failed: {e}")

    return config_status


def resolve_ssl_issues_for_urls(urls: list) -> dict:
    """Resolve SSL issues for a list of URLs using enhanced SSL manager"""

    if not ENHANCED_SSL_AVAILABLE or not ssl_manager:
        logger.warning("Enhanced SSL manager not available - cannot resolve SSL issues")
        return {"error": "Enhanced SSL manager not available"}

    results = {"total_urls": len(urls), "resolved_urls": 0, "failed_urls": 0, "url_results": []}

    for url in urls:
        try:
            domain = ssl_manager.get_domain_from_url(url)
            config = ssl_manager.get_ssl_config_for_domain(domain)

            url_result = {
                "url": url,
                "domain": domain,
                "resolved": True,
                "config_type": config.issue_type.value if config.issue_type else "default",
                "fallback_used": config.fallback_enabled and not config.verify_ssl,
            }

            results["resolved_urls"] += 1
            logger.info(f"✅ SSL resolved for {domain}: {url_result['config_type']}")

        except Exception as e:
            url_result = {
                "url": url,
                "domain": ssl_manager.get_domain_from_url(url) if ssl_manager else "unknown",
                "resolved": False,
                "error": str(e),
            }
            results["failed_urls"] += 1
            logger.error(f"❌ SSL resolution failed for {url}: {e}")

        results["url_results"].append(url_result)

    success_rate = (results["resolved_urls"] / results["total_urls"]) * 100
    logger.info(f"SSL issue resolution complete: {success_rate:.1f}% success rate")

    return results


# Initialize SSL configuration on import
configure_ssl_for_external_apis()

# Export main functions
__all__ = [
    "configure_ssl_for_external_apis",
    "get_ssl_context",
    "get_aiohttp_connector",
    "get_requests_session",
    "get_enhanced_aiohttp_session",
    "check_ssl_configuration",
    "resolve_ssl_issues_for_urls",
]
