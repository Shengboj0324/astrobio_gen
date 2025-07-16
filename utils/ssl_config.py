#!/usr/bin/env python3
"""
SSL Configuration Helper for External API Access
===============================================

This module provides SSL configuration utilities to resolve certificate 
issues when accessing external APIs and data sources.
"""

import ssl
import aiohttp
import requests
import logging
import urllib3
from urllib3.exceptions import InsecureRequestWarning
import certifi
import os
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

def configure_ssl_for_external_apis():
    """Configure SSL settings for reliable external API access"""
    
    # Disable SSL warnings for development/testing
    urllib3.disable_warnings(InsecureRequestWarning)
    
    # Set certificate bundle path
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    
    logger.info("SSL configuration updated for external API access")

def get_ssl_context(verify_ssl: bool = True):
    """Get SSL context for external connections"""
    if verify_ssl:
        context = ssl.create_default_context()
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

def get_aiohttp_connector(verify_ssl: bool = True):
    """Get aiohttp connector with appropriate SSL settings"""
    if verify_ssl:
        # Use default SSL context with certificate verification
        connector = aiohttp.TCPConnector(
            ssl=get_ssl_context(verify_ssl=True),
            limit=100,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
    else:
        # For development/testing - disable SSL verification
        connector = aiohttp.TCPConnector(
            ssl=False,
            limit=100,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
    
    return connector

def get_requests_session(verify_ssl: bool = True):
    """Get requests session with appropriate SSL settings"""
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
    session.headers.update({
        'User-Agent': 'Astrobiology-Research-Platform/1.0 (Python; Scientific Research)'
    })
    
    return session

def check_ssl_configuration():
    """Check and report SSL configuration status"""
    config_status = {
        'certifi_available': False,
        'cert_bundle_path': None,
        'ssl_context_working': False,
        'external_api_accessible': False
    }
    
    try:
        import certifi
        config_status['certifi_available'] = True
        config_status['cert_bundle_path'] = certifi.where()
        logger.info(f"Certificate bundle found: {certifi.where()}")
    except ImportError:
        logger.warning("Certifi package not available")
    
    try:
        context = get_ssl_context(verify_ssl=True)
        config_status['ssl_context_working'] = True
        logger.info("SSL context created successfully")
    except Exception as e:
        logger.error(f"SSL context creation failed: {e}")
    
    # Test external API access
    try:
        session = get_requests_session(verify_ssl=False)  # Start with relaxed SSL for testing
        response = session.get('https://httpbin.org/get', timeout=10)
        if response.status_code == 200:
            config_status['external_api_accessible'] = True
            logger.info("External API access test successful")
    except Exception as e:
        logger.warning(f"External API access test failed: {e}")
    
    return config_status

# Initialize SSL configuration on import
configure_ssl_for_external_apis()

# Export main functions
__all__ = [
    'configure_ssl_for_external_apis',
    'get_ssl_context', 
    'get_aiohttp_connector',
    'get_requests_session',
    'check_ssl_configuration'
] 