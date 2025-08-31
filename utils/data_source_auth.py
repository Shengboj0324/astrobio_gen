#!/usr/bin/env python3
"""
Data Source Authentication Manager
==================================

Secure authentication and access management for all scientific data sources.
Handles API keys, OAuth tokens, and cookie-based authentication.

Version: 1.0.0 (Production Ready)
"""

import os
import requests
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DataSourceAuthManager:
    """Centralized authentication manager for all data sources"""
    
    def __init__(self):
        self.credentials = self._load_credentials()
        self.session_cache = {}
        
    def _load_credentials(self) -> Dict[str, str]:
        """Load all credentials from environment variables"""
        return {
            'nasa_mast': os.getenv('NASA_MAST_API_KEY'),
            'copernicus_cds': os.getenv('COPERNICUS_CDS_API_KEY'),
            'ncbi': os.getenv('NCBI_API_KEY'),
            'gaia_user': os.getenv('GAIA_USER'),
            'gaia_pass': os.getenv('GAIA_PASS'),
            'eso_user': os.getenv('ESO_USERNAME'),
            'eso_pass': os.getenv('ESO_PASSWORD'),
            'uniprot': os.getenv('UNIPROT_API_KEY'),
            'kegg': os.getenv('KEGG_API_KEY'),
        }
    
    def verify_nasa_mast(self) -> Dict[str, Any]:
        """Verify NASA MAST API token"""
        if not self.credentials['nasa_mast']:
            return {'status': 'error', 'message': 'NASA MAST API key not configured'}
        
        try:
            headers = {
                'Authorization': f'token {self.credentials["nasa_mast"]}',
                'Content-Type': 'application/json'
            }
            
            # Test API call
            response = requests.get(
                'https://mast.stsci.edu/api/v0.1/invoke',
                headers=headers,
                params={
                    'service': 'Mast.Name.Lookup',
                    'params': {'input': 'M31', 'format': 'json'}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return {'status': 'success', 'message': 'NASA MAST API access verified'}
            else:
                return {'status': 'error', 'message': f'API call failed: {response.status_code}'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Connection error: {str(e)}'}
    
    def verify_copernicus_cds(self) -> Dict[str, Any]:
        """Verify Copernicus CDS API key"""
        if not self.credentials['copernicus_cds']:
            return {'status': 'error', 'message': 'Copernicus CDS API key not configured'}
        
        try:
            import cdsapi
            
            # Test CDS API connection
            c = cdsapi.Client()
            
            # Simple test - get dataset info
            info = c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': 'surface_pressure',
                    'year': '2023',
                    'month': '01',
                    'day': '01',
                    'time': '00:00',
                    'format': 'netcdf',
                    'area': [90, -180, -90, 180],  # Global
                },
                target=None  # Don't actually download
            )
            
            return {'status': 'success', 'message': 'Copernicus CDS API access verified'}
            
        except Exception as e:
            return {'status': 'error', 'message': f'CDS API error: {str(e)}'}
    
    def verify_ncbi(self) -> Dict[str, Any]:
        """Verify NCBI API key"""
        if not self.credentials['ncbi']:
            return {'status': 'error', 'message': 'NCBI API key not configured'}
        
        try:
            # Test NCBI E-utilities API
            response = requests.get(
                'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi',
                params={
                    'api_key': self.credentials['ncbi'],
                    'retmode': 'json'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return {'status': 'success', 'message': 'NCBI API access verified'}
            else:
                return {'status': 'error', 'message': f'NCBI API call failed: {response.status_code}'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'NCBI connection error: {str(e)}'}
    
    def verify_gaia_access(self) -> Dict[str, Any]:
        """Verify ESA Gaia archive access"""
        if not (self.credentials['gaia_user'] and self.credentials['gaia_pass']):
            return {'status': 'error', 'message': 'Gaia credentials not configured'}
        
        try:
            from astroquery.gaia import Gaia
            
            # Login to Gaia archive
            Gaia.login(
                user=self.credentials['gaia_user'],
                password=self.credentials['gaia_pass']
            )
            
            # Test query
            job = Gaia.launch_job_async(
                "SELECT TOP 5 source_id, ra, dec FROM gaiadr3.gaia_source"
            )
            results = job.get_results()
            
            return {'status': 'success', 'message': f'Gaia access verified - {len(results)} records retrieved'}
            
        except Exception as e:
            return {'status': 'error', 'message': f'Gaia access error: {str(e)}'}
    
    def get_eso_token(self) -> Optional[str]:
        """Get ESO authentication token"""
        if not (self.credentials['eso_user'] and self.credentials['eso_pass']):
            logger.error("ESO credentials not configured")
            return None
        
        try:
            # Get ESO token using OAuth2
            token_url = "https://www.eso.org/sso/oidc/token"
            params = {
                'response_type': 'id_token token',
                'grant_type': 'password',
                'client_id': 'clientid',
                'client_secret': 'clientSecret',
                'username': self.credentials['eso_user'],
                'password': self.credentials['eso_pass']
            }
            
            response = requests.get(token_url, params=params, timeout=30)
            
            if response.status_code == 200:
                # Parse token from response
                token_data = response.json()
                return token_data.get('id_token')
            else:
                logger.error(f"ESO token request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"ESO token error: {e}")
            return None
    
    def verify_all_sources(self) -> Dict[str, Dict[str, Any]]:
        """Verify access to all configured data sources"""
        results = {}
        
        print("🔍 VERIFYING ALL DATA SOURCE ACCESS...")
        
        # Test NASA MAST
        print("\n🌌 Testing NASA MAST...")
        results['nasa_mast'] = self.verify_nasa_mast()
        
        # Test Copernicus CDS
        print("\n🌍 Testing Copernicus CDS...")
        results['copernicus_cds'] = self.verify_copernicus_cds()
        
        # Test NCBI
        print("\n🧬 Testing NCBI...")
        results['ncbi'] = self.verify_ncbi()
        
        # Test Gaia
        print("\n🛰️ Testing ESA Gaia...")
        results['gaia'] = self.verify_gaia_access()
        
        # Test ESO (token generation only)
        print("\n🔭 Testing ESO token generation...")
        eso_token = self.get_eso_token()
        if eso_token:
            results['eso'] = {'status': 'success', 'message': 'ESO token generated successfully'}
        else:
            results['eso'] = {'status': 'error', 'message': 'ESO token generation failed'}
        
        return results


# Global authentication manager instance
auth_manager = DataSourceAuthManager()
