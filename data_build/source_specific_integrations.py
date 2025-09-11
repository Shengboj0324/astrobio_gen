#!/usr/bin/env python3
"""
Source-Specific Integration Methods
===================================

Detailed implementation of data acquisition methods for each of the 13 data sources.
Each method is optimized for the specific protocols and data formats of its source.

This module contains the specialized acquisition logic that is called by the main
Comprehensive13SourcesIntegration class.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp
import pandas as pd
import requests
from astroquery.mast import Observations
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

logger = logging.getLogger(__name__)


class SourceSpecificIntegrations:
    """Source-specific data acquisition methods"""
    
    def __init__(self, session: requests.Session, auth_manager):
        self.session = session
        self.auth_manager = auth_manager
    
    async def acquire_nasa_exoplanet_data(self, source_config) -> Dict[str, Any]:
        """Acquire data from NASA Exoplanet Archive using nstedAPI and TAP"""
        try:
            logger.info("🌌 Acquiring NASA Exoplanet Archive data...")
            
            # Method 1: Use nstedAPI for confirmed planets
            nsted_url = f"{source_config.base_url}{source_config.api_endpoints['nstedAPI']}"
            params = {
                'table': 'ps',  # Planetary Systems table
                'format': 'csv',
                'select': 'pl_name,hostname,pl_orbper,pl_rade,pl_masse,st_teff,st_rad,st_mass',
                'where': 'pl_controv_flag=0'  # Only confirmed planets
            }
            
            response = self.session.get(nsted_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse CSV data
            import io
            csv_data = pd.read_csv(io.StringIO(response.text), comment='#')
            
            # Method 2: Use astroquery for additional data
            try:
                # Get composite planet data
                composite_data = NasaExoplanetArchive.query_criteria(
                    table="pscomppars", 
                    select="pl_name,hostname,pl_orbper,pl_rade,pl_masse",
                    where="pl_controv_flag=0"
                ).to_pandas()
                
                # Combine datasets
                combined_data = pd.concat([csv_data, composite_data], ignore_index=True)
                combined_data = combined_data.drop_duplicates(subset=['pl_name'], keep='first')
                
            except Exception as e:
                logger.warning(f"Astroquery fallback failed: {e}, using nstedAPI data only")
                combined_data = csv_data
            
            # Data quality metrics
            quality_metrics = {
                'total_records': len(combined_data),
                'completeness_score': 1.0 - (combined_data.isnull().sum().sum() / combined_data.size),
                'unique_planets': combined_data['pl_name'].nunique(),
                'unique_hosts': combined_data['hostname'].nunique()
            }
            
            return {
                'success': True,
                'data': combined_data,
                'record_count': len(combined_data),
                'data_size_mb': combined_data.memory_usage(deep=True).sum() / 1024 / 1024,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            logger.error(f"NASA Exoplanet Archive acquisition failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def acquire_jwst_mast_data(self, source_config) -> Dict[str, Any]:
        """Acquire JWST data from MAST using multiple protocols"""
        try:
            logger.info("🔭 Acquiring JWST MAST data...")
            
            # Method 1: Use astroquery.mast for JWST observations
            observations = Observations.query_criteria(
                obs_collection="JWST",
                dataproduct_type=["image", "spectrum"],
                calib_level=[2, 3],  # Processed data
                obstype="science"
            )
            
            if len(observations) == 0:
                return {'success': False, 'error': 'No JWST observations found'}
            
            # Convert to pandas for easier handling
            jwst_data = observations.to_pandas()
            
            # Method 2: Get additional metadata via GraphQL (if available)
            try:
                graphql_url = f"{source_config.base_url}{source_config.api_endpoints['graphql']}"
                graphql_query = """
                query {
                    jwst_observations(limit: 100) {
                        obs_id
                        target_name
                        instrument_name
                        filters
                        exposure_time
                        obs_begin_mjd
                    }
                }
                """
                
                async with aiohttp.ClientSession() as session:
                    headers = {'Content-Type': 'application/json'}
                    if 'Authorization' in self.session.headers:
                        headers['Authorization'] = self.session.headers['Authorization']
                    
                    async with session.post(
                        graphql_url,
                        json={'query': graphql_query},
                        headers=headers,
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            graphql_data = await response.json()
                            logger.info("✅ GraphQL metadata acquired")
                        else:
                            logger.warning("GraphQL query failed, using astroquery data only")
            
            except Exception as e:
                logger.warning(f"GraphQL acquisition failed: {e}")
            
            # Data quality metrics
            quality_metrics = {
                'total_observations': len(jwst_data),
                'unique_targets': jwst_data['target_name'].nunique() if 'target_name' in jwst_data.columns else 0,
                'instruments': jwst_data['instrument_name'].nunique() if 'instrument_name' in jwst_data.columns else 0,
                'data_completeness': 1.0 - (jwst_data.isnull().sum().sum() / jwst_data.size)
            }
            
            return {
                'success': True,
                'data': jwst_data,
                'record_count': len(jwst_data),
                'data_size_mb': jwst_data.memory_usage(deep=True).sum() / 1024 / 1024,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            logger.error(f"JWST MAST acquisition failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def acquire_kepler_k2_data(self, source_config) -> Dict[str, Any]:
        """Acquire Kepler/K2 data from MAST"""
        try:
            logger.info("🌟 Acquiring Kepler/K2 data...")
            
            # Query Kepler observations
            kepler_obs = Observations.query_criteria(
                obs_collection=["Kepler", "K2"],
                dataproduct_type="timeseries",
                calib_level=2
            )
            
            if len(kepler_obs) == 0:
                return {'success': False, 'error': 'No Kepler/K2 observations found'}
            
            kepler_data = kepler_obs.to_pandas()
            
            # Get target pixel files for a subset
            try:
                # Limit to first 100 observations for efficiency
                sample_obs = kepler_obs[:100] if len(kepler_obs) > 100 else kepler_obs
                
                # Get data products
                data_products = Observations.get_product_list(sample_obs)
                if len(data_products) > 0:
                    products_data = data_products.to_pandas()
                    logger.info(f"✅ Found {len(data_products)} data products")
                else:
                    products_data = pd.DataFrame()
                    
            except Exception as e:
                logger.warning(f"Data products query failed: {e}")
                products_data = pd.DataFrame()
            
            # Quality metrics
            quality_metrics = {
                'total_observations': len(kepler_data),
                'kepler_missions': kepler_data['obs_collection'].value_counts().to_dict() if 'obs_collection' in kepler_data.columns else {},
                'data_products_count': len(products_data),
                'unique_targets': kepler_data['target_name'].nunique() if 'target_name' in kepler_data.columns else 0
            }
            
            return {
                'success': True,
                'data': kepler_data,
                'record_count': len(kepler_data),
                'data_size_mb': kepler_data.memory_usage(deep=True).sum() / 1024 / 1024,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            logger.error(f"Kepler/K2 acquisition failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def acquire_tess_data(self, source_config) -> Dict[str, Any]:
        """Acquire TESS data from MAST"""
        try:
            logger.info("🛰️ Acquiring TESS data...")
            
            # Query TESS observations
            tess_obs = Observations.query_criteria(
                obs_collection="TESS",
                dataproduct_type=["timeseries", "image"],
                calib_level=[2, 3]
            )
            
            if len(tess_obs) == 0:
                return {'success': False, 'error': 'No TESS observations found'}
            
            tess_data = tess_obs.to_pandas()
            
            # Quality metrics
            quality_metrics = {
                'total_observations': len(tess_data),
                'sectors': tess_data['sequence_number'].nunique() if 'sequence_number' in tess_data.columns else 0,
                'unique_targets': tess_data['target_name'].nunique() if 'target_name' in tess_data.columns else 0,
                'data_types': tess_data['dataproduct_type'].value_counts().to_dict() if 'dataproduct_type' in tess_data.columns else {}
            }
            
            return {
                'success': True,
                'data': tess_data,
                'record_count': len(tess_data),
                'data_size_mb': tess_data.memory_usage(deep=True).sum() / 1024 / 1024,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            logger.error(f"TESS acquisition failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def acquire_exoplanets_org_data(self, source_config) -> Dict[str, Any]:
        """Acquire data from exoplanets.org CSV"""
        try:
            logger.info("🪐 Acquiring exoplanets.org data...")
            
            csv_url = f"{source_config.base_url}{source_config.api_endpoints['csv_download']}"
            
            # Download CSV data
            response = self.session.get(csv_url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV
            import io
            exoplanet_data = pd.read_csv(io.StringIO(response.text))
            
            # Quality metrics
            quality_metrics = {
                'total_planets': len(exoplanet_data),
                'columns_count': len(exoplanet_data.columns),
                'completeness_score': 1.0 - (exoplanet_data.isnull().sum().sum() / exoplanet_data.size),
                'discovery_methods': exoplanet_data['DISCOVERYMETHOD'].value_counts().to_dict() if 'DISCOVERYMETHOD' in exoplanet_data.columns else {}
            }
            
            return {
                'success': True,
                'data': exoplanet_data,
                'record_count': len(exoplanet_data),
                'data_size_mb': exoplanet_data.memory_usage(deep=True).sum() / 1024 / 1024,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            logger.error(f"Exoplanets.org acquisition failed: {e}")
            return {'success': False, 'error': str(e)}

    async def acquire_ncbi_data(self, source_config) -> Dict[str, Any]:
        """Acquire data from NCBI GenBank using E-utilities"""
        try:
            logger.info("🧬 Acquiring NCBI GenBank data...")

            # Get API key for enhanced rate limits
            api_key = self.auth_manager.credentials.get('ncbi')

            # Search for astrobiology-related sequences
            search_terms = [
                "astrobiology",
                "extremophile",
                "thermophile",
                "halophile",
                "methanogen",
                "archaea"
            ]

            all_sequences = []

            for term in search_terms:
                try:
                    # ESearch to get IDs
                    esearch_url = f"{source_config.base_url}{source_config.api_endpoints['esearch']}"
                    search_params = {
                        'db': 'nucleotide',
                        'term': term,
                        'retmax': 100,  # Limit for efficiency
                        'retmode': 'json'
                    }
                    if api_key:
                        search_params['api_key'] = api_key

                    response = self.session.get(esearch_url, params=search_params, timeout=30)
                    response.raise_for_status()

                    search_result = response.json()
                    id_list = search_result.get('esearchresult', {}).get('idlist', [])

                    if id_list:
                        # EFetch to get sequence data
                        efetch_url = f"{source_config.base_url}{source_config.api_endpoints['efetch']}"
                        fetch_params = {
                            'db': 'nucleotide',
                            'id': ','.join(id_list[:50]),  # Limit to first 50
                            'rettype': 'fasta',
                            'retmode': 'text'
                        }
                        if api_key:
                            fetch_params['api_key'] = api_key

                        fetch_response = self.session.get(efetch_url, params=fetch_params, timeout=30)
                        fetch_response.raise_for_status()

                        # Parse FASTA sequences
                        sequences = self._parse_fasta_sequences(fetch_response.text)
                        all_sequences.extend(sequences)

                        logger.info(f"✅ {term}: {len(sequences)} sequences")

                    # Rate limiting
                    await asyncio.sleep(0.1 if api_key else 0.34)  # 10 r/s with key, 3 r/s without

                except Exception as e:
                    logger.warning(f"Failed to acquire data for term '{term}': {e}")
                    continue

            if not all_sequences:
                return {'success': False, 'error': 'No sequences acquired'}

            # Convert to DataFrame
            ncbi_data = pd.DataFrame(all_sequences)

            # Quality metrics
            quality_metrics = {
                'total_sequences': len(ncbi_data),
                'unique_organisms': ncbi_data['organism'].nunique() if 'organism' in ncbi_data.columns else 0,
                'avg_sequence_length': ncbi_data['sequence_length'].mean() if 'sequence_length' in ncbi_data.columns else 0,
                'search_terms_successful': len([term for term in search_terms if any(term in seq.get('description', '') for seq in all_sequences)])
            }

            return {
                'success': True,
                'data': ncbi_data,
                'record_count': len(ncbi_data),
                'data_size_mb': ncbi_data.memory_usage(deep=True).sum() / 1024 / 1024,
                'quality_metrics': quality_metrics
            }

        except Exception as e:
            logger.error(f"NCBI GenBank acquisition failed: {e}")
            return {'success': False, 'error': str(e)}

    def _parse_fasta_sequences(self, fasta_text: str) -> List[Dict[str, Any]]:
        """Parse FASTA format sequences"""
        sequences = []
        current_seq = None

        for line in fasta_text.split('\n'):
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)

                # Parse header
                header_parts = line[1:].split('|')
                current_seq = {
                    'accession': header_parts[1] if len(header_parts) > 1 else '',
                    'description': line[1:],
                    'sequence': '',
                    'organism': '',
                    'sequence_length': 0
                }

                # Extract organism if present
                if '[' in line and ']' in line:
                    start = line.rfind('[')
                    end = line.rfind(']')
                    if start < end:
                        current_seq['organism'] = line[start+1:end]

            elif line and current_seq:
                current_seq['sequence'] += line

        # Add the last sequence
        if current_seq:
            sequences.append(current_seq)

        # Calculate sequence lengths
        for seq in sequences:
            seq['sequence_length'] = len(seq['sequence'])

        return sequences

    async def acquire_ensembl_data(self, source_config) -> Dict[str, Any]:
        """Acquire data from Ensembl REST API"""
        try:
            logger.info("🧬 Acquiring Ensembl data...")

            # Get species information
            species_url = f"{source_config.base_url}/info/species"

            async with aiohttp.ClientSession() as session:
                async with session.get(species_url, timeout=30) as response:
                    if response.status == 200:
                        species_data = await response.json()
                        species_list = species_data.get('species', [])
                    else:
                        return {'success': False, 'error': f'Species query failed: {response.status}'}

            # Focus on model organisms and extremophiles
            target_species = [
                'homo_sapiens',
                'escherichia_coli_str_k_12_substr_mg1655',
                'saccharomyces_cerevisiae',
                'drosophila_melanogaster',
                'caenorhabditis_elegans'
            ]

            all_gene_data = []

            for species in target_species:
                try:
                    # Get genes for this species
                    genes_url = f"{source_config.base_url}/lookup/genome/{species}"

                    async with aiohttp.ClientSession() as session:
                        async with session.get(genes_url, timeout=30) as response:
                            if response.status == 200:
                                genome_data = await response.json()
                                all_gene_data.append({
                                    'species': species,
                                    'genome_data': genome_data
                                })
                                logger.info(f"✅ {species}: genome data acquired")
                            else:
                                logger.warning(f"Failed to get genome data for {species}")

                    # Rate limiting
                    await asyncio.sleep(0.067)  # 15 requests per second limit

                except Exception as e:
                    logger.warning(f"Failed to acquire data for {species}: {e}")
                    continue

            if not all_gene_data:
                return {'success': False, 'error': 'No genome data acquired'}

            # Convert to DataFrame
            ensembl_data = pd.DataFrame(all_gene_data)

            # Quality metrics
            quality_metrics = {
                'species_count': len(ensembl_data),
                'total_species_available': len(species_list),
                'target_species_success_rate': len(all_gene_data) / len(target_species),
                'data_completeness': 1.0
            }

            return {
                'success': True,
                'data': ensembl_data,
                'record_count': len(ensembl_data),
                'data_size_mb': ensembl_data.memory_usage(deep=True).sum() / 1024 / 1024,
                'quality_metrics': quality_metrics
            }

        except Exception as e:
            logger.error(f"Ensembl acquisition failed: {e}")
            return {'success': False, 'error': str(e)}
