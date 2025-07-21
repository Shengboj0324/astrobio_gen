#!/usr/bin/env python3
"""
Predictive URL Discovery System
===============================

AI-powered system for proactively discovering new data source URLs before current ones fail.
Uses machine learning, pattern matching, social media monitoring, and institutional intelligence.

Features:
- Pattern-based URL prediction
- Social media monitoring for infrastructure announcements
- ML-based change prediction
- Institution-specific discovery patterns
- Web crawling for new endpoints
- API discovery through documentation parsing
"""

import asyncio
import aiohttp
import re
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
import pandas as pd
import numpy as np

# Optional ML and NLP imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    feedparser = None

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    tweepy = None

try:
    import nltk
    from textblob import TextBlob
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    nltk = None
    TextBlob = None

import requests
from bs4 import BeautifulSoup
import hashlib
import time

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class URLPrediction:
    """URL prediction result"""
    source_name: str
    predicted_url: str
    confidence_score: float
    discovery_method: str
    evidence: List[str]
    prediction_date: datetime
    verification_status: str = "pending"
    actual_url_found: Optional[str] = None

@dataclass
class InstitutionPattern:
    """Institution-specific URL patterns"""
    institution: str
    domain_patterns: List[str]
    subdomain_patterns: List[str]
    path_patterns: List[str]
    api_patterns: List[str]
    archive_patterns: List[str]
    historical_changes: List[Dict[str, Any]]

class PredictiveURLDiscovery:
    """
    AI-powered predictive URL discovery system
    """
    
    def __init__(self, config_path: str = "config/data_sources"):
        self.config_path = Path(config_path)
        self.db_path = Path("data/metadata/url_predictions.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases
        self._initialize_databases()
        
        # Load institution patterns
        self.institution_patterns = self._load_institution_patterns()
        
        # ML models for prediction
        self.url_classifier = None
        self.change_predictor = None
        self._initialize_ml_models()
        
        # Social media monitoring
        self.social_monitors = {}
        self._initialize_social_monitoring()
        
        # Web crawling session
        self.session = None
        
        # Discovery patterns
        self.common_patterns = self._load_common_patterns()
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_lock = threading.Lock()
        
        logger.info("Predictive URL Discovery system initialized")
    
    def _initialize_databases(self):
        """Initialize prediction tracking databases"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS url_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    predicted_url TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    discovery_method TEXT NOT NULL,
                    evidence TEXT NOT NULL, -- JSON array
                    prediction_date TIMESTAMP NOT NULL,
                    verification_status TEXT DEFAULT 'pending',
                    actual_url_found TEXT,
                    verified_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS institution_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    institution TEXT NOT NULL,
                    old_url TEXT NOT NULL,
                    new_url TEXT NOT NULL,
                    change_date TIMESTAMP NOT NULL,
                    change_type TEXT NOT NULL, -- migration, rebranding, restructure
                    announcement_source TEXT,
                    detection_method TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS social_mentions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL, -- twitter, reddit, github
                    mention_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    author TEXT,
                    url_mentioned TEXT,
                    institution_mentioned TEXT,
                    sentiment_score REAL,
                    relevance_score REAL,
                    mention_date TIMESTAMP NOT NULL,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(platform, mention_id)
                );
                
                CREATE TABLE IF NOT EXISTS api_discoveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    api_endpoint TEXT NOT NULL,
                    api_version TEXT,
                    discovery_method TEXT NOT NULL,
                    documentation_url TEXT,
                    response_format TEXT,
                    authentication_required BOOLEAN,
                    rate_limit TEXT,
                    confidence_score REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS pattern_effectiveness (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_value TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    total_attempts INTEGER DEFAULT 0,
                    last_success TIMESTAMP,
                    effectiveness_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pattern_type, pattern_value)
                );
                
                CREATE INDEX IF NOT EXISTS idx_predictions_source ON url_predictions(source_name);
                CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON url_predictions(confidence_score);
                CREATE INDEX IF NOT EXISTS idx_changes_institution ON institution_changes(institution);
                CREATE INDEX IF NOT EXISTS idx_mentions_platform ON social_mentions(platform);
                CREATE INDEX IF NOT EXISTS idx_mentions_processed ON social_mentions(processed);
            """)
    
    def _load_institution_patterns(self) -> Dict[str, InstitutionPattern]:
        """Load institution-specific URL patterns"""
        patterns = {
            'nasa': InstitutionPattern(
                institution='nasa',
                domain_patterns=[
                    '*.nasa.gov',
                    '*.ipac.caltech.edu',
                    '*.stsci.edu',
                    '*.gsfc.nasa.gov'
                ],
                subdomain_patterns=[
                    'data', 'archive', 'api', 'mast', 'heasarc',
                    'exoplanetarchive', 'giss', 'simplex'
                ],
                path_patterns=[
                    '/data/', '/archive/', '/api/', '/cgi-bin/',
                    '/missions/', '/catalogs/', '/TAP/'
                ],
                api_patterns=[
                    '/api/v{version}',
                    '/cgi-bin/nstedAPI',
                    '/TAP/sync',
                    '/api/v0.1'
                ],
                archive_patterns=[
                    '/data/{year}/',
                    '/archive/{mission}/',
                    '/catalogs/{survey}/'
                ],
                historical_changes=[
                    {
                        'date': '2023-01-15',
                        'old_pattern': 'irsa.ipac.caltech.edu',
                        'new_pattern': 'sha.ipac.caltech.edu',
                        'reason': 'infrastructure_update'
                    }
                ]
            ),
            
            'ncbi': InstitutionPattern(
                institution='ncbi',
                domain_patterns=[
                    '*.ncbi.nlm.nih.gov',
                    '*.nih.gov',
                    'ftp*.ncbi.nlm.nih.gov'
                ],
                subdomain_patterns=[
                    'ftp', 'ftp-private', 'ftp-trace', 'eutils',
                    'api', 'www', 'pubmed'
                ],
                path_patterns=[
                    '/genomes/', '/pubmed/', '/nucleotide/',
                    '/protein/', '/taxonomy/', '/sra/'
                ],
                api_patterns=[
                    '/entrez/eutils/{tool}.fcgi',
                    '/datasets/v2/',
                    '/sra/docs/toolkitsoft/'
                ],
                archive_patterns=[
                    '/genomes/{category}/',
                    '/sra/sra-instant/reads/',
                    '/blast/db/'
                ],
                historical_changes=[]
            ),
            
            'ebi': InstitutionPattern(
                institution='ebi',
                domain_patterns=[
                    '*.ebi.ac.uk',
                    '*.embl.org'
                ],
                subdomain_patterns=[
                    'ftp', 'www', 'rest', 'api'
                ],
                path_patterns=[
                    '/pub/databases/', '/Tools/', '/services/'
                ],
                api_patterns=[
                    '/proteins/api/',
                    '/ebisearch/ws/rest/'
                ],
                archive_patterns=[
                    '/pub/databases/{database}/'
                ],
                historical_changes=[]
            )
        }
        
        return patterns
    
    def _load_common_patterns(self) -> Dict[str, List[str]]:
        """Load common URL transformation patterns"""
        return {
            'subdomain_variations': [
                'data', 'api', 'archive', 'ftp', 'download',
                'files', 'static', 'cdn', 'www2', 'mirror',
                'backup', 'old', 'legacy', 'v2', 'beta'
            ],
            'path_variations': [
                '/data/', '/files/', '/download/', '/static/',
                '/public/', '/archive/', '/api/', '/v1/', '/v2/',
                '/legacy/', '/old/', '/current/', '/latest/'
            ],
            'protocol_variations': [
                'https://', 'http://', 'ftp://', 'rsync://'
            ],
            'common_migrations': [
                ('www.', ''),
                ('http://', 'https://'),
                ('/old/', '/current/'),
                ('/v1/', '/v2/'),
                ('ftp.', 'files.'),
                ('.org/', '.gov/'),
                ('.edu/', '.org/')
            ]
        }
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for URL prediction"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.info("Scikit-learn not available, ML features disabled")
                return
                
            # Load historical data for training
            historical_data = self._load_historical_changes()
            
            if len(historical_data) > 10:  # Need minimum data for training
                self._train_url_classifier(historical_data)
                self._train_change_predictor(historical_data)
            else:
                logger.info("Insufficient historical data for ML model training")
                
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    def _load_historical_changes(self) -> pd.DataFrame:
        """Load historical URL changes for ML training"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT institution, old_url, new_url, change_date, 
                           change_type, detection_method
                    FROM institution_changes
                    ORDER BY change_date DESC
                """, conn)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical changes: {e}")
            return pd.DataFrame()
    
    def _train_url_classifier(self, data: pd.DataFrame):
        """Train ML model to classify URL change patterns"""
        try:
            if not SKLEARN_AVAILABLE or len(data) < 5:
                return
            
            # Extract features from URLs
            features = []
            labels = []
            
            for _, row in data.iterrows():
                old_url = row['old_url']
                new_url = row['new_url']
                
                # Extract URL features
                old_parsed = urlparse(old_url)
                new_parsed = urlparse(new_url)
                
                feature_vector = [
                    len(old_parsed.netloc),
                    len(old_parsed.path),
                    old_parsed.netloc.count('.'),
                    old_parsed.path.count('/'),
                    1 if 'www' in old_parsed.netloc else 0,
                    1 if 'api' in old_parsed.netloc else 0,
                    1 if 'data' in old_parsed.netloc else 0,
                    1 if 'ftp' in old_parsed.netloc else 0
                ]
                
                features.append(feature_vector)
                labels.append(row['change_type'])
            
            # Train classifier
            self.url_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            self.url_classifier.fit(features, labels)
            
            logger.info("URL classifier trained successfully")
            
        except Exception as e:
            logger.error(f"Error training URL classifier: {e}")
    
    def _train_change_predictor(self, data: pd.DataFrame):
        """Train model to predict when URLs might change"""
        try:
            if not SKLEARN_AVAILABLE or len(data) < 5:
                return
            
            # Create time-based features
            features = []
            labels = []
            
            for _, row in data.iterrows():
                change_date = pd.to_datetime(row['change_date'])
                
                # Time-based features
                feature_vector = [
                    change_date.month,
                    change_date.day,
                    change_date.weekday(),
                    change_date.year % 10,  # Decade pattern
                    1 if row['change_type'] == 'migration' else 0,
                    1 if row['change_type'] == 'rebranding' else 0
                ]
                
                features.append(feature_vector)
                # Binary: major change (1) or minor change (0)
                labels.append(1 if row['change_type'] in ['migration', 'rebranding'] else 0)
            
            self.change_predictor = RandomForestClassifier(n_estimators=30, random_state=42)
            self.change_predictor.fit(features, labels)
            
            logger.info("Change predictor trained successfully")
            
        except Exception as e:
            logger.error(f"Error training change predictor: {e}")
    
    def _initialize_social_monitoring(self):
        """Initialize social media monitoring for infrastructure announcements"""
        # Note: In production, you'd configure actual API keys
        self.social_keywords = [
            'data migration', 'url change', 'site moved', 'new api',
            'infrastructure update', 'server migration', 'domain change',
            'archive moved', 'new data portal', 'api upgrade'
        ]
    
    async def predict_url_changes(self, source_name: str, current_url: str) -> List[URLPrediction]:
        """Predict potential new URLs for a data source"""
        predictions = []
        
        try:
            # Pattern-based predictions
            pattern_predictions = await self._predict_by_patterns(source_name, current_url)
            predictions.extend(pattern_predictions)
            
            # Institution-specific predictions
            institution_predictions = await self._predict_by_institution(source_name, current_url)
            predictions.extend(institution_predictions)
            
            # ML-based predictions
            if self.url_classifier:
                ml_predictions = await self._predict_by_ml(source_name, current_url)
                predictions.extend(ml_predictions)
            
            # Social media intelligence
            social_predictions = await self._predict_by_social_intelligence(source_name)
            predictions.extend(social_predictions)
            
            # Documentation parsing
            doc_predictions = await self._predict_by_documentation(source_name, current_url)
            predictions.extend(doc_predictions)
            
            # Remove duplicates and sort by confidence
            unique_predictions = self._deduplicate_predictions(predictions)
            unique_predictions.sort(key=lambda x: x.confidence_score, reverse=True)
            
            # Store predictions
            await self._store_predictions(unique_predictions)
            
            return unique_predictions[:10]  # Top 10 predictions
            
        except Exception as e:
            logger.error(f"Error predicting URL changes for {source_name}: {e}")
            return []
    
    async def _predict_by_patterns(self, source_name: str, current_url: str) -> List[URLPrediction]:
        """Predict URLs using common transformation patterns"""
        predictions = []
        parsed = urlparse(current_url)
        
        try:
            # Subdomain variations
            for subdomain in self.common_patterns['subdomain_variations']:
                new_netloc = f"{subdomain}.{parsed.netloc}"
                new_url = f"{parsed.scheme}://{new_netloc}{parsed.path}"
                
                confidence = self._calculate_pattern_confidence('subdomain', subdomain)
                
                predictions.append(URLPrediction(
                    source_name=source_name,
                    predicted_url=new_url,
                    confidence_score=confidence,
                    discovery_method='pattern_subdomain',
                    evidence=[f"Subdomain pattern: {subdomain}"],
                    prediction_date=datetime.now(timezone.utc)
                ))
            
            # Path variations
            for path in self.common_patterns['path_variations']:
                new_url = f"{parsed.scheme}://{parsed.netloc}{path}"
                
                confidence = self._calculate_pattern_confidence('path', path)
                
                predictions.append(URLPrediction(
                    source_name=source_name,
                    predicted_url=new_url,
                    confidence_score=confidence,
                    discovery_method='pattern_path',
                    evidence=[f"Path pattern: {path}"],
                    prediction_date=datetime.now(timezone.utc)
                ))
            
            # Common migrations
            for old_pattern, new_pattern in self.common_patterns['common_migrations']:
                if old_pattern in current_url:
                    new_url = current_url.replace(old_pattern, new_pattern)
                    
                    confidence = self._calculate_pattern_confidence('migration', 
                                                                 f"{old_pattern}->{new_pattern}")
                    
                    predictions.append(URLPrediction(
                        source_name=source_name,
                        predicted_url=new_url,
                        confidence_score=confidence,
                        discovery_method='pattern_migration',
                        evidence=[f"Migration pattern: {old_pattern} -> {new_pattern}"],
                        prediction_date=datetime.now(timezone.utc)
                    ))
            
        except Exception as e:
            logger.error(f"Error in pattern-based prediction: {e}")
        
        return predictions
    
    async def _predict_by_institution(self, source_name: str, current_url: str) -> List[URLPrediction]:
        """Predict URLs using institution-specific patterns"""
        predictions = []
        parsed = urlparse(current_url)
        
        try:
            # Identify institution
            institution = self._identify_institution(current_url)
            if not institution or institution not in self.institution_patterns:
                return predictions
            
            pattern = self.institution_patterns[institution]
            
            # Generate predictions based on institution patterns
            for subdomain in pattern.subdomain_patterns:
                # Skip if already current subdomain
                if subdomain in parsed.netloc:
                    continue
                
                base_domain = self._extract_base_domain(parsed.netloc)
                new_netloc = f"{subdomain}.{base_domain}"
                new_url = f"{parsed.scheme}://{new_netloc}{parsed.path}"
                
                confidence = 0.7  # Institution patterns have higher confidence
                
                predictions.append(URLPrediction(
                    source_name=source_name,
                    predicted_url=new_url,
                    confidence_score=confidence,
                    discovery_method='institution_pattern',
                    evidence=[f"Institution {institution} subdomain: {subdomain}"],
                    prediction_date=datetime.now(timezone.utc)
                ))
            
            # API pattern predictions
            for api_pattern in pattern.api_patterns:
                new_path = api_pattern.format(version='2', tool='search')  # Example formatting
                new_url = f"{parsed.scheme}://{parsed.netloc}{new_path}"
                
                predictions.append(URLPrediction(
                    source_name=source_name,
                    predicted_url=new_url,
                    confidence_score=0.6,
                    discovery_method='institution_api_pattern',
                    evidence=[f"Institution {institution} API pattern: {api_pattern}"],
                    prediction_date=datetime.now(timezone.utc)
                ))
        
        except Exception as e:
            logger.error(f"Error in institution-based prediction: {e}")
        
        return predictions
    
    def _identify_institution(self, url: str) -> Optional[str]:
        """Identify institution from URL"""
        url_lower = url.lower()
        
        if any(pattern in url_lower for pattern in ['nasa.gov', 'ipac.caltech.edu', 'stsci.edu']):
            return 'nasa'
        elif any(pattern in url_lower for pattern in ['ncbi.nlm.nih.gov', 'nih.gov']):
            return 'ncbi'
        elif any(pattern in url_lower for pattern in ['ebi.ac.uk', 'embl.org']):
            return 'ebi'
        
        return None
    
    def _extract_base_domain(self, netloc: str) -> str:
        """Extract base domain from netloc"""
        parts = netloc.split('.')
        if len(parts) >= 2:
            return '.'.join(parts[-2:])
        return netloc
    
    async def _predict_by_ml(self, source_name: str, current_url: str) -> List[URLPrediction]:
        """Predict URLs using machine learning models"""
        predictions = []
        
        try:
            if not SKLEARN_AVAILABLE or not self.url_classifier:
                return predictions
            
            # Extract features from current URL
            parsed = urlparse(current_url)
            features = [[
                len(parsed.netloc),
                len(parsed.path),
                parsed.netloc.count('.'),
                parsed.path.count('/'),
                1 if 'www' in parsed.netloc else 0,
                1 if 'api' in parsed.netloc else 0,
                1 if 'data' in parsed.netloc else 0,
                1 if 'ftp' in parsed.netloc else 0
            ]]
            
            # Predict change type
            predicted_change = self.url_classifier.predict(features)[0]
            prediction_proba = self.url_classifier.predict_proba(features)[0]
            confidence = max(prediction_proba)
            
            # Generate URL based on predicted change type
            if predicted_change == 'migration':
                # Predict migration patterns
                migration_urls = self._generate_migration_urls(current_url)
                for url in migration_urls:
                    predictions.append(URLPrediction(
                        source_name=source_name,
                        predicted_url=url,
                        confidence_score=confidence * 0.8,  # Adjust for ML uncertainty
                        discovery_method='ml_migration',
                        evidence=[f"ML predicted migration with confidence {confidence:.2f}"],
                        prediction_date=datetime.now(timezone.utc)
                    ))
        
        except Exception as e:
            logger.error(f"Error in ML-based prediction: {e}")
        
        return predictions
    
    def _generate_migration_urls(self, current_url: str) -> List[str]:
        """Generate likely migration URLs"""
        parsed = urlparse(current_url)
        migration_urls = []
        
        # Common migration patterns
        patterns = [
            f"https://new.{parsed.netloc}{parsed.path}",
            f"https://api.{parsed.netloc}{parsed.path}",
            f"https://data.{parsed.netloc}{parsed.path}",
            f"{parsed.scheme}://{parsed.netloc}/new{parsed.path}",
            f"{parsed.scheme}://{parsed.netloc}/v2{parsed.path}"
        ]
        
        migration_urls.extend(patterns)
        return migration_urls
    
    async def _predict_by_social_intelligence(self, source_name: str) -> List[URLPrediction]:
        """Predict URLs using social media intelligence"""
        predictions = []
        
        try:
            # Check for recent social media mentions
            recent_mentions = await self._get_recent_social_mentions(source_name)
            
            for mention in recent_mentions:
                # Extract URLs from mention content
                urls = self._extract_urls_from_text(mention['content'])
                
                for url in urls:
                    confidence = mention.get('relevance_score', 0.5)
                    
                    predictions.append(URLPrediction(
                        source_name=source_name,
                        predicted_url=url,
                        confidence_score=confidence,
                        discovery_method='social_intelligence',
                        evidence=[f"Social mention: {mention['content'][:100]}..."],
                        prediction_date=datetime.now(timezone.utc)
                    ))
        
        except Exception as e:
            logger.error(f"Error in social intelligence prediction: {e}")
        
        return predictions
    
    async def _get_recent_social_mentions(self, source_name: str) -> List[Dict[str, Any]]:
        """Get recent social media mentions related to the data source"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT content, relevance_score, mention_date
                    FROM social_mentions
                    WHERE institution_mentioned LIKE ?
                    AND mention_date > datetime('now', '-30 days')
                    AND processed = FALSE
                    ORDER BY relevance_score DESC
                    LIMIT 10
                """, (f"%{source_name}%",))
                
                mentions = []
                for row in cursor:
                    mentions.append({
                        'content': row[0],
                        'relevance_score': row[1],
                        'mention_date': row[2]
                    })
                
                return mentions
                
        except Exception as e:
            logger.error(f"Error getting social mentions: {e}")
            return []
    
    def _extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text using regex"""
        url_pattern = r'https?://[^\s<>"\'{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return urls
    
    async def _predict_by_documentation(self, source_name: str, current_url: str) -> List[URLPrediction]:
        """Predict URLs by parsing documentation and announcements"""
        predictions = []
        
        try:
            # Look for documentation pages
            doc_urls = await self._find_documentation_pages(current_url)
            
            for doc_url in doc_urls:
                # Parse documentation for API endpoints
                endpoints = await self._parse_api_documentation(doc_url)
                
                for endpoint in endpoints:
                    predictions.append(URLPrediction(
                        source_name=source_name,
                        predicted_url=endpoint['url'],
                        confidence_score=endpoint['confidence'],
                        discovery_method='documentation_parsing',
                        evidence=[f"Found in documentation: {doc_url}"],
                        prediction_date=datetime.now(timezone.utc)
                    ))
        
        except Exception as e:
            logger.error(f"Error in documentation-based prediction: {e}")
        
        return predictions
    
    async def _find_documentation_pages(self, base_url: str) -> List[str]:
        """Find documentation pages for a given base URL"""
        doc_urls = []
        parsed = urlparse(base_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        
        # Common documentation paths
        doc_paths = [
            '/docs/', '/documentation/', '/api/', '/help/',
            '/guide/', '/manual/', '/readme', '/api-docs/'
        ]
        
        session = await self._get_session()
        
        for path in doc_paths:
            doc_url = urljoin(base, path)
            try:
                async with session.get(doc_url, timeout=10) as response:
                    if response.status == 200:
                        doc_urls.append(doc_url)
            except Exception as e:
                logger.debug(f"Failed to access documentation URL {doc_url}: {e}")  # ✅ IMPROVED - Better error reporting
                continue
        
        return doc_urls
    
    async def _parse_api_documentation(self, doc_url: str) -> List[Dict[str, Any]]:
        """Parse API documentation to find endpoints"""
        endpoints = []
        
        try:
            session = await self._get_session()
            async with session.get(doc_url, timeout=15) as response:
                if response.status != 200:
                    return endpoints
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Look for API endpoint patterns
                endpoint_patterns = [
                    r'https?://[^\s<>"\']+/api/[^\s<>"\']*',
                    r'/api/v\d+/[^\s<>"\']*',
                    r'GET\s+([^\s<>"\']+)',
                    r'POST\s+([^\s<>"\']+)'
                ]
                
                text_content = soup.get_text()
                
                for pattern in endpoint_patterns:
                    matches = re.findall(pattern, text_content)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0]
                        
                        # Calculate confidence based on pattern match
                        confidence = 0.7 if match.startswith('http') else 0.5
                        
                        endpoints.append({
                            'url': match,
                            'confidence': confidence
                        })
        
        except Exception as e:
            logger.error(f"Error parsing API documentation: {e}")
        
        return endpoints
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    def _calculate_pattern_confidence(self, pattern_type: str, pattern_value: str) -> float:
        """Calculate confidence score for a pattern based on historical effectiveness"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT effectiveness_score
                    FROM pattern_effectiveness
                    WHERE pattern_type = ? AND pattern_value = ?
                """, (pattern_type, pattern_value))
                
                result = cursor.fetchone()
                if result:
                    return result[0]
                else:
                    # Default confidence for new patterns
                    return 0.5
                    
        except Exception as e:
            logger.error(f"Error calculating pattern confidence: {e}")
            return 0.5
    
    def _deduplicate_predictions(self, predictions: List[URLPrediction]) -> List[URLPrediction]:
        """Remove duplicate predictions"""
        seen_urls = set()
        unique_predictions = []
        
        for prediction in predictions:
            if prediction.predicted_url not in seen_urls:
                seen_urls.add(prediction.predicted_url)
                unique_predictions.append(prediction)
        
        return unique_predictions
    
    async def _store_predictions(self, predictions: List[URLPrediction]):
        """Store predictions in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for prediction in predictions:
                    conn.execute("""
                        INSERT OR REPLACE INTO url_predictions
                        (source_name, predicted_url, confidence_score, discovery_method,
                         evidence, prediction_date)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        prediction.source_name,
                        prediction.predicted_url,
                        prediction.confidence_score,
                        prediction.discovery_method,
                        json.dumps(prediction.evidence),
                        prediction.prediction_date
                    ))
        
        except Exception as e:
            logger.error(f"Error storing predictions: {e}")
    
    async def verify_predictions(self, source_name: Optional[str] = None) -> Dict[str, Any]:
        """Verify stored predictions by testing URLs"""
        verification_results = {
            'verified_count': 0,
            'failed_count': 0,
            'pending_count': 0,
            'results': []
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                if source_name:
                    cursor = conn.execute("""
                        SELECT id, source_name, predicted_url, confidence_score, discovery_method
                        FROM url_predictions
                        WHERE source_name = ? AND verification_status = 'pending'
                        ORDER BY confidence_score DESC
                    """, (source_name,))
                else:
                    cursor = conn.execute("""
                        SELECT id, source_name, predicted_url, confidence_score, discovery_method
                        FROM url_predictions
                        WHERE verification_status = 'pending'
                        ORDER BY confidence_score DESC
                        LIMIT 50
                    """)
                
                predictions = cursor.fetchall()
                
                for prediction in predictions:
                    pred_id, src_name, pred_url, confidence, method = prediction
                    
                    # Test the predicted URL
                    is_valid = await self._test_predicted_url(pred_url)
                    
                    # Update verification status
                    if is_valid:
                        status = 'verified'
                        verification_results['verified_count'] += 1
                    else:
                        status = 'failed'
                        verification_results['failed_count'] += 1
                    
                    conn.execute("""
                        UPDATE url_predictions
                        SET verification_status = ?, verified_date = ?
                        WHERE id = ?
                    """, (status, datetime.now(timezone.utc), pred_id))
                    
                    verification_results['results'].append({
                        'source_name': src_name,
                        'predicted_url': pred_url,
                        'confidence_score': confidence,
                        'discovery_method': method,
                        'verification_status': status
                    })
        
        except Exception as e:
            logger.error(f"Error verifying predictions: {e}")
        
        return verification_results
    
    async def _test_predicted_url(self, url: str) -> bool:
        """Test if a predicted URL is valid"""
        try:
            session = await self._get_session()
            async with session.get(url, timeout=10) as response:
                return response.status in [200, 201, 202]
        except:
            return False
    
    def update_pattern_effectiveness(self, pattern_type: str, pattern_value: str, success: bool):
        """Update pattern effectiveness based on verification results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get current stats
                cursor = conn.execute("""
                    SELECT success_count, total_attempts
                    FROM pattern_effectiveness
                    WHERE pattern_type = ? AND pattern_value = ?
                """, (pattern_type, pattern_value))
                
                result = cursor.fetchone()
                
                if result:
                    success_count, total_attempts = result
                    new_success = success_count + (1 if success else 0)
                    new_total = total_attempts + 1
                    effectiveness = new_success / new_total if new_total > 0 else 0
                    
                    conn.execute("""
                        UPDATE pattern_effectiveness
                        SET success_count = ?, total_attempts = ?, 
                            effectiveness_score = ?, last_success = ?
                        WHERE pattern_type = ? AND pattern_value = ?
                    """, (
                        new_success, new_total, effectiveness,
                        datetime.now(timezone.utc) if success else None,
                        pattern_type, pattern_value
                    ))
                else:
                    # Insert new pattern
                    success_count = 1 if success else 0
                    effectiveness = success_count / 1
                    
                    conn.execute("""
                        INSERT INTO pattern_effectiveness
                        (pattern_type, pattern_value, success_count, total_attempts,
                         effectiveness_score, last_success)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        pattern_type, pattern_value, success_count, 1, effectiveness,
                        datetime.now(timezone.utc) if success else None
                    ))
        
        except Exception as e:
            logger.error(f"Error updating pattern effectiveness: {e}")
    
    async def monitor_social_media(self):
        """Monitor social media for infrastructure announcements"""
        # Placeholder for social media monitoring
        # In production, integrate with Twitter API, Reddit API, etc.
        logger.info("Social media monitoring would run here")
    
    async def close(self):
        """Close resources"""
        if self.session:
            await self.session.close()

# Global instance
predictive_discovery = None

def get_predictive_discovery() -> PredictiveURLDiscovery:
    """Get global predictive discovery instance"""
    global predictive_discovery
    if predictive_discovery is None:
        predictive_discovery = PredictiveURLDiscovery()
    return predictive_discovery

if __name__ == "__main__":
    # Test the predictive discovery system
    async def test_predictions():
        discovery = PredictiveURLDiscovery()
        
        # Test predictions for NASA Exoplanet Archive
        predictions = await discovery.predict_url_changes(
            'nasa_exoplanet_archive',
            'https://exoplanetarchive.ipac.caltech.edu'
        )
        
        print(f"Generated {len(predictions)} predictions:")
        for pred in predictions[:5]:
            print(f"  {pred.predicted_url} (confidence: {pred.confidence_score:.2f})")
        
        # Verify predictions
        results = await discovery.verify_predictions('nasa_exoplanet_archive')
        print(f"Verification: {results['verified_count']} verified, {results['failed_count']} failed")
        
        await discovery.close()
    
    asyncio.run(test_predictions()) 