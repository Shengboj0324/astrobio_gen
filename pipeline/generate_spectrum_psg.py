"""
Production-grade PSG interface with Enterprise URL Management
------------------------------------------------------------
* Reads your PSG API key from `PSG_KEY` environment variable.
* Retries on 50x/timeout with exponential back-off.
* Caches identical requests on disk (see utils.cache).
* Returns numpy arrays (wave [µm], flux) already convolved to requested R.
* Integrated with enterprise URL management for resilient NASA PSG access.
"""
from __future__ import annotations
import os, time, logging, gzip, json, pathlib
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.cache import get as cache_get, put as cache_put
import asyncio
import xml.etree.ElementTree as ET
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

# Enterprise URL system integration
import sys
sys.path.append(str(Path(__file__).parent.parent))
try:
    from utils.integrated_url_system import get_integrated_url_system
    from utils.autonomous_data_acquisition import DataPriority
    URL_SYSTEM_AVAILABLE = True
except ImportError:
    URL_SYSTEM_AVAILABLE = False
    DataPriority = None

_LOG = logging.getLogger(__name__)

# PSG API Configuration
_PSG_BASE_URL = "https://psg.gsfc.nasa.gov/api.php"  # Primary URL
_FALLBACK_API = "https://psg.gsfc.nasa.gov/api.php"  # Fallback URL

class PSGInterface:
    """Enterprise-managed PSG interface with intelligent URL routing"""
    
    def __init__(self):
        self.url_system = None
        self.current_api_url = _FALLBACK_API
        self._initialize_url_system()
    
    def _initialize_url_system(self):
        """Initialize enterprise URL management for PSG"""
        try:
            if not URL_SYSTEM_AVAILABLE:
                _LOG.info("Enterprise URL system not available, using fallback PSG API")
                return
                
            self.url_system = get_integrated_url_system()
            self.data_priority = DataPriority.HIGH  # PSG is critical for spectrum generation
            
            # Register PSG in enterprise system if not already registered
            self._register_psg_source()
            
            _LOG.info("✅ PSG interface integrated with enterprise URL system")
            
        except Exception as e:
            _LOG.warning(f"Failed to initialize enterprise URL system for PSG: {e}")
            _LOG.info("Falling back to direct PSG API access")
    
    def _register_psg_source(self):
        """Register PSG as a managed data source"""
        try:
            # Check if PSG is already registered in the URL system
            # If not, this will be handled by the URL registry
            test_url = "https://psg.gsfc.nasa.gov/api.php"
            managed_url = asyncio.run(self.url_system.get_url(test_url, self.data_priority))
            if managed_url:
                self.current_api_url = managed_url
                _LOG.info(f"PSG registered with managed URL: {managed_url}")
        except Exception as e:
            _LOG.warning(f"Could not register PSG source: {e}")
    
    async def get_psg_api_url(self) -> str:
        """Get the optimal PSG API URL using enterprise routing"""
        try:
            if self.url_system:
                # Get managed URL for PSG API
                test_url = "https://psg.gsfc.nasa.gov/api.php"
                managed_url = await self.url_system.get_url(test_url, self.data_priority)
                if managed_url:
                    return managed_url
        
        except Exception as e:
            _LOG.warning(f"Failed to get managed PSG URL: {e}")
        
        # Fallback to configured URL
        return self.current_api_url

def _cfg_xml(atmos: dict[str, float], planet: dict, instrument: str, R: int) -> str:
    gases_xml = "\n".join(f"<{g}>{mix}</{g}>" for g, mix in atmos.items())
    return f"""
<OBJECT>
  <RADIUS>{planet['radius']:.3f}</RADIUS>
  <GRAVITY>{planet.get('gravity',9.8):.2f}</GRAVITY>
</OBJECT>
<ATMOSPHERE>{gases_xml}</ATMOSPHERE>
<GENERATOR>{instrument}</GENERATOR>
<RESOLUTION>{R}</RESOLUTION>
"""

@retry(wait=wait_exponential(multiplier=2), stop=stop_after_attempt(4))
def _call_psg(cfg: str) -> str:
    """Call PSG API using enterprise-managed URL"""
    key = os.getenv("PSG_KEY")
    if not key:
        raise RuntimeError("Set environment variable PSG_KEY with your API token.")
    
    # Get current best PSG API URL
    api_url = _psg_interface.current_api_url
    
    try:
        # Use enterprise URL system if available
        if _psg_interface.url_system:
            import asyncio
            try:
                # Try to get updated URL
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                api_url = loop.run_until_complete(_psg_interface.get_psg_api_url())
                loop.close()
            except:
                pass  # Use current URL if async fails
        
        _LOG.info(f"Calling PSG API at: {api_url}")
        resp = requests.post(api_url, files={"file": ("config.xml", cfg)}, data={"key": key}, timeout=120)
        resp.raise_for_status()
        
        # Log successful access for URL health monitoring
        if _psg_interface.url_system and hasattr(_psg_interface.url_system, 'log_successful_access'):
            _psg_interface.url_system.log_successful_access("nasa_psg_api", api_url)
            
        return resp.text
        
    except Exception as e:
        # Log failed access for URL health monitoring
        if _psg_interface.url_system and hasattr(_psg_interface.url_system, 'log_failed_access'):
            _psg_interface.url_system.log_failed_access("nasa_psg_api", api_url, str(e))
        raise

def get_spectrum(atmos: dict[str,float], planet: dict,
                 instrument="JWST-NIRSpec", R=1000) -> tuple[np.ndarray,np.ndarray]:
    """Generate spectrum using enterprise-managed PSG API access"""
    request_obj = {"atm": atmos, "pl": planet["pl_name"], "inst": instrument, "R": R}
    cached = cache_get(request_obj)
    if cached:
        wave, flux = np.loadtxt(cached, unpack=True, delimiter=",")
        return wave, flux
    cfg = _cfg_xml(atmos, planet, instrument, R)
    raw = _call_psg(cfg)
    # PSG returns gzip by default; auto-detect
    if raw[:2] == "PK":
        raw = gzip.decompress(raw).decode()
    wave, flux = np.loadtxt(raw.splitlines(), unpack=True)
    # cache
    fname = f"{planet['pl_name']}_{instrument}_{R}.csv".replace(" ", "_")
    path = pathlib.Path("data/psg_outputs") / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([wave, flux]), delimiter=",")
    cache_put(request_obj, str(path))
    _LOG.info("PSG spectrum cached ➜ %s", path)
    return wave, flux