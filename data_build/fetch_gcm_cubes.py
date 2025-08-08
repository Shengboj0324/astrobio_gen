#!/usr/bin/env python3
"""
GCM Datacube Fetcher for ROCKE-3D Climate Simulations
====================================================

Fetches and processes 4-D climate datacubes for surrogate modeling.
Integrates with existing data management and quality systems.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import zarr

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GCMCubeFetcher:
    """
    Fetch and process GCM datacubes with SLURM integration
    """

    def __init__(self, output_path: str = "data/raw/gcm_cubes"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.zarr_path = Path("data/processed/gcm_zarr")
        self.zarr_path.mkdir(parents=True, exist_ok=True)

        # GCM simulation parameters
        self.default_params = {
            "n_runs": 1000,
            "model_days": 100,
            "snapshot_interval": 10,
            "resolution": "low",  # 32x32x20 grid
            "variables": ["temp", "humidity", "pressure", "wind_u", "wind_v"],
            "chunks": {"lat": 40, "lon": 40, "lev": 15, "time": 4},
        }

        self.progress_file = self.output_path / "fetch_progress.json"

    def generate_slurm_script(self, run_id: int, planet_params: Dict[str, float]) -> str:
        """Generate SLURM script for single GCM run"""
        script_content = f"""#!/bin/bash
#SBATCH --job-name=gcm_run_{run_id:04d}
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32GB
#SBATCH --output={self.output_path}/logs/run_{run_id:04d}.out
#SBATCH --error={self.output_path}/logs/run_{run_id:04d}.err

# Load modules
module load netcdf/4.7.4
module load openmpi/4.0.3

# Set environment
export OMP_NUM_THREADS=1
export PLANET_RADIUS={planet_params.get('radius', 1.0)}
export PLANET_MASS={planet_params.get('mass', 1.0)}
export STELLAR_FLUX={planet_params.get('stellar_flux', 1.0)}
export ATMOSPHERIC_CO2={planet_params.get('co2', 400)}
export SURFACE_PRESSURE={planet_params.get('pressure', 1.0)}

# Run directory
RUN_DIR={self.output_path}/run_{run_id:04d}
mkdir -p $RUN_DIR
cd $RUN_DIR

# Execute GCM simulation
echo "Starting GCM run {run_id:04d} at $(date)"
mpirun -np 8 rocke3d_exoplanet \\
    --planet-radius $PLANET_RADIUS \\
    --planet-mass $PLANET_MASS \\
    --stellar-flux $STELLAR_FLUX \\
    --co2-mixing-ratio $ATMOSPHERIC_CO2 \\
    --surface-pressure $SURFACE_PRESSURE \\
    --output-interval {self.default_params['snapshot_interval']} \\
    --total-days {self.default_params['model_days']} \\
    --resolution {self.default_params['resolution']} \\
    --netcdf-output

echo "GCM run {run_id:04d} completed at $(date)"

# Convert to zarr format
python {Path(__file__).parent}/convert_netcdf_to_zarr.py \\
    --input-dir $RUN_DIR \\
    --output-dir {self.zarr_path}/run_{run_id:04d} \\
    --chunk-config '{json.dumps(self.default_params["chunks"])}'

echo "Conversion to zarr completed for run {run_id:04d}"
"""
        return script_content

    def generate_planet_parameters(self, n_runs: int) -> List[Dict[str, float]]:
        """Generate diverse planet parameter sets for GCM runs"""
        np.random.seed(42)  # Reproducible parameter sets

        parameters = []
        for i in range(n_runs):
            # Sample from realistic parameter ranges
            params = {
                "radius": np.random.uniform(0.5, 2.5),  # Earth radii
                "mass": np.random.uniform(0.1, 5.0),  # Earth masses
                "stellar_flux": np.random.uniform(0.5, 2.0),  # Solar constants
                "co2": np.random.uniform(100, 2000),  # ppm
                "pressure": np.random.uniform(0.1, 5.0),  # bars
                "obliquity": np.random.uniform(0, 45),  # degrees
                "eccentricity": np.random.uniform(0, 0.3),
                "rotation_period": np.random.uniform(0.5, 100),  # days
            }
            parameters.append(params)

        return parameters

    async def submit_gcm_runs(self, max_concurrent: int = 20) -> Dict[str, Any]:
        """Submit GCM runs to SLURM cluster"""
        logger.info(f"Preparing {self.default_params['n_runs']} GCM runs")

        # Create necessary directories
        (self.output_path / "logs").mkdir(exist_ok=True)
        (self.output_path / "scripts").mkdir(exist_ok=True)

        # Generate planet parameters
        planet_params = self.generate_planet_parameters(self.default_params["n_runs"])

        # Save parameter sets
        params_file = self.output_path / "planet_parameters.json"
        with open(params_file, "w") as f:
            json.dump(planet_params, f, indent=2)

        logger.info(f"Saved planet parameters to {params_file}")

        # Generate and submit SLURM scripts
        submitted_jobs = []

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            for run_id, params in enumerate(planet_params):
                # Generate SLURM script
                script_content = self.generate_slurm_script(run_id, params)
                script_path = self.output_path / "scripts" / f"run_{run_id:04d}.slurm"

                with open(script_path, "w") as f:
                    f.write(script_content)

                # Submit to SLURM (if available)
                if shutil.which("sbatch"):
                    try:
                        result = subprocess.run(
                            ["sbatch", str(script_path)], capture_output=True, text=True, check=True
                        )
                        job_id = result.stdout.strip().split()[-1]
                        submitted_jobs.append(
                            {
                                "run_id": run_id,
                                "job_id": job_id,
                                "script_path": str(script_path),
                                "status": "submitted",
                                "submitted_at": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        logger.info(f"Submitted job {job_id} for run {run_id:04d}")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to submit run {run_id:04d}: {e}")
                        submitted_jobs.append(
                            {
                                "run_id": run_id,
                                "job_id": None,
                                "script_path": str(script_path),
                                "status": "failed",
                                "error": str(e),
                            }
                        )
                else:
                    logger.warning("SLURM not available - scripts generated but not submitted")
                    submitted_jobs.append(
                        {
                            "run_id": run_id,
                            "job_id": None,
                            "script_path": str(script_path),
                            "status": "script_ready",
                        }
                    )

        # Save submission results
        submission_results = {
            "total_runs": len(planet_params),
            "submitted_jobs": submitted_jobs,
            "submission_time": datetime.now(timezone.utc).isoformat(),
            "parameters": self.default_params,
        }

        results_file = self.output_path / "submission_results.json"
        with open(results_file, "w") as f:
            json.dump(submission_results, f, indent=2)

        logger.info(f"Submission complete. Results saved to {results_file}")
        return submission_results

    def convert_netcdf_to_zarr(self, netcdf_dir: Path, zarr_dir: Path) -> bool:
        """Convert NetCDF files to chunked zarr format"""
        try:
            logger.info(f"Converting {netcdf_dir} to zarr format")
            zarr_dir.mkdir(parents=True, exist_ok=True)

            # Find NetCDF files
            nc_files = list(netcdf_dir.glob("*.nc"))
            if not nc_files:
                logger.warning(f"No NetCDF files found in {netcdf_dir}")
                return False

            # Open and combine datasets
            datasets = []
            for nc_file in sorted(nc_files):
                try:
                    ds = xr.open_dataset(nc_file)
                    datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Failed to open {nc_file}: {e}")

            if not datasets:
                logger.error("No valid NetCDF datasets found")
                return False

            # Combine along time dimension
            combined_ds = xr.concat(datasets, dim="time")

            # Ensure we have the required variables
            required_vars = set(self.default_params["variables"])
            available_vars = set(combined_ds.data_vars)
            missing_vars = required_vars - available_vars

            if missing_vars:
                logger.warning(f"Missing variables: {missing_vars}")
                # Use available variables
                vars_to_save = list(available_vars.intersection(required_vars))
            else:
                vars_to_save = self.default_params["variables"]

            # Select variables and convert to zarr
            ds_subset = combined_ds[vars_to_save]

            # Apply chunking
            chunks = self.default_params["chunks"]
            ds_chunked = ds_subset.chunk(chunks)

            # Save to zarr
            zarr_store = zarr_dir / "data.zarr"
            ds_chunked.to_zarr(zarr_store, mode="w")

            # Save metadata
            metadata = {
                "source_files": [str(f) for f in nc_files],
                "variables": vars_to_save,
                "chunks": chunks,
                "shape": {var: list(ds_subset[var].shape) for var in vars_to_save},
                "converted_at": datetime.now(timezone.utc).isoformat(),
            }

            metadata_file = zarr_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Successfully converted to zarr: {zarr_store}")
            return True

        except Exception as e:
            logger.error(f"Error converting to zarr: {e}")
            return False

    def check_progress(self) -> Dict[str, Any]:
        """Check progress of GCM runs and conversions"""
        progress = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runs_submitted": 0,
            "runs_completed": 0,
            "runs_converted": 0,
            "runs_failed": 0,
            "zarr_datasets": 0,
            "total_size_gb": 0.0,
        }

        # Check submission results
        submission_file = self.output_path / "submission_results.json"
        if submission_file.exists():
            with open(submission_file, "r") as f:
                submission_data = json.load(f)
                progress["runs_submitted"] = len(submission_data["submitted_jobs"])

        # Check completed runs
        for run_dir in self.output_path.glob("run_*"):
            if run_dir.is_dir():
                nc_files = list(run_dir.glob("*.nc"))
                if nc_files:
                    progress["runs_completed"] += 1

        # Check zarr conversions
        for zarr_dir in self.zarr_path.glob("run_*"):
            if zarr_dir.is_dir() and (zarr_dir / "data.zarr").exists():
                progress["runs_converted"] += 1

                # Calculate size
                try:
                    size = sum(f.stat().st_size for f in zarr_dir.rglob("*") if f.is_file())
                    progress["total_size_gb"] += size / (1024**3)
                except Exception:
                    pass

        progress["zarr_datasets"] = progress["runs_converted"]

        return progress

    async def monitor_and_convert(self, check_interval: int = 300):
        """Monitor GCM runs and convert completed ones to zarr"""
        logger.info("Starting monitoring and conversion process")

        while True:
            try:
                progress = self.check_progress()
                logger.info(
                    f"Progress: {progress['runs_completed']}/{progress['runs_submitted']} completed, "
                    f"{progress['runs_converted']} converted to zarr"
                )

                # Find new completed runs to convert
                for run_dir in self.output_path.glob("run_*"):
                    if not run_dir.is_dir():
                        continue

                    run_id = run_dir.name
                    zarr_dir = self.zarr_path / run_id

                    # Check if already converted
                    if (zarr_dir / "data.zarr").exists():
                        continue

                    # Check if NetCDF files are available
                    nc_files = list(run_dir.glob("*.nc"))
                    if nc_files:
                        logger.info(f"Converting {run_id} to zarr")
                        success = self.convert_netcdf_to_zarr(run_dir, zarr_dir)
                        if success:
                            logger.info(f"Successfully converted {run_id}")
                        else:
                            logger.error(f"Failed to convert {run_id}")

                # Check if all runs are complete
                if (
                    progress["runs_submitted"] > 0
                    and progress["runs_converted"] >= progress["runs_submitted"]
                ):
                    logger.info("All runs completed and converted!")
                    break

                await asyncio.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                await asyncio.sleep(check_interval)


# Standalone conversion utility
def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="GCM Datacube Fetcher")
    parser.add_argument(
        "--action",
        choices=["submit", "monitor", "convert", "status"],
        default="status",
        help="Action to perform",
    )
    parser.add_argument("--input-dir", type=str, help="Input directory for conversion")
    parser.add_argument("--output-dir", type=str, help="Output directory for conversion")
    parser.add_argument("--chunk-config", type=str, help="JSON chunk configuration")
    parser.add_argument("--n-runs", type=int, default=1000, help="Number of GCM runs")

    args = parser.parse_args()

    fetcher = GCMCubeFetcher()

    if args.n_runs != 1000:
        fetcher.default_params["n_runs"] = args.n_runs

    if args.action == "submit":
        asyncio.run(fetcher.submit_gcm_runs())

    elif args.action == "monitor":
        asyncio.run(fetcher.monitor_and_convert())

    elif args.action == "convert":
        if not args.input_dir or not args.output_dir:
            logger.error("--input-dir and --output-dir required for convert action")
            return

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

        if args.chunk_config:
            chunk_config = json.loads(args.chunk_config)
            fetcher.default_params["chunks"] = chunk_config

        success = fetcher.convert_netcdf_to_zarr(input_dir, output_dir)
        logger.info(f"Conversion {'succeeded' if success else 'failed'}")

    elif args.action == "status":
        progress = fetcher.check_progress()
        print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
