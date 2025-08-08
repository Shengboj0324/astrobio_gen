#!/usr/bin/env python3
"""
Command Line Interface for Astrobio-Gen
=======================================

Production-ready CLI for training, serving, data processing, and evaluation.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def main():
    """Astrobio-Gen: World-Class Astrobiology Research Platform CLI"""
    pass


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to configuration file")
@click.option("--model", "-m", default="enhanced_datacube", help="Model to train")
@click.option("--epochs", "-e", default=100, type=int, help="Number of training epochs")
@click.option("--gpus", "-g", default=1, type=int, help="Number of GPUs to use")
@click.option("--batch-size", "-b", default=32, type=int, help="Batch size")
@click.option("--learning-rate", "-lr", default=1e-4, type=float, help="Learning rate")
@click.option("--physics-constraints", is_flag=True, help="Enable physics constraints")
@click.option("--mixed-precision", is_flag=True, help="Enable mixed precision training")
@click.option("--experiment", "-exp", help="Hydra experiment name")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--resume", type=click.Path(), help="Resume from checkpoint")
@click.option("--wandb", is_flag=True, help="Enable Weights & Biases logging")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def train(
    config,
    model,
    epochs,
    gpus,
    batch_size,
    learning_rate,
    physics_constraints,
    mixed_precision,
    experiment,
    output_dir,
    resume,
    wandb,
    debug,
):
    """Train astrobiology models with advanced features"""

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo("🚀 Starting Astrobio-Gen Training")
    click.echo(f"Model: {model}")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"GPUs: {gpus}")

    try:
        # Import training modules
        if experiment:
            # Use Hydra-based training
            from ..training.enhanced_training_orchestrator import run_hydra_training

            result = run_hydra_training(
                experiment=experiment, config_path=config, output_dir=output_dir, resume=resume
            )
        else:
            # Use direct training
            from ..training.direct_training import run_direct_training

            result = run_direct_training(
                model=model,
                epochs=epochs,
                gpus=gpus,
                batch_size=batch_size,
                learning_rate=learning_rate,
                physics_constraints=physics_constraints,
                mixed_precision=mixed_precision,
                config_path=config,
                output_dir=output_dir,
                resume=resume,
                wandb=wandb,
            )

        if result["success"]:
            click.echo("✅ Training completed successfully!")
            click.echo(f"Final metrics: {result.get('final_metrics', {})}")
            if result.get("checkpoint_path"):
                click.echo(f"Checkpoint saved: {result['checkpoint_path']}")
        else:
            click.echo(f"❌ Training failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Training error: {e}")
        if debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option("--model", "-m", default="enhanced_datacube", help="Model to serve")
@click.option(
    "--checkpoint", "-ckpt", type=click.Path(exists=True), help="Path to model checkpoint"
)
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=8000, type=int, help="Port to bind to")
@click.option("--workers", "-w", default=1, type=int, help="Number of worker processes")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--gpu", is_flag=True, help="Use GPU for inference")
@click.option("--batch-size", "-b", default=1, type=int, help="Inference batch size")
@click.option("--mixed-precision", is_flag=True, help="Enable mixed precision inference")
def serve(model, checkpoint, host, port, workers, reload, gpu, batch_size, mixed_precision):
    """Serve trained models via REST API"""

    click.echo("🌐 Starting Astrobio-Gen API Server")
    click.echo(f"Model: {model}")
    click.echo(f"Host: {host}:{port}")

    try:
        from ..api.server import create_app, start_server

        app = create_app(
            model=model,
            checkpoint=checkpoint,
            gpu=gpu,
            batch_size=batch_size,
            mixed_precision=mixed_precision,
        )

        start_server(app=app, host=host, port=port, workers=workers, reload=reload)

    except Exception as e:
        click.echo(f"❌ Server error: {e}")
        sys.exit(1)


@main.command()
@click.option("--source", "-s", multiple=True, help="Data sources to process")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option(
    "--format",
    "-f",
    default="zarr",
    type=click.Choice(["zarr", "hdf5", "netcdf"]),
    help="Output format",
)
@click.option("--workers", "-w", default=4, type=int, help="Number of worker processes")
@click.option("--chunk-size", default=1000, type=int, help="Chunk size for processing")
@click.option("--quality-check", is_flag=True, help="Enable quality checks")
@click.option("--cache", is_flag=True, help="Enable caching")
@click.option("--resume", is_flag=True, help="Resume interrupted processing")
def data(source, output_dir, format, workers, chunk_size, quality_check, cache, resume):
    """Process and prepare scientific data"""

    click.echo("📊 Starting Data Processing")
    click.echo(f"Sources: {list(source) if source else 'All configured sources'}")
    click.echo(f"Output format: {format}")

    try:
        from ..data.processing_pipeline import run_data_pipeline

        result = run_data_pipeline(
            sources=list(source) if source else None,
            output_dir=output_dir,
            output_format=format,
            workers=workers,
            chunk_size=chunk_size,
            quality_check=quality_check,
            cache=cache,
            resume=resume,
        )

        if result["success"]:
            click.echo("✅ Data processing completed!")
            click.echo(f"Processed {result.get('files_processed', 0)} files")
            click.echo(f"Output directory: {result.get('output_dir')}")
        else:
            click.echo(f"❌ Data processing failed: {result.get('error')}")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Data processing error: {e}")
        sys.exit(1)


@main.command()
@click.option("--model", "-m", default="enhanced_datacube", help="Model to evaluate")
@click.option(
    "--checkpoint", "-ckpt", type=click.Path(exists=True), help="Path to model checkpoint"
)
@click.option("--dataset", "-d", help="Dataset to evaluate on")
@click.option("--metrics", "-metric", multiple=True, help="Metrics to compute")
@click.option("--output-file", "-o", type=click.Path(), help="Output file for results")
@click.option("--batch-size", "-b", default=32, type=int, help="Evaluation batch size")
@click.option("--gpu", is_flag=True, help="Use GPU for evaluation")
@click.option("--save-predictions", is_flag=True, help="Save model predictions")
@click.option("--uncertainty", is_flag=True, help="Compute uncertainty estimates")
def eval(
    model, checkpoint, dataset, metrics, output_file, batch_size, gpu, save_predictions, uncertainty
):
    """Evaluate trained models"""

    click.echo("📈 Starting Model Evaluation")
    click.echo(f"Model: {model}")
    click.echo(f"Dataset: {dataset}")

    try:
        from ..evaluation.evaluator import run_evaluation

        result = run_evaluation(
            model=model,
            checkpoint=checkpoint,
            dataset=dataset,
            metrics=list(metrics) if metrics else None,
            batch_size=batch_size,
            gpu=gpu,
            save_predictions=save_predictions,
            uncertainty=uncertainty,
        )

        if result["success"]:
            click.echo("✅ Evaluation completed!")
            click.echo(f"Results: {result.get('metrics', {})}")

            if output_file:
                import json

                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                click.echo(f"Results saved to: {output_file}")
        else:
            click.echo(f"❌ Evaluation failed: {result.get('error')}")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Evaluation error: {e}")
        sys.exit(1)


@main.group()
def system():
    """System management commands"""
    pass


@system.command()
def status():
    """Check system status"""
    click.echo("🔍 Checking System Status...")

    try:
        from .. import check_dependencies, verify_installation

        # Check installation
        install_status = verify_installation()
        click.echo(f"Installation: {'✅' if install_status['status'] == 'success' else '❌'}")
        click.echo(f"Components: {install_status['components_available']}/6")

        # Check dependencies
        deps = check_dependencies()
        click.echo(
            f"Dependencies: {len(deps['available'])}/{len(deps['available']) + len(deps['missing'])}"
        )
        click.echo(f"Coverage: {deps['coverage']:.1%}")

        if deps["missing"]:
            click.echo(f"Missing: {', '.join(deps['missing'])}")

    except Exception as e:
        click.echo(f"❌ Status check failed: {e}")


@system.command()
def info():
    """Show package information"""
    try:
        from .. import get_package_info

        info = get_package_info()
        click.echo("📦 Astrobio-Gen Package Information")
        click.echo("=" * 40)
        click.echo(f"Name: {info['name']}")
        click.echo(f"Version: {info['version']}")
        click.echo(f"Status: {info['status']}")
        click.echo(f"Zero Error Tolerance: {info['zero_error_tolerance']}")
        click.echo(f"Real Data Only: {info['real_data_only']}")
        click.echo("\n🚀 Capabilities:")
        for capability in info["capabilities"]:
            click.echo(f"  • {capability}")

    except Exception as e:
        click.echo(f"❌ Info retrieval failed: {e}")


@system.command()
@click.option(
    "--format", "-f", default="yaml", type=click.Choice(["yaml", "json"]), help="Output format"
)
@click.option("--output", "-o", type=click.Path(), help="Output file")
def config(format, output):
    """Generate default configuration"""
    try:
        from ..config import get_default_config

        config_data = get_default_config()

        if format == "yaml":
            import yaml

            content = yaml.dump(config_data, default_flow_style=False, indent=2)
        else:
            import json

            content = json.dumps(config_data, indent=2)

        if output:
            with open(output, "w") as f:
                f.write(content)
            click.echo(f"Configuration saved to: {output}")
        else:
            click.echo(content)

    except Exception as e:
        click.echo(f"❌ Config generation failed: {e}")


# Entry points for setuptools
def train_cli():
    """Entry point for astro-train command"""
    main(["train"] + sys.argv[1:])


def serve_cli():
    """Entry point for astro-serve command"""
    main(["serve"] + sys.argv[1:])


def data_cli():
    """Entry point for astro-data command"""
    main(["data"] + sys.argv[1:])


def eval_cli():
    """Entry point for astro-eval command"""
    main(["eval"] + sys.argv[1:])


if __name__ == "__main__":
    main()
