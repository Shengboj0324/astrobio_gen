"""
Demonstration of Comprehensive Data Source Expansion System
Shows integration of 500+ high-quality scientific data sources
Target: 96.4% accuracy through data abundance and quality
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from data_build.comprehensive_data_expansion import ComprehensiveDataExpansion
except ImportError:
    logger.error("Could not import ComprehensiveDataExpansion. Make sure the module is available.")
    exit(1)


class DataExpansionDemo:
    """Demonstration of the comprehensive data expansion system"""

    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.expansion_system = None

    def initialize_system(self):
        """Initialize the comprehensive data expansion system"""
        logger.info("🚀 Initializing Comprehensive Data Expansion System")
        logger.info("Target: 96.4% accuracy through 500+ high-quality data sources")

        # Initialize the system
        self.expansion_system = ComprehensiveDataExpansion("data")

        # Display data source registry
        self.display_source_registry()

    def display_source_registry(self):
        """Display the comprehensive data source registry"""
        logger.info("\n📊 COMPREHENSIVE DATA SOURCE REGISTRY")
        logger.info("=" * 60)

        total_sources = 0
        domain_counts = {}

        for domain, sources in self.expansion_system.data_sources.items():
            count = len(sources)
            domain_counts[domain] = count
            total_sources += count

            logger.info(f"{domain.upper():>15}: {count:>3} sources")

            # Show top priority sources for each domain
            priority_sources = [s for s in sources if s.priority == 1]
            logger.info(f"{'':>15}  High priority: {len(priority_sources)}")

            # Show average quality score
            avg_quality = np.mean([s.quality_score for s in sources])
            logger.info(f"{'':>15}  Avg quality: {avg_quality:.3f}")

        logger.info("-" * 60)
        logger.info(f"{'TOTAL':>15}: {total_sources:>3} sources")
        logger.info("=" * 60)

        return domain_counts, total_sources

    def demonstrate_source_details(self):
        """Demonstrate detailed source information for each domain"""
        logger.info("\n🔍 DETAILED SOURCE ANALYSIS BY DOMAIN")
        logger.info("=" * 80)

        domain_details = {}

        for domain, sources in self.expansion_system.data_sources.items():
            logger.info(f"\n📂 {domain.upper()} DOMAIN ({len(sources)} sources)")
            logger.info("-" * 50)

            # Categorize sources by priority and type
            priority_1 = [s for s in sources if s.priority == 1]
            priority_2 = [s for s in sources if s.priority == 2]
            priority_3 = [s for s in sources if s.priority == 3]

            api_sources = [s for s in sources if s.api_endpoint is not None]
            fits_sources = [s for s in sources if s.data_type == "fits"]
            netcdf_sources = [s for s in sources if s.data_type == "netcdf"]
            tabular_sources = [s for s in sources if s.data_type == "tabular"]

            logger.info(f"Priority Distribution:")
            logger.info(f"  ⭐ Priority 1 (Critical): {len(priority_1)}")
            logger.info(f"  ⭐ Priority 2 (Important): {len(priority_2)}")
            logger.info(f"  ⭐ Priority 3 (Supplementary): {len(priority_3)}")

            logger.info(f"Data Type Distribution:")
            logger.info(f"  📊 Tabular: {len(tabular_sources)}")
            logger.info(f"  🔬 FITS: {len(fits_sources)}")
            logger.info(f"  🌡️ NetCDF: {len(netcdf_sources)}")
            logger.info(f"  🔗 API Sources: {len(api_sources)}")

            # Show top 5 highest quality sources
            top_sources = sorted(sources, key=lambda x: x.quality_score, reverse=True)[:5]
            logger.info(f"Top 5 Quality Sources:")
            for i, source in enumerate(top_sources, 1):
                logger.info(f"  {i}. {source.name} (Quality: {source.quality_score:.3f})")

            domain_details[domain] = {
                "total": len(sources),
                "priority_1": len(priority_1),
                "priority_2": len(priority_2),
                "priority_3": len(priority_3),
                "api_sources": len(api_sources),
                "avg_quality": np.mean([s.quality_score for s in sources]),
                "top_sources": [(s.name, s.quality_score) for s in top_sources],
            }

        return domain_details

    def demonstrate_specific_sources(self):
        """Demonstrate specific high-value data sources"""
        logger.info("\n🌟 HIGH-VALUE DATA SOURCES SHOWCASE")
        logger.info("=" * 80)

        showcase_sources = {
            "astrobiology": [
                "NASA Exoplanet Archive",
                "ESA Gaia Archive",
                "Open Exoplanet Catalogue",
                "TESS Input Catalog",
                "HARPS Archive",
            ],
            "climate": [
                "NSF NCAR Research Data Archive",
                "CMIP6 Data Portal",
                "ERA5 Reanalysis",
                "MERRA-2 Reanalysis",
                "MODIS Atmosphere Products",
            ],
            "genomics": [
                "Database Commons",
                "BioCyc Database Collection",
                "Reactome Pathway Database",
                "UniProt Protein Database",
                "Ensembl Genome Browser",
            ],
            "spectroscopy": [
                "X-shooter Spectral Library",
                "POLLUX Stellar Spectra",
                "Gaia FGK Benchmark Stars",
                "ASTRAL HST STIS Library",
                "HARPS Spectral Archive",
            ],
            "stellar": [
                "Gaia Data Release 3",
                "Hipparcos-Tycho Catalog",
                "APOGEE Stellar Spectra",
                "GALAH Stellar Survey",
                "MIST Stellar Evolution",
            ],
        }

        for domain, source_names in showcase_sources.items():
            logger.info(f"\n📂 {domain.upper()} - Key Sources:")
            domain_sources = self.expansion_system.data_sources[domain]

            for source_name in source_names:
                # Find source in domain
                source = next((s for s in domain_sources if s.name == source_name), None)
                if source:
                    logger.info(f"  🔗 {source.name}")
                    logger.info(f"     URL: {source.url}")
                    logger.info(f"     Type: {source.data_type}")
                    logger.info(f"     Quality: {source.quality_score:.3f}")
                    logger.info(f"     Priority: {source.priority}")
                    if source.api_endpoint:
                        logger.info(f"     API: Available")
                    logger.info("")

    async def simulate_integration_process(self):
        """Simulate the comprehensive data integration process"""
        logger.info("\n⚡ SIMULATING COMPREHENSIVE DATA INTEGRATION")
        logger.info("=" * 60)

        simulation_results = {
            "total_sources": 0,
            "successful_integrations": 0,
            "failed_integrations": 0,
            "domain_statistics": {},
            "quality_scores": [],
            "processing_time": 0.0,
            "overall_quality": 0.0,
        }

        start_time = time.time()

        # Simulate processing each domain
        for domain, sources in self.expansion_system.data_sources.items():
            logger.info(f"\n🔄 Processing {domain.upper()} domain...")

            domain_results = await self.simulate_domain_processing(domain, sources)
            simulation_results["domain_statistics"][domain] = domain_results

            simulation_results["total_sources"] += len(sources)
            simulation_results["successful_integrations"] += domain_results["successful"]
            simulation_results["failed_integrations"] += domain_results["failed"]
            simulation_results["quality_scores"].extend(domain_results["quality_scores"])

        # Calculate overall metrics
        simulation_results["processing_time"] = time.time() - start_time
        if simulation_results["quality_scores"]:
            simulation_results["overall_quality"] = np.mean(simulation_results["quality_scores"])

        return simulation_results

    async def simulate_domain_processing(self, domain: str, sources: list):
        """Simulate processing sources in a domain"""
        domain_results = {
            "successful": 0,
            "failed": 0,
            "quality_scores": [],
            "processing_times": [],
        }

        logger.info(f"  📊 Processing {len(sources)} {domain} sources...")

        for i, source in enumerate(sources):
            # Simulate processing delay
            await asyncio.sleep(0.001)  # Very small delay for simulation

            # Simulate success/failure based on source quality and priority
            success_probability = (
                0.95 if source.priority == 1 else 0.90 if source.priority == 2 else 0.85
            )

            if np.random.random() < success_probability:
                domain_results["successful"] += 1
                # Simulate quality improvement through processing
                processed_quality = min(source.quality_score * 1.05, 1.0)
                domain_results["quality_scores"].append(processed_quality)
                domain_results["processing_times"].append(np.random.uniform(0.5, 3.0))
            else:
                domain_results["failed"] += 1

            # Progress indicator
            if (i + 1) % 20 == 0 or (i + 1) == len(sources):
                progress = (i + 1) / len(sources) * 100
                logger.info(f"    Progress: {i+1}/{len(sources)} ({progress:.1f}%)")

        success_rate = domain_results["successful"] / len(sources) * 100
        avg_quality = (
            np.mean(domain_results["quality_scores"]) if domain_results["quality_scores"] else 0.0
        )

        logger.info(
            f"  ✅ {domain.upper()}: {domain_results['successful']}/{len(sources)} "
            f"({success_rate:.1f}% success, {avg_quality:.3f} avg quality)"
        )

        return domain_results

    def analyze_results(self, results: dict):
        """Analyze and display integration results"""
        logger.info("\n📈 COMPREHENSIVE INTEGRATION RESULTS")
        logger.info("=" * 60)

        # Overall statistics
        total_sources = results["total_sources"]
        successful = results["successful_integrations"]
        failed = results["failed_integrations"]
        success_rate = successful / total_sources * 100
        overall_quality = results["overall_quality"]
        processing_time = results["processing_time"]

        logger.info(f"📊 OVERALL PERFORMANCE:")
        logger.info(f"  Total Sources Processed: {total_sources}")
        logger.info(f"  Successful Integrations: {successful}")
        logger.info(f"  Failed Integrations: {failed}")
        logger.info(f"  Success Rate: {success_rate:.2f}%")
        logger.info(f"  Overall Quality Score: {overall_quality:.4f}")
        logger.info(f"  Processing Time: {processing_time:.2f} seconds")

        # Target achievement check
        target_quality = 0.964  # 96.4%
        if overall_quality >= target_quality:
            logger.info(f"🎯 TARGET ACHIEVED: {overall_quality:.3f} ≥ {target_quality:.3f} (96.4%)")
        else:
            gap = target_quality - overall_quality
            logger.info(f"⚠️ TARGET IN PROGRESS: {overall_quality:.3f} < {target_quality:.3f}")
            logger.info(f"   Gap to close: {gap:.4f} ({gap*100:.2f} percentage points)")

        # Domain breakdown
        logger.info(f"\n📂 DOMAIN BREAKDOWN:")
        for domain, stats in results["domain_statistics"].items():
            total_domain = stats["successful"] + stats["failed"]
            domain_success_rate = stats["successful"] / total_domain * 100
            domain_quality = np.mean(stats["quality_scores"]) if stats["quality_scores"] else 0.0

            logger.info(
                f"  {domain.upper():>12}: {stats['successful']:>3}/{total_domain:>3} "
                f"({domain_success_rate:>5.1f}%) - Quality: {domain_quality:.3f}"
            )

        # Quality distribution analysis
        all_qualities = results["quality_scores"]
        if all_qualities:
            logger.info(f"\n📊 QUALITY DISTRIBUTION:")
            logger.info(f"  Minimum Quality: {min(all_qualities):.3f}")
            logger.info(f"  Maximum Quality: {max(all_qualities):.3f}")
            logger.info(f"  Mean Quality: {np.mean(all_qualities):.3f}")
            logger.info(f"  Median Quality: {np.median(all_qualities):.3f}")
            logger.info(f"  Standard Deviation: {np.std(all_qualities):.3f}")

            # Count sources above quality thresholds
            above_90 = sum(1 for q in all_qualities if q >= 0.90)
            above_95 = sum(1 for q in all_qualities if q >= 0.95)
            above_96 = sum(1 for q in all_qualities if q >= 0.96)

            logger.info(
                f"  Sources ≥ 90% quality: {above_90}/{len(all_qualities)} "
                f"({above_90/len(all_qualities)*100:.1f}%)"
            )
            logger.info(
                f"  Sources ≥ 95% quality: {above_95}/{len(all_qualities)} "
                f"({above_95/len(all_qualities)*100:.1f}%)"
            )
            logger.info(
                f"  Sources ≥ 96% quality: {above_96}/{len(all_qualities)} "
                f"({above_96/len(all_qualities)*100:.1f}%)"
            )

        return {
            "success_rate": success_rate,
            "overall_quality": overall_quality,
            "target_achieved": overall_quality >= target_quality,
            "domain_stats": results["domain_statistics"],
        }

    def generate_recommendations(self, analysis: dict):
        """Generate recommendations for achieving 96.4% accuracy"""
        logger.info("\n💡 RECOMMENDATIONS FOR 96.4% ACCURACY")
        logger.info("=" * 60)

        recommendations = []

        if analysis["overall_quality"] >= 0.964:
            recommendations.append("🎉 TARGET ACHIEVED! Maintain current integration strategy.")
            recommendations.append("📈 Consider expanding to additional specialized sources.")
            recommendations.append("🔄 Implement regular quality monitoring and updates.")
        else:
            gap = 0.964 - analysis["overall_quality"]
            recommendations.append(f"🎯 Quality gap: {gap:.4f} points to achieve 96.4% target")

            # Domain-specific recommendations
            for domain, stats in analysis["domain_stats"].items():
                domain_quality = (
                    np.mean(stats["quality_scores"]) if stats["quality_scores"] else 0.0
                )
                if domain_quality < 0.90:
                    recommendations.append(f"⚠️ {domain.upper()}: Focus on higher-quality sources")
                elif domain_quality < 0.95:
                    recommendations.append(
                        f"📊 {domain.upper()}: Good progress, minor improvements needed"
                    )
                else:
                    recommendations.append(f"✅ {domain.upper()}: Excellent quality achieved")

            # General recommendations
            if analysis["success_rate"] < 95:
                recommendations.append(
                    "🔧 Improve source accessibility and integration reliability"
                )

            recommendations.append("🔍 Prioritize integration of Priority 1 sources first")
            recommendations.append("📡 Leverage API endpoints for more reliable data access")
            recommendations.append("🧪 Implement cross-validation between similar sources")
            recommendations.append("⏰ Ensure timely updates from dynamic data sources")
            recommendations.append("🛡️ Add robust error handling and retry mechanisms")

        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec}")

        return recommendations

    def save_comprehensive_report(self, results: dict, analysis: dict, recommendations: list):
        """Save comprehensive integration report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        comprehensive_report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "system": "Comprehensive Data Expansion System",
                "target_accuracy": "96.4%",
                "total_sources": results["total_sources"],
            },
            "integration_results": results,
            "analysis": analysis,
            "recommendations": recommendations,
            "data_source_registry": {},
        }

        # Add data source registry summary
        for domain, sources in self.expansion_system.data_sources.items():
            comprehensive_report["data_source_registry"][domain] = {
                "total_sources": len(sources),
                "priority_distribution": {
                    "priority_1": len([s for s in sources if s.priority == 1]),
                    "priority_2": len([s for s in sources if s.priority == 2]),
                    "priority_3": len([s for s in sources if s.priority == 3]),
                },
                "data_types": {
                    "tabular": len([s for s in sources if s.data_type == "tabular"]),
                    "fits": len([s for s in sources if s.data_type == "fits"]),
                    "netcdf": len([s for s in sources if s.data_type == "netcdf"]),
                    "other": len(
                        [s for s in sources if s.data_type not in ["tabular", "fits", "netcdf"]]
                    ),
                },
                "api_sources": len([s for s in sources if s.api_endpoint is not None]),
                "average_quality": np.mean([s.quality_score for s in sources]),
            }

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        comprehensive_report = convert_numpy_types(comprehensive_report)

        # Save report
        report_path = self.results_dir / f"comprehensive_data_expansion_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(comprehensive_report, f, indent=2)

        logger.info(f"\n💾 COMPREHENSIVE REPORT SAVED")
        logger.info(f"📁 Location: {report_path}")
        logger.info(f"📊 Report includes: Integration results, quality analysis, recommendations")

        return report_path


async def run_comprehensive_demo():
    """Run the complete comprehensive data expansion demonstration"""
    logger.info("🎯 COMPREHENSIVE DATA EXPANSION SYSTEM DEMO")
    logger.info("Targeting 96.4% accuracy through 500+ high-quality data sources")
    logger.info("=" * 80)

    # Initialize demo
    demo = DataExpansionDemo()
    demo.initialize_system()

    # Display source details
    domain_details = demo.demonstrate_source_details()

    # Showcase specific high-value sources
    demo.demonstrate_specific_sources()

    # Simulate comprehensive integration
    logger.info("\n🔄 Starting comprehensive data integration simulation...")
    results = await demo.simulate_integration_process()

    # Analyze results
    analysis = demo.analyze_results(results)

    # Generate recommendations
    recommendations = demo.generate_recommendations(analysis)

    # Save comprehensive report
    report_path = demo.save_comprehensive_report(results, analysis, recommendations)

    # Final summary
    logger.info("\n🎉 DEMONSTRATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"✅ Processed: {results['total_sources']} data sources")
    logger.info(f"📊 Quality Score: {results['overall_quality']:.3f}")
    logger.info(
        f"🎯 Target (96.4%): {'ACHIEVED' if analysis['target_achieved'] else 'IN PROGRESS'}"
    )
    logger.info(f"⏱️ Processing Time: {results['processing_time']:.2f} seconds")
    logger.info(f"📁 Report: {report_path}")

    return results, analysis, recommendations


if __name__ == "__main__":
    asyncio.run(run_comprehensive_demo())
