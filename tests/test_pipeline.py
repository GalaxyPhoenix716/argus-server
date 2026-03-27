"""
Test Script for Pipeline Orchestrator

Tests the complete pipeline including anomaly detection integration,
threat classification, explainability, and event handling.
"""

import sys
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import asyncio

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from app.services.pipeline_orchestrator import PipelineOrchestrator, PipelineStage
from app.ai.threat_classifier import ThreatClassifier
from app.ai.explainability_engine import ExplainabilityEngine
from app.ai.config.threat_config import ThreatClassificationConfig

def create_sample_telemetry_data():
    """Create sample telemetry data for multiple channels"""
    np.random.seed(42)

    channels = {
        "P-1": np.random.randn(500, 25) * 0.1,
        "P-2": np.random.randn(500, 25) * 0.1,
        "A-7": np.random.randn(500, 25) * 0.1,
    }

    # Add structure to data
    for channel_id, data in channels.items():
        # Position
        data[:, 0] = np.cumsum(np.random.randn(500) * 0.01)
        data[:, 1] = np.cumsum(np.random.randn(500) * 0.01)
        # Velocity
        data[:, 2] = np.random.randn(500) * 0.5
        data[:, 3] = np.random.randn(500) * 0.5
        # Altitude
        data[:, 4] = 100 + np.random.randn(500) * 5

    # Inject anomaly in P-1
    channels["P-1"][200:250, 0] += 0.5  # Position anomaly
    channels["P-1"][200:250, 1] += 0.5
    channels["P-1"][200:250, 2] += 2.0  # Velocity anomaly

    return channels

def create_anomaly_result(channel_id):
    """Create sample anomaly result"""
    class AnomalyResult:
        def __init__(self, has_anomaly=True):
            if has_anomaly:
                self.anomaly_sequences = [(200, 250)]
            else:
                self.anomaly_sequences = []
            self.threshold = 0.5
            self.errors_smoothed = np.random.randn(500) * 0.1

    # P-1 has anomaly, others don't
    has_anomaly = (channel_id == "P-1")
    return AnomalyResult(has_anomaly)

async def test_single_channel_pipeline():
    """Test pipeline processing for a single channel"""
    print("=" * 80)
    print("TEST 1: Single Channel Pipeline")
    print("=" * 80)

    # Initialize orchestrator
    config = ThreatClassificationConfig()
    orchestrator = PipelineOrchestrator(config=config)

    # Create sample data
    channels = create_sample_telemetry_data()
    channel_id = "P-1"
    data = channels[channel_id]
    anomaly_result = create_anomaly_result(channel_id)

    print(f"\nProcessing channel: {channel_id}")
    print(f"Data shape: {data.shape}")

    # Process through pipeline
    print(f"\nRunning pipeline...")
    result = await orchestrator.process_telemetry(
        channel_id=channel_id,
        data=data,
        anomaly_detector_result=anomaly_result
    )

    # Display results
    print(f"\nPipeline Results:")
    print(f"- Anomaly Detected: {result.anomaly_detected}")
    print(f"- Anomaly Score: {result.anomaly_score:.4f}")

    if result.anomaly_window:
        print(f"- Anomaly Window: {result.anomaly_window}")

    if result.threat_classification:
        print(f"\n- Threat Classification:")
        print(f"  • Class: {result.threat_classification.threat_class}")
        print(f"  • Confidence: {result.threat_classification.confidence:.4f}")
        print(f"  • Risk Score: {result.threat_classification.risk_score:.4f}")

    if result.explanation:
        print(f"\n- Explainability:")
        print(f"  • Pattern Type: {result.explanation.pattern_type}")
        print(f"  • Confidence: {result.explanation.confidence_explanation:.4f}")

    print(f"\n- Processing Time: {result.total_processing_time_ms:.2f} ms")
    print(f"- Success: {result.success}")

    print(f"\n- Events ({len(result.events)}):")
    for event in result.events:
        print(f"  • {event.stage.value}: {event.processing_time_ms:.2f} ms - {event.data}")

    print(f"\n✓ Single channel pipeline test successful")

    return result

async def test_batch_pipeline():
    """Test batch pipeline processing"""
    print("\n" + "=" * 80)
    print("TEST 2: Batch Pipeline Processing")
    print("=" * 80)

    # Initialize orchestrator
    config = ThreatClassificationConfig()
    orchestrator = PipelineOrchestrator(config=config)

    # Create sample data for multiple channels
    channels = create_sample_telemetry_data()
    anomaly_results = {
        channel_id: create_anomaly_result(channel_id)
        for channel_id in channels.keys()
    }

    print(f"\nProcessing {len(channels)} channels in batch")

    # Process batch
    results = await orchestrator.batch_process(
        channels=channels,
        anomaly_results=anomaly_results
    )

    # Display results
    print(f"\nBatch Results:")
    for channel_id, result in results.items():
        print(f"\n{channel_id}:")
        print(f"  • Anomaly: {result.anomaly_detected}")
        print(f"  • Score: {result.anomaly_score:.4f}")
        print(f"  • Processing time: {result.total_processing_time_ms:.2f} ms")

        if result.threat_classification:
            print(f"  • Classification: {result.threat_classification.threat_class}")
            print(f"  • Confidence: {result.threat_classification.confidence:.4f}")

    print(f"\n✓ Batch pipeline test successful")

    return results

def test_pipeline_info():
    """Test pipeline information"""
    print("\n" + "=" * 80)
    print("TEST 3: Pipeline Information")
    print("=" * 80)

    # Initialize orchestrator
    config = ThreatClassificationConfig()
    orchestrator = PipelineOrchestrator(config=config)

    # Get pipeline info
    info = orchestrator.get_pipeline_info()

    print(f"\nPipeline Information:")
    print(f"- Stages: {', '.join(info['stages'])}")
    print(f"- Has Threat Classifier: {info['has_threat_classifier']}")
    print(f"- Has Explainability Engine: {info['has_explainability_engine']}")

    print(f"\n- Performance Targets:")
    for key, value in info['performance_targets'].items():
        print(f"  • {key}: {value} ms")

    print(f"\n- Event Handlers: {', '.join(info['event_handlers'])}")
    print(f"- Total Events Processed: {info['total_events_processed']}")

    print(f"\n✓ Pipeline info test successful")

    return info

def test_processing_stats():
    """Test processing statistics"""
    print("\n" + "=" * 80)
    print("TEST 4: Processing Statistics")
    print("=" * 80)

    # Initialize orchestrator
    config = ThreatClassificationConfig()
    orchestrator = PipelineOrchestrator(config=config)

    # Get processing stats
    stats = orchestrator.get_processing_stats()

    print(f"\nProcessing Statistics:")
    if stats:
        for stage, stat in stats.items():
            print(f"\n{stage}:")
            print(f"  • Mean: {stat['mean_ms']:.2f} ms")
            print(f"  • Std: {stat['std_ms']:.2f} ms")
            print(f"  • Min: {stat['min_ms']:.2f} ms")
            print(f"  • Max: {stat['max_ms']:.2f} ms")
            print(f"  • Count: {stat['count']}")
    else:
        print("  No statistics available (no processing completed yet)")

    print(f"\n✓ Processing stats test successful")

    return stats

async def run_all_tests():
    """Run all pipeline tests"""
    print("\n" + "=" * 80)
    print("PIPELINE ORCHESTRATOR TEST SUITE")
    print("=" * 80)
    print(f"Test Date: {datetime.now().isoformat()}")
    print(f"Test Environment: Python {sys.version.split()[0]}")

    results = {
        "test_date": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "tests": {}
    }

    # Test 1: Single Channel Pipeline
    try:
        single_result = await test_single_channel_pipeline()
        results["tests"]["single_channel_pipeline"] = {
            "status": "passed",
            "anomaly_detected": single_result.anomaly_detected,
            "total_processing_time_ms": single_result.total_processing_time_ms,
            "n_events": len(single_result.events)
        }
    except Exception as e:
        print(f"\n❌ Single channel pipeline test failed: {e}")
        results["tests"]["single_channel_pipeline"] = {"status": "failed", "error": str(e)}

    # Test 2: Batch Pipeline
    try:
        batch_results = await test_batch_pipeline()
        results["tests"]["batch_pipeline"] = {
            "status": "passed",
            "n_channels": len(batch_results),
            "channels_processed": list(batch_results.keys())
        }
    except Exception as e:
        print(f"\n❌ Batch pipeline test failed: {e}")
        results["tests"]["batch_pipeline"] = {"status": "failed", "error": str(e)}

    # Test 3: Pipeline Info
    try:
        info_result = test_pipeline_info()
        results["tests"]["pipeline_info"] = {
            "status": "passed",
            "n_stages": len(info_result["stages"]),
            "has_threat_classifier": info_result["has_threat_classifier"],
            "has_explainability_engine": info_result["has_explainability_engine"]
        }
    except Exception as e:
        print(f"\n❌ Pipeline info test failed: {e}")
        results["tests"]["pipeline_info"] = {"status": "failed", "error": str(e)}

    # Test 4: Processing Stats
    try:
        stats_result = test_processing_stats()
        results["tests"]["processing_stats"] = {
            "status": "passed",
            "has_stats": len(stats_result) > 0,
            "n_stages": len(stats_result)
        }
    except Exception as e:
        print(f"\n❌ Processing stats test failed: {e}")
        results["tests"]["processing_stats"] = {"status": "failed", "error": str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(1 for t in results["tests"].values() if t["status"] == "passed")
    total = len(results["tests"])
    print(f"Tests Passed: {passed}/{total}")

    for test_name, result in results["tests"].items():
        status_icon = "✓" if result["status"] == "passed" else "❌"
        print(f"{status_icon} {test_name}: {result['status']}")

    # Performance Summary
    if "single_channel_pipeline" in results["tests"] and results["tests"]["single_channel_pipeline"]["status"] == "passed":
        single_result = results["tests"]["single_channel_pipeline"]
        print(f"\nPerformance Summary:")
        print(f"  • Processing time: {single_result['total_processing_time_ms']:.2f} ms")
        print(f"  • Target: <100 ms")
        print(f"  • Status: {'✓ PASS' if single_result['total_processing_time_ms'] < 100 else '⚠ SLOW'}")

    # Save results
    output_file = Path(__file__).parent / "pipeline_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Test results saved to: {output_file}")

    return results

if __name__ == "__main__":
    results = asyncio.run(run_all_tests())