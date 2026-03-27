"""
Test Script for Explainability Layer

Tests the explainability system including temporal pattern analysis,
explanation templates, and the explainability engine.
"""

import sys
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from app.ai.temporal_analyzer import TemporalPatternDetector, TemporalPattern
from app.ai.explanation_templates import ExplanationTemplates, TemplateContext
from app.ai.explainability_engine import ExplainabilityEngine
from app.ai.config.threat_config import ExplainabilityConfig

def create_sample_error_signal(pattern_type="spike"):
    """Create sample error signal for testing"""
    np.random.seed(42)
    n_points = 100

    if pattern_type == "spike":
        # Sudden spike
        errors = np.random.randn(n_points) * 0.1
        errors[40:50] = np.random.randn(10) * 3.0
    elif pattern_type == "drift":
        # Gradual drift
        errors = np.random.randn(n_points) * 0.1 + np.linspace(0, 2, n_points)
    elif pattern_type == "persistent":
        # Persistent deviation
        errors = np.random.randn(n_points) * 0.5 + 1.5
    elif pattern_type == "intermittent":
        # Multiple bursts
        errors = np.random.randn(n_points) * 0.1
        errors[20:25] = np.random.randn(5) * 2.0
        errors[60:65] = np.random.randn(5) * 2.5
    else:
        errors = np.random.randn(n_points) * 0.1

    return errors

def create_sample_features():
    """Create sample features for testing"""
    class SimpleFeatures:
        def __init__(self):
            self.feature_names = [
                'gps_velocity_mismatch',
                'position_velocity_correlation',
                'sensor_cross_validation_error',
                'mean_absolute_error',
                'max_error',
                'change_point_score',
                'volatility_index',
                'inter_sensor_correlation'
            ]
            self.values = np.random.randn(len(self.feature_names))

        def to_array(self):
            return self.values

        def get_feature_names(self):
            return self.feature_names

    return SimpleFeatures()

def create_sample_classification(threat_class="attack", confidence=0.85):
    """Create sample classification result"""
    class ClassificationResult:
        def __init__(self, threat_class, confidence):
            self.threat_class = threat_class
            self.confidence = confidence
            self.risk_score = confidence * 0.9
            self.specific_threat_type = f"{threat_class}_type_1"
            self.top_features = []

    return ClassificationResult(threat_class, confidence)

def test_temporal_pattern_detection():
    """Test temporal pattern detection"""
    print("=" * 80)
    print("TEST 1: Temporal Pattern Detection")
    print("=" * 80)

    # Initialize detector
    detector = TemporalPatternDetector()

    patterns_to_test = ["spike", "drift", "persistent", "intermittent"]
    results = {}

    for pattern in patterns_to_test:
        print(f"\n--- Testing {pattern.upper()} pattern ---")

        # Create sample data
        errors = create_sample_error_signal(pattern)
        window = (20, 70)
        threshold = 1.0

        # Analyze pattern
        analysis = detector.analyze_pattern(errors, window, threshold)

        print(f"Pattern Type: {analysis.pattern_type.value}")
        print(f"Duration: {analysis.duration} timesteps")
        print(f"Intensity: {analysis.intensity:.4f}")
        print(f"Trend Slope: {analysis.trend_slope:.4f}")
        print(f"Stability Score: {analysis.stability_score:.4f}")
        print(f"Volatility Index: {analysis.volatility_index:.4f}")
        print(f"Change Point Score: {analysis.change_point_score:.4f}")

        # Get pattern summary
        summary = detector.get_pattern_summary(analysis)
        print(f"\nPattern Summary:")
        for key, value in summary.items():
            print(f"  • {key}: {value}")

        results[pattern] = {
            "detected_pattern": analysis.pattern_type.value,
            "duration": analysis.duration,
            "intensity": analysis.intensity,
            "stability_score": analysis.stability_score,
            "volatility_index": analysis.volatility_index
        }

    print(f"\n✓ Temporal pattern detection test successful")
    return results

def test_explanation_templates():
    """Test explanation templates"""
    print("\n" + "=" * 80)
    print("TEST 2: Explanation Templates")
    print("=" * 80)

    # Initialize templates
    templates = ExplanationTemplates(detail_level="detailed")

    # Test cases
    test_cases = [
        {
            "name": "GPS Spoofing Attack",
            "context": TemplateContext(
                channel_id="P-1",
                threat_type="attack",
                confidence=0.92,
                risk_score=0.85,
                top_features=[
                    {"name": "gps_velocity_mismatch", "value": 0.8, "importance": 0.95},
                    {"name": "position_velocity_correlation", "value": 0.3, "importance": 0.87},
                    {"name": "acceleration_consistency", "value": 0.4, "importance": 0.75}
                ],
                specific_type="gps_spoofing"
            )
        },
        {
            "name": "Sensor Drift Failure",
            "context": TemplateContext(
                channel_id="A-7",
                threat_type="failure",
                confidence=0.78,
                risk_score=0.62,
                top_features=[
                    {"name": "trend_r_squared", "value": 0.85, "importance": 0.92},
                    {"name": "trend_slope", "value": 0.004, "importance": 0.88},
                    {"name": "stability_score", "value": 0.65, "importance": 0.70}
                ],
                specific_type="sensor_drift"
            )
        },
        {
            "name": "Communication Loss",
            "context": TemplateContext(
                channel_id="P-2",
                threat_type="failure",
                confidence=0.88,
                risk_score=0.75,
                top_features=[
                    {"name": "correlation_breakdown_score", "value": 0.9, "importance": 0.95},
                    {"name": "stability_score", "value": 0.92, "importance": 0.90},
                    {"name": "inter_sensor_correlation", "value": 0.2, "importance": 0.85}
                ],
                specific_type="communication_loss"
            )
        }
    ]

    results = {}

    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")

        # Generate explanation
        explanation = templates.generate_explanation(test_case['context'])

        # Show summary
        summary = templates.generate_summary(test_case['context'])

        print(f"\nExplanation Summary:")
        print(f"  {summary}")

        print(f"\nExplanation Preview (first 300 chars):")
        print(f"  {explanation[:300]}...")

        results[test_case['name']] = {
            "summary": summary,
            "length": len(explanation),
            "specific_type": test_case['context'].specific_type
        }

    print(f"\n✓ Explanation templates test successful")
    return results

def test_explainability_engine():
    """Test explainability engine"""
    print("\n" + "=" * 80)
    print("TEST 3: Explainability Engine")
    print("=" * 80)

    # Initialize engine
    engine = ExplainabilityEngine()

    # Create sample data
    features = create_sample_features()
    classification = create_sample_classification("attack", 0.85)
    error_signal = create_sample_error_signal("spike")
    anomaly_window = (20, 70)

    print(f"\nGenerating explanation...")

    # Generate explanation
    result = engine.explain(
        classification_result=classification,
        features=features,
        anomaly_window=anomaly_window,
        error_signal=error_signal.tolist(),
        detection_threshold=1.0,
        channel_id="P-1"
    )

    print(f"\nExplainability Results:")
    print(f"- Pattern Type: {result.pattern_type}")
    print(f"- Confidence: {result.confidence_explanation:.4f}")
    print(f"- Processing Time: {result.processing_time_ms:.2f} ms")

    print(f"\n- Top Features:")
    for feature in result.top_features[:5]:
        print(f"  • {feature['name']}: {feature['importance']:.4f}")

    print(f"\n- Temporal Analysis:")
    print(f"  • Pattern: {result.temporal_analysis.pattern_type.value}")
    print(f"  • Duration: {result.temporal_analysis.duration}")
    print(f"  • Intensity: {result.temporal_analysis.intensity:.4f}")
    print(f"  • Stability: {result.temporal_analysis.stability_score:.4f}")

    print(f"\n- Explanation Preview (first 400 chars):")
    print(f"  {result.reason[:400]}...")

    # Test batch processing
    print(f"\n--- Testing Batch Processing ---")
    classifications = {
        "P-1": create_sample_classification("attack", 0.85),
        "P-2": create_sample_classification("failure", 0.78),
        "A-7": create_sample_classification("unknown", 0.55)
    }

    features_dict = {
        "P-1": features,
        "P-2": features,
        "A-7": features
    }

    batch_results = engine.batch_explain(
        classifications=classifications,
        features_dict=features_dict
    )

    print(f"Batch processed {len(batch_results)} classifications")
    for channel_id, result in batch_results.items():
        print(f"  • {channel_id}: {result.pattern_type} (confidence: {result.confidence_explanation:.3f})")

    print(f"\n✓ Explainability engine test successful")

    return {
        "single_explanation": {
            "pattern_type": result.pattern_type,
            "confidence": result.confidence_explanation,
            "processing_time_ms": result.processing_time_ms,
            "n_features": len(result.top_features)
        },
        "batch_results": {
            channel_id: {
                "pattern_type": r.pattern_type,
                "confidence": r.confidence_explanation
            }
            for channel_id, r in batch_results.items()
        }
    }

def run_all_tests():
    """Run all explainability tests"""
    print("\n" + "=" * 80)
    print("EXPLAINABILITY LAYER TEST SUITE")
    print("=" * 80)
    print(f"Test Date: {datetime.now().isoformat()}")
    print(f"Test Environment: Python {sys.version.split()[0]}")

    results = {
        "test_date": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "tests": {}
    }

    # Test 1: Temporal Pattern Detection
    try:
        pattern_results = test_temporal_pattern_detection()
        results["tests"]["temporal_pattern_detection"] = {
            "status": "passed",
            "patterns_tested": list(pattern_results.keys()),
            "results": pattern_results
        }
    except Exception as e:
        print(f"\n❌ Temporal pattern detection test failed: {e}")
        results["tests"]["temporal_pattern_detection"] = {"status": "failed", "error": str(e)}

    # Test 2: Explanation Templates
    try:
        template_results = test_explanation_templates()
        results["tests"]["explanation_templates"] = {
            "status": "passed",
            "templates_tested": list(template_results.keys()),
            "results": template_results
        }
    except Exception as e:
        print(f"\n❌ Explanation templates test failed: {e}")
        results["tests"]["explanation_templates"] = {"status": "failed", "error": str(e)}

    # Test 3: Explainability Engine
    try:
        engine_results = test_explainability_engine()
        results["tests"]["explainability_engine"] = {
            "status": "passed",
            "results": engine_results
        }
    except Exception as e:
        print(f"\n❌ Explainability engine test failed: {e}")
        results["tests"]["explainability_engine"] = {"status": "failed", "error": str(e)}

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
    if "explainability_engine" in results["tests"] and results["tests"]["explainability_engine"]["status"] == "passed":
        engine_result = results["tests"]["explainability_engine"]["results"]["single_explanation"]
        print(f"\nPerformance Summary:")
        print(f"  • Average processing time: {engine_result['processing_time_ms']:.2f} ms")
        print(f"  • Target: <15 ms")
        print(f"  • Status: {'✓ PASS' if engine_result['processing_time_ms'] < 15 else '⚠ SLOW'}")

    # Save results
    output_file = Path(__file__).parent / "explainability_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Test results saved to: {output_file}")

    return results

if __name__ == "__main__":
    results = run_all_tests()