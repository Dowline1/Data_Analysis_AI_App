"""
Evaluation Framework for Bank Statement Analysis

This module provides evaluation functions to test the quality and
accuracy of categorization, subscription detection, and analysis.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
from collections import defaultdict


class CategorizationEvaluator:
    """
    Evaluates transaction categorization accuracy.
    
    Metrics:
    - Accuracy: % correctly categorized
    - Precision/Recall per category
    - Confusion matrix
    """
    
    def evaluate(
        self,
        predicted: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate categorization against ground truth.
        
        Args:
            predicted: List of transactions with predicted categories
            ground_truth: List of transactions with correct categories
            
        Returns:
            Evaluation metrics
        """
        if len(predicted) != len(ground_truth):
            return {
                "error": "Predicted and ground truth must have same length"
            }
        
        correct = 0
        total = len(predicted)
        
        # Track per-category stats
        category_stats = defaultdict(lambda: {
            "tp": 0,  # true positives
            "fp": 0,  # false positives
            "fn": 0   # false negatives
        })
        
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        for pred, truth in zip(predicted, ground_truth):
            pred_cat = pred.get("category", "Unknown")
            true_cat = truth.get("category", "Unknown")
            
            if pred_cat == true_cat:
                correct += 1
                category_stats[true_cat]["tp"] += 1
            else:
                category_stats[true_cat]["fn"] += 1
                category_stats[pred_cat]["fp"] += 1
            
            confusion_matrix[true_cat][pred_cat] += 1
        
        accuracy = correct / total if total > 0 else 0
        
        # Calculate per-category precision and recall
        category_metrics = {}
        for cat, stats in category_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            category_metrics[cat] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": tp + fn
            }
        
        return {
            "overall_accuracy": accuracy,
            "correct_count": correct,
            "total_count": total,
            "category_metrics": category_metrics,
            "confusion_matrix": dict(confusion_matrix)
        }


class SubscriptionDetectionEvaluator:
    """
    Evaluates subscription detection quality.
    
    Metrics:
    - Detection rate: % of true subscriptions found
    - False positive rate: % of non-subscriptions flagged
    - Frequency accuracy: % with correct recurrence pattern
    """
    
    def evaluate(
        self,
        detected: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate subscription detection against ground truth.
        
        Args:
            detected: List of detected subscriptions
            ground_truth: List of true subscriptions
            
        Returns:
            Evaluation metrics
        """
        # Match detected to ground truth by merchant name
        detected_merchants = {sub["merchant"].lower() for sub in detected}
        true_merchants = {sub["merchant"].lower() for sub in ground_truth}
        
        # True positives: correctly detected subscriptions
        true_positives = detected_merchants & true_merchants
        
        # False positives: incorrectly flagged as subscription
        false_positives = detected_merchants - true_merchants
        
        # False negatives: missed subscriptions
        false_negatives = true_merchants - detected_merchants
        
        precision = len(true_positives) / len(detected_merchants) if detected_merchants else 0
        recall = len(true_positives) / len(true_merchants) if true_merchants else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Check frequency accuracy for true positives
        frequency_correct = 0
        for det in detected:
            if det["merchant"].lower() in true_positives:
                # Find corresponding ground truth
                true_sub = next(
                    (s for s in ground_truth if s["merchant"].lower() == det["merchant"].lower()),
                    None
                )
                if true_sub and det.get("frequency") == true_sub.get("frequency"):
                    frequency_correct += 1
        
        frequency_accuracy = frequency_correct / len(true_positives) if true_positives else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "frequency_accuracy": frequency_accuracy,
            "detected_count": len(detected_merchants),
            "ground_truth_count": len(true_merchants)
        }


class HealthScoreEvaluator:
    """
    Evaluates health score calculation consistency.
    
    Metrics:
    - Score range validity (0-100)
    - Component score consistency
    - Assessment alignment with score
    """
    
    def evaluate(self, health_score: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate health score calculation.
        
        Args:
            health_score: Calculated health score dict
            
        Returns:
            Validation results
        """
        issues = []
        
        # Check overall score range
        overall = health_score.get("overall_score", 0)
        if not 0 <= overall <= 100:
            issues.append(f"Overall score out of range: {overall}")
        
        # Check component scores
        components = {
            "checking_score": health_score.get("checking_score", 0),
            "credit_card_score": health_score.get("credit_card_score", 0),
            "income_ratio_score": health_score.get("income_ratio_score", 0),
            "savings_rate_score": health_score.get("savings_rate_score", 0),
            "payment_ratio_score": health_score.get("payment_ratio_score", 0),
            "balance_health_score": health_score.get("balance_health_score", 0)
        }
        
        for name, score in components.items():
            if not 0 <= score <= 100:
                issues.append(f"{name} out of range: {score}")
        
        # Check assessment alignment
        assessment = health_score.get("assessment", "")
        expected_assessment = None
        
        if overall >= 80:
            expected_assessment = "EXCELLENT"
        elif overall >= 60:
            expected_assessment = "GOOD"
        elif overall >= 40:
            expected_assessment = "FAIR"
        else:
            expected_assessment = "NEEDS_ATTENTION"
        
        if assessment not in expected_assessment:
            issues.append(
                f"Assessment '{assessment}' doesn't align with score {overall}"
            )
        
        return {
            "valid": len(issues) == 0,
            "overall_score": overall,
            "issues": issues,
            "component_scores": components
        }


class EndToEndEvaluator:
    """
    End-to-end evaluation of the complete analysis pipeline.
    
    Runs a test dataset through the workflow and validates outputs.
    """
    
    def __init__(self):
        self.categorization_eval = CategorizationEvaluator()
        self.subscription_eval = SubscriptionDetectionEvaluator()
        self.health_eval = HealthScoreEvaluator()
    
    def evaluate_pipeline(
        self,
        test_case: Dict[str, Any],
        actual_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate complete pipeline against test case.
        
        Args:
            test_case: Dict with ground truth data
                - transactions: expected categorized transactions
                - subscriptions: expected subscriptions
                - expected_health_score: expected health score
            actual_results: Dict with actual pipeline outputs
                - transactions: actual categorized transactions
                - subscriptions: actual detected subscriptions
                - health_score: actual health score
                
        Returns:
            Comprehensive evaluation report
        """
        results = {}
        
        # Evaluate categorization
        if "transactions" in test_case and "transactions" in actual_results:
            results["categorization"] = self.categorization_eval.evaluate(
                predicted=actual_results["transactions"],
                ground_truth=test_case["transactions"]
            )
        
        # Evaluate subscription detection
        if "subscriptions" in test_case and "subscriptions" in actual_results:
            results["subscriptions"] = self.subscription_eval.evaluate(
                detected=actual_results["subscriptions"],
                ground_truth=test_case["subscriptions"]
            )
        
        # Evaluate health score
        if "health_score" in actual_results:
            results["health_score"] = self.health_eval.evaluate(
                health_score=actual_results["health_score"]
            )
        
        # Calculate overall pass/fail
        categorization_pass = results.get("categorization", {}).get("overall_accuracy", 0) >= 0.8
        subscription_pass = results.get("subscriptions", {}).get("f1_score", 0) >= 0.7
        health_pass = results.get("health_score", {}).get("valid", False)
        
        results["overall_pass"] = categorization_pass and subscription_pass and health_pass
        results["summary"] = {
            "categorization_pass": categorization_pass,
            "subscription_pass": subscription_pass,
            "health_score_pass": health_pass
        }
        
        return results


def create_test_dataset() -> Dict[str, Any]:
    """
    Create a sample test dataset with ground truth labels.
    
    Returns:
        Test dataset with labeled transactions and subscriptions
    """
    return {
        "transactions": [
            {
                "date": "2024-01-01",
                "description": "Netflix Subscription",
                "amount": -15.99,
                "category": "Entertainment",
                "account_type": "checking"
            },
            {
                "date": "2024-01-05",
                "description": "Whole Foods Market",
                "amount": -87.45,
                "category": "Groceries",
                "account_type": "credit_card"
            },
            {
                "date": "2024-01-15",
                "description": "Salary Deposit",
                "amount": 3500.00,
                "category": "Income",
                "account_type": "checking"
            },
            {
                "date": "2024-02-01",
                "description": "Netflix Subscription",
                "amount": -15.99,
                "category": "Entertainment",
                "account_type": "checking"
            }
        ],
        "subscriptions": [
            {
                "merchant": "Netflix",
                "frequency": "monthly",
                "amount": 15.99,
                "category": "Entertainment"
            }
        ],
        "expected_health_score": {
            "overall_score": 75,
            "assessment": "GOOD"
        }
    }


# Export evaluators
__all__ = [
    "CategorizationEvaluator",
    "SubscriptionDetectionEvaluator",
    "HealthScoreEvaluator",
    "EndToEndEvaluator",
    "create_test_dataset"
]
