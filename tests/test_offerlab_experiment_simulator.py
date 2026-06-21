from __future__ import annotations

import _bootstrap  # noqa: F401

import unittest

from behavior_lab.offerlab_experiment import ExperimentAssumptions, simulate_self_funded_experiment


class OfferLabExperimentSimulatorTests(unittest.TestCase):
    def test_simulator_compares_required_strategies_and_outputs_incremental_margin(self) -> None:
        report = simulate_self_funded_experiment(
            ExperimentAssumptions(
                listings=100,
                item_cost=100.0,
                asking_price=200.0,
                baseline_sale_probability=0.25,
                baseline_sale_price=185.0,
                fee_rate=0.13,
                shipping_cost=12.0,
                holding_cost_per_listing_day=0.05,
                experiment_setup_cost=250.0,
            )
        )
        strategies = {row["strategy"]: row for row in report["strategies"]}
        self.assertIn("no_experiment_current_policy", strategies)
        self.assertIn("shadow_decision_support", strategies)
        self.assertIn("two_policy_randomized_test", strategies)
        self.assertIn("multi_price_randomized_test", strategies)
        for row in strategies.values():
            self.assertIn("expected_contribution_margin", row)
            self.assertIn("expected_incremental_margin_vs_no_experiment", row)
            self.assertIn("expected_margin_per_sale", row)
        self.assertEqual(strategies["no_experiment_current_policy"]["sale_probability"], 0.25)
        self.assertEqual(strategies["no_experiment_current_policy"]["expected_units_sold"], 25.0)
        self.assertEqual(strategies["no_experiment_current_policy"]["expected_sale_price"], 185.0)
        self.assertEqual(strategies["no_experiment_current_policy"]["expected_margin_per_sale"], 48.95)
        self.assertEqual(strategies["no_experiment_current_policy"]["expected_holding_cost"], 112.5)
        self.assertEqual(strategies["no_experiment_current_policy"]["expected_contribution_margin"], 1111.25)
        self.assertEqual(strategies["shadow_decision_support"]["experiment_setup_cost"], 87.5)
        self.assertEqual(strategies["shadow_decision_support"]["expected_contribution_margin"], 1023.75)
        self.assertEqual(strategies["two_policy_randomized_test"]["sale_probability"], 0.32)
        self.assertEqual(strategies["two_policy_randomized_test"]["expected_sale_price"], 190.0)
        self.assertEqual(strategies["two_policy_randomized_test"]["expected_margin_per_sale"], 53.3)
        self.assertEqual(strategies["two_policy_randomized_test"]["expected_holding_cost"], 102.0)
        self.assertEqual(strategies["two_policy_randomized_test"]["expected_contribution_margin"], 1353.6)
        self.assertEqual(strategies["multi_price_randomized_test"]["sale_probability"], 0.29)
        self.assertEqual(strategies["multi_price_randomized_test"]["expected_sale_price"], 180.0)
        self.assertEqual(strategies["multi_price_randomized_test"]["expected_margin_per_sale"], 44.6)
        self.assertEqual(strategies["multi_price_randomized_test"]["expected_holding_cost"], 106.5)
        self.assertEqual(strategies["multi_price_randomized_test"]["expected_contribution_margin"], 936.9)
        self.assertEqual(strategies["no_experiment_current_policy"]["expected_incremental_margin_vs_no_experiment"], 0.0)
        self.assertEqual(strategies["two_policy_randomized_test"]["expected_incremental_margin_vs_no_experiment"], 242.35)
        self.assertIn("recommended_next_step", report)
        self.assertIn("not evidence of causal lift", " ".join(report["warnings"]))
        self.assertNotIn("net floor", " ".join(report["warnings"]).lower())

    def test_simulator_validates_assumptions(self) -> None:
        with self.assertRaises(ValueError):
            simulate_self_funded_experiment(
                {
                    "listings": 0,
                    "item_cost": 10.0,
                    "asking_price": 20.0,
                    "baseline_sale_probability": 0.2,
                    "baseline_sale_price": 18.0,
                }
            )
        with self.assertRaises(ValueError):
            ExperimentAssumptions(
                listings=10,
                item_cost=10.0,
                asking_price=20.0,
                baseline_sale_probability=1.2,
                baseline_sale_price=18.0,
            ).validate()

    def test_simulator_does_not_recommend_negative_margin_experiment(self) -> None:
        report = simulate_self_funded_experiment(
            ExperimentAssumptions(
                listings=10,
                item_cost=1000.0,
                asking_price=100.0,
                baseline_sale_probability=0.2,
                baseline_sale_price=90.0,
                experiment_setup_cost=100.0,
            )
        )
        self.assertEqual(report["recommended_next_step"]["status"], "do_not_spend_yet")


if __name__ == "__main__":
    unittest.main()
