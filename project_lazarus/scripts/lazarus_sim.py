"""
Lazarus Simulation Script - Time Warped Loan Application Generator.

This script generates 100,000 synthetic loan applications and simulates
the feedback loop to generate training data for causal inference.

The key trick: We inject a HIDDEN VARIABLE (gambling_addiction) that the
model doesn't see, representing real-world hidden risk factors.
"""

import asyncio
import random
import uuid
from datetime import datetime, timedelta
from typing import Any

import httpx
import numpy as np
import structlog
from faker import Faker
from tqdm import tqdm

from config.settings import get_settings

logger = structlog.get_logger(__name__)
fake = Faker()
Faker.seed(42)
np.random.seed(42)


class LoanSimulator:
    """
    Simulates loan applications with hidden risk factors.

    The simulator creates realistic applicant profiles and determines
    default outcomes based on both visible and hidden features.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        num_applications: int = 100_000,
        default_rate_base: float = 0.08,  # 8% base default rate
    ):
        """
        Initialize the simulator.

        Args:
            api_url: URL of the Lazarus API
            num_applications: Number of applications to generate
            default_rate_base: Base default rate before risk adjustments
        """
        self.api_url = api_url
        self.num_applications = num_applications
        self.default_rate_base = default_rate_base

        # Statistics tracking
        self.stats = {
            "total_applications": 0,
            "approved": 0,
            "rejected": 0,
            "explore_approved": 0,
            "exploit_approved": 0,
            "defaults": 0,
            "explore_defaults": 0,
            "exploit_defaults": 0,
            "hidden_risk_defaults": 0,
        }

        # Results storage
        self.results: list[dict[str, Any]] = []

    def generate_applicant(self) -> dict[str, Any]:
        """
        Generate a synthetic loan applicant with visible and hidden features.

        Returns:
            Dictionary with applicant features including hidden variables
        """
        # Generate visible features
        age = random.randint(18, 75)
        income = max(15000, np.random.lognormal(10.8, 0.5))  # ~$50k median
        debt = max(0, income * np.random.uniform(0, 0.8))
        credit_score = int(np.clip(np.random.normal(680, 80), 300, 850))
        employment_years = max(0, np.random.exponential(5))
        recent_bankruptcy = random.random() < 0.03  # 3% bankruptcy rate
        num_credit_lines = int(np.clip(np.random.poisson(3), 0, 15))
        avg_txn_amt_30d = max(0, np.random.lognormal(4.5, 1))
        credit_history_months = int(np.clip(np.random.exponential(60), 0, 360))

        # ===================================================================
        # THE TRICK: Hidden Variables (Model doesn't see these)
        # ===================================================================
        # Gambling addiction - correlates weakly with visible features
        # but strongly predicts default
        gambling_addiction = random.random() < 0.05  # 5% of population

        # Unstable life situation (divorce, illness, job loss pending)
        # Not captured in current employment status
        unstable_situation = random.random() < 0.08

        # Secondary income (gig work, family support) - makes them safer
        # Not captured in reported income
        secondary_income = random.random() < 0.15

        # Generate user ID and application ID
        user_id = f"usr_{uuid.uuid4().hex[:12]}"
        application_id = f"app_{uuid.uuid4().hex[:12]}"

        return {
            # Visible features (sent to API)
            "user_id": user_id,
            "application_id": application_id,
            "age": age,
            "income": round(income, 2),
            "debt": round(debt, 2),
            "credit_score": credit_score,
            "employment_years": round(employment_years, 2),
            "recent_bankruptcy": recent_bankruptcy,
            "num_credit_lines": num_credit_lines,
            "avg_txn_amt_30d": round(avg_txn_amt_30d, 2),
            "credit_history_months": credit_history_months,
            "requested_amount": round(np.random.lognormal(8.5, 0.8), 2),  # ~$5k median

            # Hidden features (for default calculation only)
            "_gambling_addiction": gambling_addiction,
            "_unstable_situation": unstable_situation,
            "_secondary_income": secondary_income,
        }

    def calculate_default_probability(self, applicant: dict[str, Any]) -> float:
        """
        Calculate the true default probability using ALL features.

        This is the "ground truth" that includes hidden variables.

        Args:
            applicant: Applicant data including hidden features

        Returns:
            Probability of default
        """
        # Base default probability
        p_default = self.default_rate_base

        # Visible feature effects
        # Credit score effect (strong)
        credit_effect = (700 - applicant["credit_score"]) / 200
        p_default += credit_effect * 0.15

        # Debt-to-income effect
        if applicant["income"] > 0:
            dti = applicant["debt"] / applicant["income"]
            p_default += dti * 0.1

        # Employment stability effect
        if applicant["employment_years"] < 1:
            p_default += 0.05
        elif applicant["employment_years"] > 5:
            p_default -= 0.02

        # Recent bankruptcy (strong effect)
        if applicant["recent_bankruptcy"]:
            p_default += 0.20

        # ===================================================================
        # HIDDEN VARIABLE EFFECTS (This is what the model misses!)
        # ===================================================================
        # Gambling addiction has MASSIVE impact on default
        if applicant["_gambling_addiction"]:
            p_default += 0.35  # 35% increase in default probability!

        # Unstable situation effect
        if applicant["_unstable_situation"]:
            p_default += 0.15

        # Secondary income reduces default (protective factor)
        if applicant["_secondary_income"]:
            p_default -= 0.05

        # Clip to valid probability range
        return np.clip(p_default, 0.01, 0.95)

    def simulate_default(self, applicant: dict[str, Any]) -> tuple[bool, int | None]:
        """
        Simulate whether an approved loan defaults.

        Args:
            applicant: Applicant data

        Returns:
            Tuple of (defaulted, days_to_default)
        """
        p_default = self.calculate_default_probability(applicant)

        if random.random() < p_default:
            # Default occurred
            # Days to default follows exponential distribution
            days = int(np.random.exponential(90))  # ~90 days average
            days = min(days, 365)  # Cap at 1 year
            return True, days

        return False, None

    async def send_application(
        self,
        client: httpx.AsyncClient,
        applicant: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Send a loan application to the API.

        Args:
            client: HTTP client
            applicant: Applicant data

        Returns:
            API response or None if failed
        """
        # Prepare API payload (exclude hidden features)
        payload = {
            "application_id": applicant["application_id"],
            "user_features": {
                "user_id": applicant["user_id"],
                "age": applicant["age"],
                "income": applicant["income"],
                "debt": applicant["debt"],
                "credit_score": applicant["credit_score"],
                "employment_years": applicant["employment_years"],
                "recent_bankruptcy": applicant["recent_bankruptcy"],
                "num_credit_lines": applicant["num_credit_lines"],
                "avg_txn_amt_30d": applicant["avg_txn_amt_30d"],
                "credit_history_months": applicant["credit_history_months"],
            },
            "requested_amount": applicant["requested_amount"],
            "loan_purpose": "personal",
        }

        try:
            response = await client.post(
                f"{self.api_url}/decide",
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("api_request_failed", error=str(e))
            return None

    async def run_simulation(self, batch_size: int = 100) -> dict[str, Any]:
        """
        Run the full simulation.

        Args:
            batch_size: Number of concurrent requests

        Returns:
            Simulation statistics
        """
        logger.info(
            "simulation_started",
            num_applications=self.num_applications,
            batch_size=batch_size
        )

        async with httpx.AsyncClient() as client:
            # Generate all applicants first
            applicants = [
                self.generate_applicant()
                for _ in range(self.num_applications)
            ]

            # Process in batches
            for i in tqdm(range(0, len(applicants), batch_size), desc="Processing"):
                batch = applicants[i:i + batch_size]

                # Send batch concurrently
                tasks = [
                    self.send_application(client, app)
                    for app in batch
                ]
                responses = await asyncio.gather(*tasks)

                # Process results
                for applicant, response in zip(batch, responses):
                    if response is None:
                        continue

                    self.stats["total_applications"] += 1

                    result = {
                        "applicant": applicant,
                        "response": response,
                        "defaulted": None,
                        "days_to_default": None,
                    }

                    if response["decision"] == "APPROVE":
                        self.stats["approved"] += 1

                        # Track by treatment group
                        if response["treatment_group"] == "explore":
                            self.stats["explore_approved"] += 1
                        else:
                            self.stats["exploit_approved"] += 1

                        # Simulate default outcome
                        defaulted, days = self.simulate_default(applicant)
                        result["defaulted"] = defaulted
                        result["days_to_default"] = days

                        if defaulted:
                            self.stats["defaults"] += 1

                            if response["treatment_group"] == "explore":
                                self.stats["explore_defaults"] += 1
                            else:
                                self.stats["exploit_defaults"] += 1

                            # Track hidden risk factor defaults
                            if applicant["_gambling_addiction"]:
                                self.stats["hidden_risk_defaults"] += 1
                    else:
                        self.stats["rejected"] += 1

                    self.results.append(result)

        # Calculate derived statistics
        self.stats["approval_rate"] = (
            self.stats["approved"] / max(1, self.stats["total_applications"])
        )
        self.stats["default_rate"] = (
            self.stats["defaults"] / max(1, self.stats["approved"])
        )
        self.stats["explore_default_rate"] = (
            self.stats["explore_defaults"] / max(1, self.stats["explore_approved"])
        )
        self.stats["exploit_default_rate"] = (
            self.stats["exploit_defaults"] / max(1, self.stats["exploit_approved"])
        )

        logger.info("simulation_completed", stats=self.stats)

        return self.stats

    def save_results(self, filepath: str = "data/simulation_results.json") -> None:
        """Save simulation results to file."""
        import json
        from pathlib import Path

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Prepare serializable results
        output = {
            "stats": self.stats,
            "results": [
                {
                    "application_id": r["applicant"]["application_id"],
                    "user_id": r["applicant"]["user_id"],
                    "decision": r["response"]["decision"] if r["response"] else None,
                    "treatment_group": r["response"]["treatment_group"] if r["response"] else None,
                    "risk_score": r["response"]["risk_score"] if r["response"] else None,
                    "defaulted": r["defaulted"],
                    "days_to_default": r["days_to_default"],
                    "hidden_gambling": r["applicant"]["_gambling_addiction"],
                    "hidden_unstable": r["applicant"]["_unstable_situation"],
                    "hidden_secondary_income": r["applicant"]["_secondary_income"],
                }
                for r in self.results
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info("results_saved", filepath=filepath)

    def generate_training_data(self) -> tuple[list[dict], list[dict]]:
        """
        Generate training data from simulation results.

        Returns explore and exploit datasets separately for causal analysis.

        Returns:
            Tuple of (explore_data, exploit_data)
        """
        explore_data = []
        exploit_data = []

        for result in self.results:
            if result["response"] is None:
                continue

            # Only approved loans have outcomes
            if result["response"]["decision"] != "APPROVE":
                continue

            record = {
                "application_id": result["applicant"]["application_id"],
                "user_id": result["applicant"]["user_id"],
                "age": result["applicant"]["age"],
                "income": result["applicant"]["income"],
                "debt": result["applicant"]["debt"],
                "credit_score": result["applicant"]["credit_score"],
                "employment_years": result["applicant"]["employment_years"],
                "recent_bankruptcy": result["applicant"]["recent_bankruptcy"],
                "num_credit_lines": result["applicant"]["num_credit_lines"],
                "avg_txn_amt_30d": result["applicant"]["avg_txn_amt_30d"],
                "credit_history_months": result["applicant"]["credit_history_months"],
                "risk_score": result["response"]["risk_score"],
                "defaulted": result["defaulted"],
                "treatment_group": result["response"]["treatment_group"],
            }

            if result["response"]["treatment_group"] == "explore":
                explore_data.append(record)
            else:
                exploit_data.append(record)

        return explore_data, exploit_data


async def main():
    """Main entry point for simulation."""
    settings = get_settings()

    simulator = LoanSimulator(
        api_url=f"http://localhost:{settings.api_port}",
        num_applications=100_000,
    )

    # Run simulation
    stats = await simulator.run_simulation(batch_size=50)

    # Save results
    simulator.save_results()

    # Print summary
    print("\n" + "=" * 60)
    print("PROJECT LAZARUS SIMULATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal Applications: {stats['total_applications']:,}")
    print(f"Approved: {stats['approved']:,} ({stats['approval_rate']:.1%})")
    print(f"Rejected: {stats['rejected']:,}")
    print(f"\nExplore Approved: {stats['explore_approved']:,}")
    print(f"Exploit Approved: {stats['exploit_approved']:,}")
    print(f"\nDefaults: {stats['defaults']:,} ({stats['default_rate']:.1%})")
    print(f"  - Explore Defaults: {stats['explore_defaults']:,} ({stats['explore_default_rate']:.1%})")
    print(f"  - Exploit Defaults: {stats['exploit_defaults']:,} ({stats['exploit_default_rate']:.1%})")
    print(f"  - Hidden Risk Factor Defaults: {stats['hidden_risk_defaults']:,}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
