from typing import Tuple, Dict, Any
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame


# =====================================================
# Utility Functions
# =====================================================

def stochastic_time_series(start_value, end_value, num_steps=100, p=0.1, x=5, final_noise=1.0, volatility=1.0,
                           seed=None):
    """
    Generates a stochastic time series with spikes and final value noise.
    """
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, 1, num_steps)
    noise = np.random.randn(num_steps) * volatility
    bridge = start_value * (1 - t) + end_value * t + np.cumsum(noise - np.mean(noise))
    spikes = (np.random.rand(num_steps) < p) * np.random.randn(num_steps) * x
    series = bridge + spikes
    series[-1] += np.random.randn() * final_noise
    return series


def refined_dynamic_spread(expected_returns, base_spread=0.015, sensitivity=1):
    """
    Computes a dynamically adjusted spread using an improved exponential function.
    """
    spread = expected_returns * (1 - np.exp(-sensitivity * expected_returns))
    spread = np.minimum(spread, expected_returns)
    spread = np.clip(spread, base_spread, 0.4)
    return spread


# =====================================================
# Interest Rate Models
# =====================================================

class ArkisPendleIRC:
    def __init__(self, performance_fee=0.1, min_reference_rate=0.1):
        self.performance_fee = performance_fee
        self.min_reference_rate = min_reference_rate

    def get_initial_rates(self, expected_return, spread, utilization=1.0):
        supply_rate = expected_return - spread
        borrow_rate = supply_rate * (1 + self.performance_fee)
        return supply_rate, borrow_rate

    def update_rates(self, expected_return, spread, utilization):
        reference_rate = (expected_return - spread) * utilization
        reference_rate = max(self.min_reference_rate, reference_rate)
        fee = reference_rate * self.performance_fee
        new_borrow_rate = reference_rate + fee * utilization
        new_supply_rate = reference_rate - fee * (1 - utilization)
        return new_supply_rate, new_borrow_rate


class MorphoAdaptiveCurveIRM:
    def __init__(self, performance_fee=0.1, u_target=0.9, k_d=4, adjustment_speed=0.138629,
                 initial_rate_at_target=0.0001):
        self.performance_fee = performance_fee
        self.u_target = u_target
        self.k_d = k_d
        self.adjustment_speed = adjustment_speed
        self.initial_rate_at_target = initial_rate_at_target
        self.r_T = initial_rate_at_target  # current rate at target

    def compute_error(self, utilization):
        if utilization <= self.u_target:
            return (utilization - self.u_target) / self.u_target
        else:
            return (utilization - self.u_target) / (1 - self.u_target)

    def compute_curve(self, utilization):
        error = self.compute_error(utilization)
        if utilization <= self.u_target:
            return (1 - 1 / self.k_d) * error + 1
        else:
            return (self.k_d - 1) * error + 1

    def get_initial_rates(self, expected_return, spread, utilization):
        # expected_return and spread are ignored in this model.
        self.r_T = self.initial_rate_at_target
        curve_val = self.compute_curve(utilization)
        borrow_rate = self.r_T * curve_val
        supply_rate = borrow_rate * utilization * (1 - self.performance_fee)
        return supply_rate * 365, borrow_rate * 365

    def update_rates(self, expected_return, spread, utilization):
        error = self.compute_error(utilization)
        # Daily time step update
        self.r_T = self.r_T * np.exp(self.adjustment_speed * error)
        curve_val = self.compute_curve(utilization)
        borrow_rate = self.r_T * curve_val
        supply_rate = borrow_rate * utilization * (1 - self.performance_fee)
        return supply_rate * 365, borrow_rate * 365


# =====================================================
# Agent Classes
# =====================================================

class Trader:
    def __init__(self, id: int, entry_apy: float, entry_amount: float, max_leverage: int = 3,
                 rs_state: np.random.RandomState = None, patience: float = 0.5):
        """
        Represents a borrower.
        Parameters:
          - patience (float): Between 0 and 1; higher values make the trader more reluctant to exit.
        """
        self.id = id
        self.entry_apy = entry_apy
        self.entry_amount = entry_amount
        self.max_leverage = max_leverage
        self.rs_state = rs_state
        self.patience = patience
        self.pnl = 0.0
        self.time_in_position = 0

    def check_exit(self, expected_apy: float, borrow_apy: float):
        """
        Decide whether to exit based on a logistic function of
        d = (borrow APY - expected APY).
        When d < 0 (i.e. expected APY > borrow APY), exit probability is low.
        """
        d = borrow_apy - expected_apy
        steepness = 50
        threshold = 0.0
        prob_exit = 1 / (1 + np.exp(-steepness * (d - threshold)))
        prob_exit *= (1 - self.patience)
        flag = self.rs_state.random() < prob_exit
        if bool(flag) is False:
            self.pnl += (expected_apy - borrow_apy) / 365
            self.time_in_position += 1
        return flag


class Lender:
    def __init__(self, id: int, initial_deposit: float, deposit_threshold: float = 0.2,
                 withdrawal_threshold: float = 0.1, rs_state: np.random.RandomState = None):
        """
        Represents a lender with an individual deposit.
        Parameters:
          - deposit_threshold: If supply rate is above this, the lender is more likely to deposit.
          - withdrawal_threshold: If supply rate is below this, the lender is more likely to withdraw.
        """
        self.id = id
        self.deposit = initial_deposit
        self.deposit_threshold = deposit_threshold
        self.withdrawal_threshold = withdrawal_threshold
        self.rs_state = rs_state if rs_state is not None else np.random.RandomState()

    def decide(self, supply_rate: float):
        """
        Decide the change in deposit based on the current supply rate:
          - If supply_rate >= deposit_threshold: deposit additional funds.
          - If supply_rate <= withdrawal_threshold: withdraw funds.
          - Otherwise, no change.
        Returns the net change in deposit.
        """
        if supply_rate >= self.deposit_threshold:
            additional = abs(self.rs_state.normal(loc=10, scale=2))
            self.deposit += additional
            return additional
        elif supply_rate <= self.withdrawal_threshold:
            withdrawal = abs(self.rs_state.normal(loc=8, scale=2))
            actual = min(withdrawal, self.deposit)
            self.deposit -= actual
            return -actual
        else:
            return 0.0


# =====================================================
# Simulation Class
# =====================================================

class DeFiLendingSimulation:
    def __init__(self, expected_returns, spread_function, interest_rate_model, seed: int = 42,
                 max_leverage: int = 3, num_lenders: int = 100):
        """
        Parameters:
          - expected_returns (np.ndarray): Expected return series.
          - spread_function (function): Function to compute the spread.
          - interest_rate_model: An IRM object.
          - num_lenders (int): Number of lender agents.
        """
        self.historical_traders = {}
        self.expected_returns = expected_returns
        self.spread_function = spread_function
        self.interest_rate_model = interest_rate_model
        self.current_step = 0
        self.total_borrowed = 0
        self.total_deposited = 0
        self.borrow_rate = 0
        self.supply_rate = 0
        self.past_borrows_to_subtract = 0
        self.logs = []
        self.rs_state = np.random.RandomState(seed)
        self.utilization = None
        self.max_leverage = max_leverage
        self.traders = {}  # Borrowers
        self.lenders = {}  # Lender agents
        self.num_lenders = num_lenders
        self.next_lender_id = 0  # to assign unique IDs to new lenders

    def init_state(self):
        # Initialize lenders.
        total = 0.0
        for i in range(self.num_lenders):
            deposit = max(0, self.rs_state.normal(loc=20, scale=5))
            lender = Lender(id=i, initial_deposit=deposit, rs_state=np.random.RandomState(i))
            self.lenders[i] = lender
            total += deposit
            self.next_lender_id = i + 1
        self.total_deposited = total

        # Initialize borrowers.
        spread = self.spread_function(self.expected_returns[0])
        self.supply_rate, self.borrow_rate = self.interest_rate_model.get_initial_rates(
            self.expected_returns[0], spread, utilization=1.0)
        self.total_borrowed += 20  # fixed initial borrow amount
        self.utilization = self.total_borrowed / self.total_deposited

        self.logs.append({
            "time": 0,
            "deposited": self.total_deposited,
            "borrowed": self.total_borrowed,
            "total_deposited": self.total_deposited,
            "total_borrowed": self.total_borrowed,
            "expected_return": self.expected_returns[0],
            "borrow_rate": self.borrow_rate,
            "supply_rate": self.supply_rate,
            "utilization": self.utilization,
        })

        # Create an initial borrower.
        tr = Trader(id=0, entry_apy=self.expected_returns[0], entry_amount=20, rs_state=np.random.RandomState(0))
        self.traders[0] = tr

    def simulate_lender_activity(self):
        """
        Iterates over all lenders and computes the net deposit change.
        Also:
          - With 50% probability, a new lender joins.
          - With 20% probability, an existing lender withdraws a portion of their funds.
        Withdrawals are capped by available liquidity: (total_deposited - total_borrowed).
        """
        total_change = 0.0

        # New lender joining event (50% probability).
        if self.rs_state.rand() < 0.5:
            new_deposit = max(0, self.rs_state.normal(loc=20, scale=5))
            new_lender = Lender(id=self.next_lender_id, initial_deposit=new_deposit,
                                rs_state=np.random.RandomState(self.next_lender_id))
            self.lenders[self.next_lender_id] = new_lender
            self.next_lender_id += 1
            total_change += new_deposit

        # Existing lender random withdrawal event (20% probability).
        if self.rs_state.rand() < 0.2 and len(self.lenders) > 0:
            chosen_id = self.rs_state.choice(list(self.lenders.keys()))
            chosen_lender = self.lenders[chosen_id]
            # Withdraw between 10% and 30% of the lender's funds.
            portion = self.rs_state.uniform(0.1, 0.3)
            withdrawal_amount = portion * chosen_lender.deposit
            # Check available liquidity.
            available_for_withdrawal = self.total_deposited - self.total_borrowed
            if withdrawal_amount > available_for_withdrawal:
                withdrawal_amount = available_for_withdrawal
            chosen_lender.deposit -= withdrawal_amount
            total_change -= withdrawal_amount

        # Process all existing lenders' normal behavior.
        available_for_withdrawal = self.total_deposited - self.total_borrowed
        lender_ids = list(self.lenders.keys())
        self.rs_state.shuffle(lender_ids)
        for lid in lender_ids:
            lender = self.lenders[lid]
            change = lender.decide(self.supply_rate)
            if change < 0:
                withdraw_amt = abs(change)
                if withdraw_amt > available_for_withdrawal:
                    withdraw_amt = available_for_withdrawal
                    change = -withdraw_amt
                    available_for_withdrawal = 0
                else:
                    available_for_withdrawal -= withdraw_amt
            total_change += change
        return total_change

    def probability_of_borrow_more(self, expected_return) -> int:
        a = 90.2
        b = 0.05
        probability = 1 / (1 + np.exp(-a * (expected_return - b)))
        return 1 if self.rs_state.random() < probability else 0

    def process_next(self):
        self.current_step += 1
        if self.current_step >= len(self.expected_returns):
            print("End of data series reached.")
            return

        transaction_value = 0.0

        # Process borrower exits.
        items_to_delete = []
        for trader_id, trader in self.traders.items():
            if trader.check_exit(self.expected_returns[self.current_step], self.borrow_rate):
                transaction_value -= trader.entry_amount
                self.past_borrows_to_subtract += trader.entry_amount * trader.entry_apy
                items_to_delete.append(trader_id)
        for item in items_to_delete:
            self.historical_traders[item] = self.traders[item]
            del self.traders[item]

        # Update lenders: simulate deposits/withdrawals.
        deposit_change = self.simulate_lender_activity()
        self.total_deposited += deposit_change

        # Process borrower borrow actions.
        current_spread = self.expected_returns[self.current_step] - self.borrow_rate
        if self.total_borrowed < self.total_deposited:
            max_borrow_amount = max(deposit_change, self.total_deposited - self.total_borrowed)
            borrowed = 0.0
            for i in range(100):
                if self.probability_of_borrow_more(current_spread) == 1 and max_borrow_amount > borrowed:
                    trader_amount = self.rs_state.normal(0.01, 0.005) * max_borrow_amount
                    trader_amount = max(0, trader_amount)
                    transaction_value += trader_amount
                    borrowed += trader_amount
                    tr = Trader(id=self.current_step * (i + 1), entry_apy=self.expected_returns[self.current_step],
                                entry_amount=trader_amount, rs_state=np.random.RandomState(self.current_step * (i + 1)))
                    self.traders[self.current_step * (i + 1)] = tr
            self.total_borrowed += transaction_value
        else:
            self.total_borrowed += transaction_value

        self.utilization = self.total_borrowed / self.total_deposited if self.total_deposited > 0 else 0
        self.logs.append({
            "time": self.current_step,
            "deposited": deposit_change,
            "borrowed": transaction_value,
            "total_deposited": self.total_deposited,
            "total_borrowed": self.total_borrowed,
            "expected_return": self.expected_returns[self.current_step],
            "borrow_rate": self.borrow_rate,
            "supply_rate": self.supply_rate,
            "utilization": self.utilization,
        })

        spread = self.spread_function(self.expected_returns[self.current_step])
        self.supply_rate, self.borrow_rate = self.interest_rate_model.update_rates(
            self.expected_returns[self.current_step], spread, self.utilization
        )

    def get_logs(self):
        return pd.DataFrame(self.logs)


# =====================================================
# Simulation Orchestration & Analytics
# =====================================================

def run_single_simulation(sim_params: dict) -> tuple[DataFrame, dict[Any, Any]]:
    expected_ret_series = stochastic_time_series(**sim_params["series_params"])
    interest_rate_model = sim_params.get("interest_rate_model",
                                         ArkisPendleIRC(performance_fee=0.1, min_reference_rate=0.1))
    simulation = DeFiLendingSimulation(
        expected_ret_series,
        lambda x: refined_dynamic_spread(x,
                                         base_spread=sim_params["spread_params"]["base_spread"],
                                         sensitivity=sim_params["spread_params"]["sensitivity"]),
        interest_rate_model=interest_rate_model,
        seed=sim_params["seed"],
        max_leverage=sim_params.get("max_leverage", 3),
        num_lenders=sim_params.get("num_lenders", 100)
    )
    simulation.init_state()
    num_steps = len(expected_ret_series)
    for _ in range(num_steps - 1):
        simulation.process_next()
    logs = simulation.get_logs()
    logs['supply_apy'] = logs['supply_rate']  # Already annualized in IRM.
    return logs, simulation.historical_traders


def analyze_simulation(logs: pd.DataFrame, traders):
    median_supply_apy = logs['supply_apy'].quantile(0.5)
    print("Median Supply APY:", median_supply_apy)
    borrowers_performance = {}
    for id, trader in traders.items():
        borrowers_performance[id] = trader.pnl * 365 / trader.time_in_position if trader.time_in_position != 0 else 0
    borrowers_performance = pd.Series(borrowers_performance)
    plt.title("Borrowers Performance")
    sns.kdeplot(borrowers_performance, fill=True)
    plt.show()
    print(borrowers_performance.describe())


def run_simulations(n: int, sim_params: dict):
    simulation_results = []
    plt.title("Borrowers Performance")
    for i in range(n):
        sim_params["seed"] = i
        sim_params["series_params"]["seed"] = i + 100
        logs, traders = run_single_simulation(sim_params)
        median_supply_apy = logs['supply_apy'].quantile(0.5)
        total_supply_return = (logs['supply_rate'] / 365).sum() * 365 / logs.shape[0]
        median_utilization = logs['utilization'].quantile(0.5)
        t_value_utilization = logs['utilization'].mean() / logs['utilization'].std()
        borrowers_performance = {}
        for id, trader in traders.items():
            borrowers_performance[id] = trader.pnl * 365/trader.time_in_position if trader.time_in_position != 0 else 0
        borrowers_performance = pd.Series(borrowers_performance)
        sns.kdeplot(borrowers_performance, fill=True)
        simulation_results.append(
            (median_supply_apy, borrowers_performance.copy(), median_utilization, t_value_utilization,
             total_supply_return))
    plt.show()
    overall_median_supply_apy = np.median([res[0] for res in simulation_results])
    overall_median_utilization = np.mean([res[2] for res in simulation_results])
    overall_avg_t_value_utilization = np.mean([res[3] for res in simulation_results])
    combined_borrower_performance = pd.concat([res[1] for res in simulation_results])
    print("Median Supply APY:", overall_median_supply_apy)
    print("10th Percentile Borrower Performance:", combined_borrower_performance.quantile(0.1))
    print("50th Percentile Borrower Performance:", combined_borrower_performance.quantile(0.5))
    print("Average  Borrower Performance:", combined_borrower_performance.mean())
    print("Median Utilization:", overall_median_utilization)
    print("Average T-Value Utilization:", overall_avg_t_value_utilization)
    print("Median supply return:", np.median([res[4] for res in simulation_results]))


# =====================================================
# Main Section
# =====================================================

def main():
    series_params = {
        "start_value": 0.3,
        "end_value": 0.16,
        "num_steps": 200,
        "p": 0.02,
        "x": 0.1,
        "final_noise": 0.01,
        "volatility": 0.01,
        "seed": 32
    }
    expected_ret_series = stochastic_time_series(**series_params)
    plt.figure(figsize=(10, 5))
    plt.plot(expected_ret_series, label="Stochastic Time Series")
    plt.axhline(series_params["start_value"], linestyle="--", color="green", label="Start Value")
    plt.axhline(series_params["end_value"], linestyle="--", color="red", label="End Value")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title("Expected Return Series")
    plt.legend()
    plt.show()

    # Create an interest rate model instance with desired parameters.
    morpho_IRM = MorphoAdaptiveCurveIRM(
        performance_fee=0.1,
        u_target=0.9,
        k_d=4,
        adjustment_speed=0.138629,  # Rate doubles in ~5 days at maximum error.
        initial_rate_at_target=0.04 / 365  # 4% per year as a daily rate.
    )

    arkis_IRM = ArkisPendleIRC(performance_fee=0.1, min_reference_rate=0.1)

    sim_params = {
        "series_params": series_params,
        "spread_params": {"base_spread": 0.03, "sensitivity": 1},
        "seed": 42,
        "max_leverage": 3,
        "num_lenders": 100,
        "interest_rate_model": arkis_IRM
    }

    logs, traders = run_single_simulation(sim_params)
    analyze_simulation(logs, traders)
    run_simulations(n=10, sim_params=sim_params)


if __name__ == '__main__':
    main()
