import unittest
import math

from options_pricing.src.black_scholes import black_scholes_price
from options_pricing.src.numerical_methods import (
    american_option_binomial,
    implied_volatility,
    monte_carlo_price,
)


class TestOptionsPricing(unittest.TestCase):
    def test_put_call_parity(self):
        call = black_scholes_price(100, 100, 0.03, 0.2, 1.0, option_type="call")
        put = black_scholes_price(100, 100, 0.03, 0.2, 1.0, option_type="put")
        parity_gap = call - put - (100 - 100 * math.exp(-0.03))
        self.assertAlmostEqual(parity_gap, 0.0, places=4)

    def test_implied_volatility_recovers_sigma(self):
        sigma = 0.27
        price = black_scholes_price(105, 100, 0.04, sigma, 0.75, option_type="call")
        recovered = implied_volatility(
            price,
            105,
            100,
            0.04,
            0.75,
            option_type="call",
        )
        self.assertAlmostEqual(recovered, sigma, places=4)

    def test_monte_carlo_close_to_black_scholes(self):
        analytic = black_scholes_price(100, 100, 0.02, 0.25, 1.0, option_type="call")
        mc = monte_carlo_price(
            100,
            100,
            0.02,
            0.25,
            1.0,
            option_type="call",
            n_paths=40_000,
            seed=123,
        )
        self.assertAlmostEqual(mc.price, analytic, delta=0.35)

    def test_american_put_at_least_european_put(self):
        european_put = black_scholes_price(95, 100, 0.01, 0.2, 1.0, option_type="put")
        american_put = american_option_binomial(
            95,
            100,
            0.01,
            0.2,
            1.0,
            option_type="put",
            n_steps=250,
        )
        self.assertGreaterEqual(american_put, european_put)


if __name__ == "__main__":
    unittest.main()
