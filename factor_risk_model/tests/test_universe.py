import unittest
from unittest.mock import patch

from factor_risk_model.src.universe import load_sp500_constituents, load_sp500_symbols


SAMPLE_HTML = """
<table>
  <thead>
    <tr>
      <th>Symbol</th>
      <th>Security</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BRK.B</td>
      <td>Berkshire Hathaway</td>
      <td>Financials</td>
      <td>Multi-Sector Holdings</td>
    </tr>
    <tr>
      <td>MSFT</td>
      <td>Microsoft</td>
      <td>Information Technology</td>
      <td>Systems Software</td>
    </tr>
  </tbody>
</table>
"""


class _MockResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class TestUniverseLoaders(unittest.TestCase):
    @patch(
        "factor_risk_model.src.universe.requests.get",
        return_value=_MockResponse(SAMPLE_HTML),
    )
    def test_load_sp500_constituents(self, _mock_get):
        constituents = load_sp500_constituents()
        self.assertEqual(list(constituents["symbol"]), ["BRK-B", "MSFT"])
        self.assertIn("sector", constituents.columns)
        self.assertIn("sub_industry", constituents.columns)

    @patch(
        "factor_risk_model.src.universe.requests.get",
        return_value=_MockResponse(SAMPLE_HTML),
    )
    def test_load_sp500_symbols_limit(self, _mock_get):
        symbols = load_sp500_symbols(limit=1)
        self.assertEqual(symbols, ["BRK-B"])


if __name__ == "__main__":
    unittest.main()
