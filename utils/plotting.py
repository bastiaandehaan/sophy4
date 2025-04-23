# utils/plotting.py
import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt


def create_visualizations(pf: vbt.Portfolio, strategy_name: str, symbol: str,
                          timeframe: Optional[Union[str, int]], output_path: Path, timestamp: str) -> None:
    """
    Maakt visualisaties voor een backtest en slaat deze op.

    Args:
        pf: Portfolio object van vectorbt
        strategy_name: Naam van de strategie
        symbol: Handelssymbool
        timeframe: Timeframe als string of int
        output_path: Map waar visualisaties worden opgeslagen
        timestamp: Tijdstempel voor de bestandsnaam
    """
    try:
        # Voeg bestandnamen toe met timeframe
        timeframe_str = f"_{timeframe}" if timeframe else ""
        file_prefix = f"{strategy_name}_{symbol}{timeframe_str}_{timestamp}"

        # Maak basisplot van rendement
        fig, ax = plt.subplots(figsize=(12, 7))
        pf.plot(ax=ax)
        ax.set_title(f"{strategy_name} op {symbol} - Cumulatief Rendement")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(output_path / f"{file_prefix}_returns.png")
        plt.close(fig)

        # Maak drawdown plot als er drawdowns zijn
        if not pf.drawdowns.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            pf.plot_drawdowns(top_n=5, ax=ax)
            ax.set_title(f"{strategy_name} op {symbol} - Top 5 Drawdowns")
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(output_path / f"{file_prefix}_drawdowns.png")
            plt.close(fig)

        # Maak een heatmap van trades als er trades zijn
        if len(pf.trades) > 0:
            fig, ax = plt.subplots(figsize=(12, 7))
            pf.trades.plot_pnl(ax=ax)
            ax.set_title(f"{strategy_name} op {symbol} - Trades PnL")
            fig.tight_layout()
            fig.savefig(output_path / f"{file_prefix}_trades_pnl.png")
            plt.close(fig)

    except Exception as e:
        print(f"Fout bij maken visualisaties: {str(e)}")