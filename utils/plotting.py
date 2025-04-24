# utils/plotting.py
import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt


def create_visualizations(pf: vbt.Portfolio, strategy_name: str, symbol: str,
                          timeframe: Optional[Union[str, int]], output_path: Path,
                          timestamp: str) -> None:
    """
    Maakt visualisaties voor een backtest en slaat deze op.
    Compatibel met verschillende VectorBT versies.
    """
    try:
        # Voeg bestandnamen toe met timeframe
        timeframe_str = f"_{timeframe}" if timeframe else ""
        file_prefix = f"{strategy_name}_{symbol}{timeframe_str}_{timestamp}"

        # Detecteer of we met een Plotly-gebaseerde of Matplotlib-gebaseerde VectorBT te maken hebben
        is_plotly_based = False
        try:
            # Controleer of dit een Plotly figuur retourneert (nieuwere VectorBT versies)
            test_fig = pf.plot()
            if hasattr(test_fig, 'layout') and hasattr(test_fig.layout, 'mapbox'):
                is_plotly_based = True
                # Als dit een Plotly figuur is, save direct als afbeelding
                test_fig.write_image(output_path / f"{file_prefix}_returns.png")
                print(
                    "Gedetecteerd: Plotly-gebaseerde VectorBT versie, direct Plotly export gebruiken")
        except Exception:
            # Als er een error is, gaan we ervan uit dat het een Matplotlib-gebaseerde versie is
            is_plotly_based = False

        if not is_plotly_based:
            # Originele Matplotlib-gebaseerde aanpak voor oudere versies
            fig, ax = plt.subplots(figsize=(12, 7))
            try:
                # Probeer met ax parameter voor Matplotlib
                pf.plot(ax=ax)
            except Exception as e:
                print(
                    f"Standaard plot methode niet ondersteund: {str(e)}, gebruik fallback")
                # Eenvoudige fallback plot met Matplotlib
                portfolio_value = pf.value()
                ax.plot(portfolio_value.index, portfolio_value, label='Portfolio Value')
                ax.set_ylabel('Value')

            ax.set_title(f"{strategy_name} op {symbol} - Cumulatief Rendement")
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(output_path / f"{file_prefix}_returns.png")
            plt.close(fig)

            # Maak drawdown plot als er drawdowns zijn
            if not pf.drawdowns.empty:
                fig, ax = plt.subplots(figsize=(12, 7))
                try:
                    pf.plot_drawdowns(top_n=5, ax=ax)
                except Exception as e:
                    print(
                        f"Drawdown plot methode niet ondersteund: {str(e)}, gebruik fallback")
                    # Eenvoudige drawdown plot
                    if hasattr(pf.drawdowns, 'drawdown'):
                        drawdown_series = pf.drawdowns.drawdown
                        ax.plot(drawdown_series.index, drawdown_series, color='red')
                        ax.set_ylabel('Drawdown %')
                        ax.fill_between(drawdown_series.index, 0, drawdown_series,
                                        color='red', alpha=0.3)

                ax.set_title(f"{strategy_name} op {symbol} - Top 5 Drawdowns")
                ax.grid(True)
                fig.tight_layout()
                fig.savefig(output_path / f"{file_prefix}_drawdowns.png")
                plt.close(fig)

            # Maak een heatmap van trades als er trades zijn
            if len(pf.trades) > 0:
                fig, ax = plt.subplots(figsize=(12, 7))
                try:
                    pf.trades.plot_pnl(ax=ax)
                except Exception as e:
                    print(
                        f"Trade PnL plot methode niet ondersteund: {str(e)}, gebruik fallback")
                    # Eenvoudige trades visualisatie
                    if hasattr(pf.trades, 'pnl'):
                        trade_pnls = pf.trades.pnl
                        ax.bar(range(len(trade_pnls)), trade_pnls)
                        ax.axhline(y=0, color='r', linestyle='-')
                        ax.set_xlabel('Trade #')
                        ax.set_ylabel('PnL')

                ax.set_title(f"{strategy_name} op {symbol} - Trades PnL")
                fig.tight_layout()
                fig.savefig(output_path / f"{file_prefix}_trades_pnl.png")
                plt.close(fig)
        else:
            # Voor Plotly-gebaseerde versies, gebruik directe export functies
            try:
                # Drawdowns plot
                if not pf.drawdowns.empty:
                    drawdown_fig = pf.plot_drawdowns(top_n=5)
                    drawdown_fig.update_layout(
                        title=f"{strategy_name} op {symbol} - Top 5 Drawdowns")
                    drawdown_fig.write_image(
                        output_path / f"{file_prefix}_drawdowns.png")

                # Trades plot
                if len(pf.trades) > 0:
                    trade_fig = pf.trades.plot_pnl()
                    trade_fig.update_layout(
                        title=f"{strategy_name} op {symbol} - Trades PnL")
                    trade_fig.write_image(output_path / f"{file_prefix}_trades_pnl.png")
            except Exception as e:
                print(f"Plotly export mislukt: {str(e)}, gebruik Matplotlib fallback")
                # Als Plotly export mislukt, val terug op Matplotlib methode
                portfolio_value = pf.value()
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(portfolio_value.index, portfolio_value, label='Portfolio Value')
                ax.set_ylabel('Value')
                ax.set_title(f"{strategy_name} op {symbol} - Cumulatief Rendement")
                ax.grid(True)
                fig.tight_layout()
                fig.savefig(output_path / f"{file_prefix}_returns.png")
                plt.close(fig)

    except Exception as e:
        print(f"Fout bij maken visualisaties: {str(e)}")