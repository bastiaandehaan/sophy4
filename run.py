import os

commands = {
    "1": "python optimization/quick_optimize.py --strategy BollongStrategy",
    "2": "python optimization/optimize.py --strategy BollongStrategy --symbol GER40.cash --timeframe D1",
    "3": "python backtest/extended_backtest.py --strategy BollongStrategy --symbol GER40.cash --params_file results/BollongStrategy_GER40.cash_quick_optim.json",
    "4": "python sophy_commander.py --strategy BollongStrategy --symbols GER40.cash US30.cash"
}

print("Sophy4 Command Launcher")
for key, cmd in commands.items():
    print(f"{key} - {cmd}")

choice = input("Kies een commando (1-4): ")
os.system(commands.get(choice, "echo Ongeldige keuze"))
