# utils/profiling.py
import cProfile
import io
import pstats
from datetime import datetime
from functools import wraps
from pathlib import Path

from config import OUTPUT_DIR, logger


def profile_function(output_filename=None):
    """
    Decorator om een functie te profileren en de resultaten op te slaan.

    Args:
        output_filename: Optionele bestandsnaam voor het opslaan van resultaten.
                         Als None, wordt een automatische naam gegenereerd.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Genereer automatische bestandsnaam als geen is opgegeven
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"profile_{func.__name__}_{timestamp}.txt"
            else:
                filename = output_filename

            # Zorg ervoor dat de output directory bestaat
            output_path = Path(OUTPUT_DIR) / "profiles"
            output_path.mkdir(exist_ok=True, parents=True)
            profile_path = output_path / filename

            # Run de profiler
            logger.info(f"Profiling {func.__name__}...")
            pr = cProfile.Profile()
            pr.enable()

            result = func(*args, **kwargs)

            pr.disable()

            # Genereer een geformatteerd rapport
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(50)  # Print top 50 tijdrovende functies

            # Schrijf naar bestand en log een kort overzicht
            with open(profile_path, 'w') as f:
                f.write(s.getvalue())

            logger.info(f"Profiling resultaat opgeslagen in {profile_path}")
            logger.info(f"Top 5 tijdrovende functies:")

            # Toon een kort overzicht in de logs
            top_lines = '\n'.join(s.getvalue().split('\n')[:20])
            logger.info(f"\n{top_lines}")

            return result

        return wrapper

    return decorator