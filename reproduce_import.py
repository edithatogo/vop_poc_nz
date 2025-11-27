try:
    from src.visualizations import plot_discordance_loss
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Other error: {e}")
