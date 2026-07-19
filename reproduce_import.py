try:
    from vop_poc_nz.visualizations import plot_discordance_loss

    print(f"Import successful: {plot_discordance_loss.__name__}")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Other error: {e}")
