import ingestion
import preproc
import eda
import garch
import os

print("--- Running Ingestion ---")
ingestion.download_assets()

print("--- Running Preprocessing ---")
preproc.convert_all_to_inr()
preproc.check_calendar_alignment()
preproc.keep_common_dates()

print("--- Running EDA / Crash Detection ---")
eda.detect_crypto_crashes(plot=False)

print("--- Building Unified Matrix ---")
preproc.build_unified_matrix(save=True)

print("--- Running GARCH & DCC ---")
garch.run_univariate_garch(save=True)
garch.run_dcc_garch(save=True)
