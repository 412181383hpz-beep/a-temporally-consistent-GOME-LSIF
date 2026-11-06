"""Retrieve GOME-LSIF for 2003-07-01 through 2011-06-30."""
import retrieve_gome_sif_orbit_Mali1_quadratic_S2_SZA75 as workflow


if __name__ == "__main__":
    workflow.RUN_START_DATE = "2003-07-01"
    workflow.RUN_END_DATE = "2011-06-30"
    workflow.main()
