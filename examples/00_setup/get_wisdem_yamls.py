import os
import requests
"""
Run this script to update geometry yamls based on WISDEM yamls
"""

def main():

    wisdem_ref_turbs = [
        "09_floating/nrel5mw-spar_oc3.yaml",
        "09_floating/nrel5mw-semi_oc4.yaml",
        "09_floating/IEA-15-240-RWT_VolturnUS-S.yaml",
        "09_floating/IEA-22-280-RWT_Floater.yaml",
        "02_reference_turbines/IEA-15-240-RWT.yaml",
        "02_reference_turbines/IEA-22-280-RWT.yaml",
        "02_reference_turbines/IEA-3p4-130-RWT.yaml",
        "03_blade/BAR_USC.yaml",
    ]

    setup_dir = os.path.dirname(__file__)

    for wisdem_ex in wisdem_ref_turbs:

        raw_url = f"https://raw.githubusercontent.com/WISDEM/WISDEM/master/examples/{wisdem_ex}"

        fname = wisdem_ex.split('/')[-1]
        local_path = os.path.join(setup_dir, "ref_turbines", fname)

        # Download the file
        response = requests.get(raw_url)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {raw_url} to {local_path}")
        else:
            print(f"Failed to download {raw_url}. Status code: {response.status_code}")


if __name__=="__main__":
    main()
