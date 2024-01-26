import os
import requests
'''
Run this script to update geometry yamls based on WISDEM yamls
'''

def main():

    wisdem_weis_map = {
        # relative to examples/
        # WISDEM example input                WEIS example
        "09_floating/nrel5mw-spar_oc3.yaml":            "03_NREL5MW_OC3_spar/nrel5mw-spar_oc3.yaml",
        "09_floating/nrel5mw-semi_oc4.yaml":            "04_NREL5MW_OC4_semi/nrel5mw-semi_oc4.yaml",
        "09_floating/IEA-15-240-RWT_VolturnUS-S.yaml":  "06_IEA-15-240-RWT/IEA-15-240-RWT_VolturnUS-S.yaml",
        "02_reference_turbines/IEA-15-240-RWT.yaml":    "06_IEA-15-240-RWT/IEA-15-240-RWT_Monopile.yaml",
        "02_reference_turbines/IEA-3p4-130-RWT.yaml":   "05_IEA-3.4-130-RWT/IEA-3p4-130-RWT.yaml",
    }

    example_dir = os.path.join(os.path.dirname(__file__),"..")

    for wisdem_ex, weis_ex in wisdem_weis_map.items():

        raw_url = f"https://raw.githubusercontent.com/WISDEM/WISDEM/master/examples/{wisdem_ex}"

        local_path = os.path.join(example_dir,weis_ex)

        # Download the file
        response = requests.get(raw_url)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {raw_url} to {local_path}")
        else:
            print(f"Failed to download {raw_url}. Status code: {response.status_code}")


if __name__=="__main__":
    main()