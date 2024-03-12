## 04 Installation on Apple M chip
There are quite a few Python libraries that are not yet compatible with Apple M chipset architecture. The best way to use Basedai on this hardware is through Conda and Miniforge. The Opentensor team has created a Conda environment that makes installing Basedai on these systems very easy. 

> NOTE: This tutorial assumes you have installed conda on mac, if you have not done so already you can install it from [here](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html).

1. Create the conda environment from the `apple_m1_environment.yml` file here:
    ```bash
    conda env create -f apple_m1_environment.yml
    ```

2. Activate the new environment: `conda activate basedai`.
3. Verify that the new environment was installed correctly:
   ```bash
   conda env list
   ```

4. Install basedai (without dependencies):
    ```bash
    conda activate basedai        # activate the env 
    pip install --no-deps basedai # install basedai
    ```