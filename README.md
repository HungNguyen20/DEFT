# DEFT
This repository contains code for paper "Causal Discovery via Vertical Partitioning of Data Features" -- under review


### Instruction to create runtime
Following these steps:
1. Make sure you have conda ready
2. Using conda, create an R-supported python environment
   ```bash
   conda create -n rpy r-essentials r-base python=3.10
   ```
3. Activate the environment, and install the following package:
   ```bash
   pip install GPy
   pip install igragh
   pip install cdt bnlearn
   pip install causal-learn gcastle CausalDisco
   ```
4. Pytorch is automatically    installed when you install bnlearn, but it can be of the wrong cuda version if you are using a GPU engine. So, reinstall it from the Pytorch official website if needed.
5. You will find in the legacy folder, the zip file **prepare-r.r**. You either let the content be
   ```Rscript
   install.packages("BiocManager", repos="http://cran.us.r-project.org")
   BiocManager::install('pcalg')
   ```
   or (try this first and the above second if this one fails)
   ```Rscript
   install.packages('pcalg_2.7-12.tar.gz', repos = NULL, type="source")
   ```
   And then run
   ```bash
   Rscript prepare-r.r
   ```
   Make sure to comfirm that the code is successfully processed as followed in the terminal:
   ```bash
    ** installing vignettes
    ** testing if installed package can be loaded from temporary location
    ** checking absolute paths in shared objects and dynamic libraries
    ** testing if installed package can be loaded from final location
    ** testing if installed package keeps a record of temporary installation path
    * DONE (pcalg)
   ```

### Instruction to run code
```bash
bash run_bash.sh
```
The bash file is reading-friendly. Make sure the data exists in data/ and the folder res/ is created in advance.
