## Installation (Ubuntu based distribution)
1. Create a separate folder on your computer.
2. Open the terminal **in this folder** and clone this repository using the following command:

        git clone https://github.com/Raylend/agnprocesses.git
        
* (optional) If you do not have git installed on your computer, you can download it using

        sudo apt install git
        
and then clone this repository.
        
3. Define LD_LIBRARY_PATH OS environment variable. To do so:

    1. Open ~/.bashrc:
     
            sudo nano ~/.bashrc
            
    2. In the end of the opened file add the following line:
    
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'absolute_path_to_the_folder_agnprocesses/bin/shared'
            
    3. Save (Ctrl+O) and close the file (Ctrl+X).
    4. Type:
            
            source ~/.bashrc
            
4. Using conda install astropy, numpy, matplotlib, subprocess, scipy:

        conda install astropy numpy matplotlib subprocess scipy
        
* (optional) For using pgamma.py, bh.py or gamma_gamma.py (this issue is to be fixed in future):
    1. In the processes/c_codes/PhotoHadron/pgamma.cpp file in the 2nd line replace '/home/raylend/anaconda3' with a relevant path to your anaconda3. Do not remove '/include/python3.7m/Python.h'.
    2. In the processes/c_codes/GammaGammaPairProduction/gamma-gamma.cpp file in the 2nd line replace '/home/raylend/anaconda3' with a relevant path to your anaconda3. Do not remove '/include/python3.7m/Python.h'.
    3. In the processes/c_codes/BHPairProduction/bh.cpp file in the 2nd line replace '/home/raylend/anaconda3' with a relevant path to your anaconda3. Do not remove '/include/python3.7m/Python.h'.
    
## Usage
All operations are to be done in a separate .py file. See main.py file as an example of the SSC (synchrotron self Compton) model.

* (optional) If you want to use hadronic part of agnprocesses, uncomment 'import processes.pgamma as pgamma', 'import processes.bh as bh', 'import processes.gamma_gamma as gamma_gamma' to use these modules. In such a case see (optional) instructions above.

## Recommendations
If you still do not have your favorite code editor I recoomend to install [Atom](https://atom.io/). It can be installed using software center in Ubuntu distribution.
