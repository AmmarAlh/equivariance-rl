# equivariance-rl
Exploiting equivaraince in RL and Transfromer architecture




# Mesa-loader error
To solve the problem of the libGL error cased by anaconda "MESA-LOADER: failed to open BACKEND, execute the following procedure"
''' bash
$ cd /home/$USER/miniconda/lib
$ mkdir backup  # Create a new folder to keep the original libstdc++
$ mv libstd* backup  # Put all libstdc++ files into the folder, including soft links
$ cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./ # Copy the c++ dynamic link library of the system here
$ ln -s libstdc++.so.6 libstdc++.so
$ ln -s libstdc++.so.6 libstdc++.so.6.0.19
'''
