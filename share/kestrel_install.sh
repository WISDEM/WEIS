#!/bin/bash
# Script to install WEIS on Kestrel for a developer with best practices
# To use the script; go to the desired location and run:
# wget https://raw.githubusercontent.com/WISDEM/WEIS/develop/share/kestrel_install.sh -O kestrel_install.sh
# chmod +xrw kestrel_install.sh
# ./kestrel_install.sh -h

# Flags and variables; Set as required for install
weisBranch="main"
weisRepoOverride=""
weisDirName="weis"

openfastBranch="main"
openfastRepoOverride=""
openfastDirName="openfast"

roscoBranch="main"
roscoRepoOverride=""
roscoDirName="rosco"

wisdemBranch="master"
wisdemRepoOverride=""
wisdemDirName="wisdem"

raftBranch="main"
raftRepoOverride=""
raftDirName="raft"


# Defaults
install_openfast_flag=true
install_rosco_flag=true
install_wisdem_flag=false
install_raft_flag=false
Cray=false

### !!!! Avoid changes below this line !!!! ###

# Default repo URLs
default_weis_repo="https://www.github.com/WISDEM/WEIS"
default_openfast_repo="https://www.github.com/OpenFAST/openfast"
default_rosco_repo="https://www.github.com/NREL/ROSCO"
default_wisdem_repo="https://www.github.com/WISDEM/WISDEM"
default_raft_repo="https://www.github.com/WISDEM/RAFT"

# Main function to install WEIS
main() {

    # Check if the script is running on Kestrel
    check_if_kestrel

    # Parse the arguments
    parse_arguments "$@"

    # Display user options
    echo_user_options

    # Load KESTREL modules
    load_modules

    # Install WEIS
    install_weis

    # Install OpenFAST
    if [[ "$install_openfast_flag" == true ]]; then
        install_openfast
    fi

    # Install ROSCO
    if [[ "$install_rosco_flag" == true ]]; then
        if [[ -n "$roscoRepoOverride" ]]; then
            uninstall_reinstall_lib_source "rosco" "$roscoRepoOverride" "$roscoBranch" "$roscoDirName"
        else
            echo "Cloning ROSCO repository to default location...."
            uninstall_reinstall_lib_source "rosco" "$default_rosco_repo" "$roscoBranch" "$roscoDirName"
        fi
    fi

    # Install WISDEM
    if [[ "$install_wisdem_flag" == true ]]; then
        if [[ -n "$wisdemRepoOverride" ]]; then
            uninstall_reinstall_lib_source "wisdem" "$wisdemRepoOverride" "$wisdemBranch" "$wisdemDirName"
        else
            echo "Cloning WISDEM repository to default location...."
            uninstall_reinstall_lib_source "wisdem" "$default_wisdem_repo" "$wisdemBranch" "$wisdemDirName"
        fi
    fi

    # Install RAFT
    if [[ "$install_raft_flag" == true ]]; then
        if [[ -n "$raftRepoOverride" ]]; then
            uninstall_reinstall_lib_source "openraft" "$raftRepoOverride" "$raftBranch" "$raftDirName"
        else
            echo "Cloning RAFT repository to default location...."
            uninstall_reinstall_lib_source "openraft" "$default_raft_repo" "$raftBranch" "$raftDirName"
        fi
    fi

    # Summary
    summary

}

# Function to display help message
help_message() {
    echo ""
    echo "Usage: $0 -n <environment_name> <optional_flags>"
    echo ""
    echo "  This script creates a conda environment for WEIS on Kestrel."
    echo "  Additionally, OpenFAST and ROSCO are compiled using native compiler options."
    echo ""
    echo " >>>> User options can be edited in the script."
    echo " Options include overiding the default repository and branch for WEIS, OpenFAST, ROSCO, WISDEM, and RAFT."
    echo ""
    echo "  Arguments:"
    echo "    -n <name>, --name <name>   The desired name for the conda environment."
    echo ""
    echo "  Optional Flags:"
    echo "    -h, --help                                    Display this help message."
    echo "    -n <name>, --name <name>                      Specify the name for the conda environment."
    echo "    -p <path/to/env>, --path <path/to/env>        Specify the path to install the conda environment." If used with -n, the name will be ignored.
    echo "    -raft, --raft                                 Install RAFT separately."
    echo "    -wisdem, --wisdem                             Install WISDEM separately."
    echo "    -v, --verbose                                 Enable verbose output. -> Not yet implemented"
    echo "    -cray, --cray                                    Use Cray specific flags for OpenFAST build."
    echo ""
    echo "  Example 1:"
    echo "    ./kestrelInstall.sh -n my_weis_env"
    echo ""
    echo "  Example 2:"
    echo "    ./kestrelInstall.sh -p /path/to/env"
    echo ""
    echo "  Example 3:"
    echo "    ./kestrelInstall.sh -p /path/to/env -raft -wisdem"
    echo ""
    exit 1
}

check_if_kestrel() {
    # Check if the script is running on Kestrel
    if [[ "$NREL_CLUSTER" != "kestrel" ]]; then
        echo "Error: This script is intended to run on Kestrel."
        exit 1
    fi
}

# Function to parse arguments
parse_arguments() {
    # Check if at least one argument is provided
    if [[ $# -eq 0 ]]; then
        echo "Error: Please provide a name for the conda environment."
        echo "Run '$0 -h' for help."
        exit 1
    fi


    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                help_message
                ;;
            -n|--name)
                # Capture the next argument (potential name for conda Environment)
                if [[ $# -lt 2 ]]; then  # Check if next argument exists
                    echo "Error: Missing name argument for -n."
                    echo "Run '$0 -h' for help."
                    exit 1
                fi
                conda_env_name="$2"  # Assign the next argument to conda_env_name
                shift # Remove one of the arguments, secons is taken care outside the loop
                ;;
            -p|--path)
                # Capture the next argument (potential path to install conda Environment)
                if [[ $# -lt 2 ]]; then  # Check if next argument exists
                    echo "Error: Missing path argument for -p."
                    echo "Run '$0 -h' for help."
                    exit 1
                fi
                conda_env_path="$2"  # Assign the next argument to conda_env_path
                # TODO: Validate the path
                shift # Remove one of the arguments, secons is taken care outside the loop
                ;;
            -raft|--raft)
                # Set a flag for installing RAFT as seperate step
                install_raft_flag=true
                ;;  
            -wisdem|--wisdem)
                # Set a flag for installing WISDEM as seperate step
                install_wisdem_flag=true
                ;;
            -v|--verbose)
                # Set a flag for verbose output (pseudo code)
                verbose=true
                ;;
            -cray|--cray)
                # Set a flag for Cray specific flags
                Cray=true
                ;;
            -quiet-mode|--quiet-mode)
                # Set a flag for installing everything and without user prompts. Goes Brrrrrr......
                quiet_mode=true
                install_openfast_flag=true
                install_rosco_flag=true
                install_raft_flag=true
                install_wisdem_flag=true
                ;;
            *)
                echo "Error: Unknown argument '$1'."
                echo "Run '$0 -h' for help."
                exit 1
                ;;
        esac
        shift  # Remove the processed argument
    done
}



load_modules() {
    # Load modules for WEIS installation

    if [[ "$Cray" == true ]]; then
        module purge
        module load conda
        module load mamba
        module load craype-x86-spr
        module load libfabric/1.15.2.0
        module load craype-network-ofi
        module load perftools-base/22.09.0
        module load intel/2023.2.0
        module load craype/2.7.17
        module load cray-dsmml/0.2.2
        module load cray-mpich/8.1.23
        module load PrgEnv-intel/8.5.0
        module load hdf5/1.14.1-2-cray-mpich-intel
        module load netcdf-c/4.9.2-cray-mpich-intel
        module load cmake/3.20.2
        module load cray-libsci/22.12.1.1

        # Set compiler environment variables
        export CC=mpicc
        export CXX=mpicxx
        export FC=mpifort
    else

        echo " "
        echo "Loading modules for Kestrel login node $HOSTNAME...."

        module purge

        for mod in conda mamba git cmake craype-x86-spr intel-oneapi-compilers intel-oneapi-mpi intel-oneapi-mkl fftw/3.3.10-intel-oneapi-mpi-intel hdf5/1.14.1-2-intel-oneapi-mpi-intel netcdf-c/4.9.2-intel-oneapi-mpi-intel petsc/3.20.4-intel-oneapi-mpi-intel PrgEnv-intel
        do
                echo "Loading $mod....."
                module load $mod
        done

        echo "Unloading GCC...."
        module unload gcc

        # Set compiler environment variables
        # export CC=icc
        # export CXX=icpc
        # export FC=ifort
        
    fi
}


git_clone_switch_branch() {
    # Clone a git repository and checkout a specific branch
    local repo_url="$1"; 
    local branch_name="$2"; 

    # If the user provides a name for the cloned directory
    if [[ -n "$1" ]]; then
        local cloned_repo_name="$3";
    else
        local cloned_repo_name
        cloned_repo_name="$(basename "$repo_url" .git)"
    fi

    # Check if the repository is already cloned with the same folder name
    check_git_repo_branch "$repo_url" "$branch_name" "$cloned_repo_name"
    if [ $? -ne 0 ]; then # if return is 0, then proceed with the git clone
        echo " "
        user_prompt "Copy of the repository already exists. Proceeding without git clone...."
        return 0
    fi


    # Check if required arguments are provided
    if [[ -z "$repo_url" || -z "$branch_name" ]]; then
        echo "Error: Please provide both the repository URL and branch name."
        echo "Usage: $0 <repo_url> <branch_name>"
        exit 1
    fi

    # Clone the repository
    git clone "$repo_url" "$cloned_repo_name" || {
        echo "Error: Failed to clone repository."
        exit 1
    }

    # Move to the cloned directory (assuming single directory)
    cd "$cloned_repo_name" || {
        echo "Error: Couldn't change directory to the cloned repository."
        exit 1
    }

    # Check if branch exists
    if ! git branch --all | grep "$branch_name"; then
        echo "Error: Branch '$branch_name' does not exist in the repository."
        exit 1
    fi

    # Checkout the desired branch
    git checkout "$branch_name" || {
        echo "Error: Failed to checkout branch '$branch_name'."
        exit 1
    }

    # Return to the original directory
    cd ..

    # Success message
    echo "Successfully cloned repository '$repo_url' and switched to branch '$branch_name'."

}


check_git_repo_branch(){
    # Check if a branch exists in a git repository
    local repo_url="$1"; 
    local branch_name="$2"; 
    local cloned_repo_name="$3";


    # Check if the repository is already cloned with the same folder name, If not return 0 to proceed with the git clone
    if [[ -d "$cloned_repo_name" ]]; then
        echo " "
        echo "Warning 09: Directory '$cloned_repo_name' already exists."

        # check if the folder is the correct git repository
        if [[ "$(git -C "$cloned_repo_name" config --get remote.origin.url)" != "$repo_url" ]]; then
            echo "Warning: Directory '$cloned_repo_name' is not from the correct repository."
            if user_selection "Do you want to proceed with the git clone and overwrite the existing directory?"; then
                echo "Proceeding with the deletion & git clone...."
                rm -rf "$cloned_repo_name"
                return 0 # return 0 to proceed with the git clone
                
            else
                echo "Exiting...."
                exit 1
            fi

        fi

        # check if the folder is in the correct branch
        if [[ "$(git -C "$cloned_repo_name" rev-parse --abbrev-ref HEAD)" != "$branch_name" ]]; then
            echo "Warning: Directory '$cloned_repo_name' is not in the correct branch."
            if user_selection "Do you want to proceed with the git clone and overwrite the existing directory?"; then
                echo "Proceeding with the deletion & git clone...."
                rm -rf "$cloned_repo_name"
                return 0 # return 0 to proceed with the git clone
            else
                echo "Exiting...."
                exit 1
            fi
        fi

        # If the folder is in the correct branch, return 1 to exit the function
        return 1

    else # exit current function and return to calling function with proceed signal
        return 0
    fi

}

install_weis() {

    # Check if WEIS repo overide is provided else use default
    if [[ -n "$weisRepoOverride" ]]; then
        git_clone_switch_branch "$weisRepoOverride" "$weisBranch" "$weisDirName"
    else
        echo "Cloning WEIS repository to default location...."
        git_clone_switch_branch "$default_weis_repo" "$weisBranch" "$weisDirName"
    fi

    user_prompt "Starting to create conda environment for WEIS...."

    # If path is provided instead of env name, use that, this will override the name if provided
    if [[ -n "$conda_env_path" ]]; then
        mamba env create -p "$conda_env_path" -f "$weisDirName/environment.yml"
    else
        mamba env create -n "$conda_env_name" -f "$weisDirName/environment.yml"
    fi

    # Activate the environment & check that we are in the correct environment
    if [[ -n "$conda_env_path" ]]; then
        echo ""
        echo "Activating conda environment at $conda_env_path...."

        source activate "$conda_env_path" || {
            echo "Error: Failed to activate the conda environment."
            exit 1
        }

        # Test if activated correctly -> using realpath to get the full path
        if [[ "$CONDA_DEFAULT_ENV" != "$(realpath "$conda_env_path")" ]]; then
            echo "Error: Not activated in required environment."
            exit 1
        fi

        user_prompt "$conda_env_path activated...."

    else
        echo ""
        echo "Activating conda environment $conda_env_name...."

        source activate "$conda_env_name" || {
            echo "Error: Failed to activate the conda environment <<<< Here."
            exit 1
        }

        # Test if activated correctly
        if [[ "$(basename "$CONDA_DEFAULT_ENV" .git)" != "$conda_env_name" ]]; then
            echo "Error: Not activated in required environment."
            exit 1
        fi

        user_prompt "$conda_env_name activated...."

    fi

    # verify if conda environemnt is activated correctly

    # install additional packages
    mamba install -y petsc4py mpi4py pyoptsparse

    # Install the WEIS package
    cd "$weisDirName" || {
        echo "Error: Couldn't change directory to the cloned WEIS repository."
        exit 1
    }

    pip install -e . || {
        echo "Error: Failed to install WEIS."
        cd ..
        exit 1
    }

    # Return to the original directory
    cd ..

    # Success message
    echo "Successfully installed WEIS."

}

install_openfast() {

    echo " "
    echo ">>>>>>>>>>>>>>>> Installing OpenFAST...."


    # Check if OpenFAST repo overide is provided else use default
    if [[ -n "$openfastRepoOverride" ]]; then
        git_clone_switch_branch "$openfastRepoOverride" "$openfastBranch" "$openfastDirName"
    else
        echo "Cloning OpenFAST repository to default location...."
        git_clone_switch_branch "$default_openfast_repo" "$openfastBranch" "$openfastDirName"
    fi

    # Installing OpenFAST
    cd "$openfastDirName" || {
        echo "Error: Couldn't change directory to the cloned OpenFAST repository."
        exit 1
    }

    mkdir -p build || {
        echo "Error: Failed to create the build directory for OpenFAST."
        cd ..
        exit 1
    }
    
    cd build || {
        echo "Error: Couldn't change directory to the build directory."
        exit 1
    }

    cmake ..\
        -DDOUBLE_PRECISION:BOOL=OFF \
        -DOPENMP=ON \
        -DCMAKE_BUILD_TYPE=Release || {
        echo "Error: Failed to run CMake for OpenFAST."
        cd ../..
        exit 1
    }

    # If user requests Crey specific flags -> This is in addition to the above cmake flags
    if [[ "$Cray" == true ]]; then
        echo "Using Cray specific flags for OpenFAST build...."
        cmake ..\
            -DBLAS_LIBRARIES="${CRAY_LIBSCI_PREFIX_DIR}"/lib/libsci_intel.a \
            -DLAPACK_LIBRARIES="${CRAY_LIBSCI_PREFIX_DIR}"/lib/libsci_intel.a || {
            echo "Error: Failed to run CMake for OpenFAST with Cray specific flags."
            cd ../..
            exit 1
        }
    fi


    make -j 4 || {
        echo "Error: Failed to build OpenFAST."
        cd ../..
        exit 1
    }

    # Install OpenFAST
    make install || {
        echo "Error: Failed to install OpenFAST."
        cd ../..
        exit 1
    }

    # Return to the original directory
    cd ../..

    # Success message
    echo "Successfully installed OpenFAST."
}


uninstall_reinstall_lib_source() {

    echo " "
    echo ">>>>>>>>>>>>>>>> Installing $1...."


    # $1 -> conda lib name
    # $2 -> githb repo
    # $3 -> lib branch
    # $4 -> lib dir name

    # Uninstall a library from the conda environment and install it from source
    local conda_lib_name="$1"; shift
    local lib_repo="$1"; shift
    local lib_branch="$1"; shift
    local lib_dir_name="$1"; shift

    # Pull github repo
    git_clone_switch_branch "$lib_repo" "$lib_branch" "$lib_dir_name" || {
        echo "Error: Failed to clone $lib_repo repository."
        exit 1
    }

    # Uninstall the library from the conda environment
    conda uninstall --force "$conda_lib_name" -y || {
        echo "Error: Failed to uninstall $conda_lib_name from the conda environment."
        exit 1
    }

    # Going into the library directory
    cd "$lib_dir_name" || {
        echo "Error: Couldn't change directory to the cloned $conda_lib_name repository."
        exit 1
    }

    # Install the library
    pip install -e . || {
        echo "Error: Failed to install $conda_lib_name."
        cd ..
        exit 1
    }

    # Return to the original directory
    cd ..

    # Success message
    echo "Successfully installed $conda_lib_name."
}

user_prompt(){

    echo " "
    echo ">>>>>>>>>>>>>>>> $1"
    echo "Press any key to continue... Crtl+C to exit...."
    
    if [[ "$quiet_mode" == true ]]; then
        echo " "
        echo "quiet mode activated. Proceeding without user prompt...., Going Brrrrr...."
        return 0
    fi

    read -n 1 -s keypress
    echo "Continuing..."
    echo " "
}

user_selection(){

    # ask user to decide between yes or no, loop if invalid input

    echo " "
    echo ">>>>>>>>>>>>>>>> $1"
    echo "Press 'y' for Yes or 'n' for No...."

    if [[ "$quiet_mode" == true ]]; then
        echo " "
        echo "quiet mode activated. Proceeding without user prompt...., Going Brrrrr...."
        return 0
    fi

    read -n 1 -s keypress

    while [[ "$keypress" != "y" && "$keypress" != "n" ]]; do
        echo "Invalid input. Press 'y' for Yes or 'n' for No...."
        read -n 1 -s keypress
    done

    if [[ "$keypress" == "y" ]]; then
        return 0
    else
        return 1
    fi
}

summary() {
    echo " "
    echo "*******************************************************"
    echo "WEIS, OpenFAST, ROSCO have been installed successfully."
    echo " "
    if [[ "$install_wisdem_flag" == true ]]; then
        echo "WISDEM has been installed successfully."
    fi
    if [[ "$install_raft_flag" == true ]]; then
        echo "RAFT has been installed successfully."
    fi
    echo ""
    echo "To activate the conda environment, run:"
    if [[ -n "$conda_env_path" ]]; then
        echo "    conda activate $conda_env_path"
    else
        echo "    conda activate $conda_env_name"
    fi
    echo " "
}

echo_user_options() {

    echo " "
    echo "*******************************************************"
    echo "User options:"
    echo " "

    #  Talk about the conda environment
    if [[ -n "$conda_env_path" ]]; then
        echo "Conda environment path: $conda_env_path [Overiding environment name]" 
    else
        echo "Conda environment name: $conda_env_name"
    fi
    echo " "

    # Talk about CRAY
    echo "Cray specific flags: $Cray"
    echo " "
    # Talk about the WEIS installation
    if [[ -n "$weisRepoOverride" ]]; then
        echo "WEIS repository: $weisRepoOverride [Overide]"
    else
        echo "WEIS repository: $default_weis_repo"
    fi
    echo "WEIS branch: $weisBranch"
    echo "WEIS directory name: $weisDirName"
    echo " "

    # Talk about the OpenFAST installation
    if [[ "$install_openfast_flag" == true ]]; then
        if [[ -n "$openfastRepoOverride" ]]; then
            echo "OpenFAST repository: $openfastRepoOverride [Overide]"
        else
            echo "OpenFAST repository: $default_openfast_repo"
        fi
        echo "OpenFAST branch: $openfastBranch"
        echo "OpenFAST directory name: $openfastDirName"
        
    else
        echo "Install OpenFAST: No"
    fi
    echo " "

    # Talk about the ROSCO installation
    if [[ "$install_rosco_flag" == true ]]; then
        if [[ -n "$roscoRepoOverride" ]]; then
            echo "ROSCO repository: $roscoRepoOverride [Overide]"
        else
            echo "ROSCO repository: $default_rosco_repo"
        fi
        echo "ROSCO branch: $roscoBranch"
        echo "ROSCO directory name: $roscoDirName"
    else
        echo "Install ROSCO: No"
    fi
    echo " "

    # Talk about the WISDEM installation
    if [[ "$install_wisdem_flag" == true ]]; then
        if [[ -n "$wisdemRepoOverride" ]]; then
            echo "WISDEM repository: $wisdemRepoOverride [Overide]"
        else
            echo "WISDEM repository: $default_wisdem_repo"
        fi
        echo "WISDEM branch: $wisdemBranch"
        echo "WISDEM directory name: $wisdemDirName"
    else
        echo "Install WISDEM: No"
    fi
    echo " "

    # Talk about the RAFT installation
    if [[ "$install_raft_flag" == true ]]; then
        if [[ -n "$raftRepoOverride" ]]; then
            echo "RAFT repository: $raftRepoOverride [Overide]"
        else
            echo "RAFT repository: $default_raft_repo"
        fi
        echo "RAFT branch: $raftBranch"
        echo "RAFT directory name: $raftDirName"
    else
        echo "Install RAFT: No"
    fi
    echo " "

    echo "*******************************************************"

    user_prompt "Review the options above...."
}

main "$@"; exit 0
