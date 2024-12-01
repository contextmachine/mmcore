#!/bin/bash



# Exit immediately if a command exits with a non-zero status
set -e

# ----------------------------
# Configuration Variables
# ----------------------------

# Repository URL to clone
REPO_URL="https://github.com/LingDong-/skeleton-tracing.git"  # Replace with your repository URL

# Directory name for cloning the repository

REPO_ROOT="skeleton-tracing"
REPO_DIR="$REPO_ROOT/swig"
# Files or directories to copy back to the parent directory after build
# Adjust these paths based on your project's structure
FILES_TO_COPY=(
    "_trace_skeleton.so"
    "example.py"
    "trace_skeleton.py"

    # Add any other necessary files
)

# ----------------------------
# Function Definitions
# ----------------------------

# Function to print error messages to stderr
error() {
    echo "Error: $1" >&2
}

# Function to check if a command exists
check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        error "'$1' is not installed or not in PATH."
        exit 1
    fi
}

# Function to clean up cloned repository
cleanup() {
    echo "Cleaning up cloned repository..."
    cd "$PARENT_DIR"
    rm -rf "$REPO_ROOT"
}

# Trap to ensure cleanup happens on script exit
trap cleanup EXIT

# ----------------------------
# Pre-build Checks
# ----------------------------

echo "Starting the build process..."

# Check for required commands
echo "Checking for required tools..."
check_command "git"
check_command "swig"
check_command "python3"
check_command "python3-config"
check_command "gcc"

# ----------------------------
# Clone the Repository
# ----------------------------

# Check if the repository directory already exists
if [ -d "$REPO_DIR" ]; then
    error "Directory '$REPO_DIR' already exists. Please remove it or choose a different directory."
    exit 1
fi

echo "Cloning the repository from $REPO_URL..."
git clone "$REPO_URL"

# Verify that the repository was cloned successfully
if [ ! -d "$REPO_DIR" ]; then
    error "Failed to clone the repository."
    exit 1
fi

# Save the parent directory path
PARENT_DIR=$(pwd)

# Navigate into the cloned repository
cd "$REPO_DIR"

# ----------------------------
# Build the Library
# ----------------------------

echo "Generating SWIG wrappers..."
swig -python trace_skeleton.i

echo "Retrieving Python compilation and linking flags..."
PYTHON_INCLUDE=$(python3-config --includes)
PYTHON_CFLAGS=$(python3-config --cflags)
PYTHON_LDFLAGS=$(python3-config --ldflags)

echo "Compiling C source files..."
gcc -fPIC -lto -O3 -c trace_skeleton.c trace_skeleton_wrap.c $PYTHON_CFLAGS

# Determine the operating system
OS=$(uname)

echo "Linking object files into a shared library..."
if [ "$OS" = "Darwin" ]; then
    # macOS-specific linking flags
    gcc $PYTHON_LDFLAGS -bundle -undefined dynamic_lookup *.o -o _trace_skeleton.so
elif [[ "$OS" == "Linux" || "$OS" == "FreeBSD" ]]; then
    # Linux and FreeBSD linking flags
    gcc $PYTHON_LDFLAGS -shared *.o -o _trace_skeleton.so
else
    error "Unsupported operating system: $OS"
    exit 1
fi

echo "Cleaning up intermediate files..."
rm -f *.o trace_skeleton_wrap.c

# ----------------------------
# Copy Built Artifacts to Parent Directory
# ----------------------------

echo "Copying necessary files to the parent directory ($PARENT_DIR)..."
for file in "${FILES_TO_COPY[@]}"; do
    if [ -e "$file" ]; then
        cp "$file" "$PARENT_DIR/"
        echo "Copied '$file' to '$PARENT_DIR/'."
    else
        echo "Warning: '$file' does not exist and will not be copied."
    fi
done

# ----------------------------
# Run Example Script (Optional)
# ----------------------------

echo "Running the example script to verify the build..."
python3 -c "import trace_skeleton"

echo "Build and verification completed successfully."

# The 'cleanup' function will be called automatically due to the trap



